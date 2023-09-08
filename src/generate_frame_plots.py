#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import jax
import jax.numpy as jnp
import h5py
import matplotlib.pyplot as plt
import pathlib
from eval import load_network
from train import load_model_params, determine_required_fields, determine_channel_size
from online_ensemble_compare import make_ensemble_net, SYS_INFO_CHANNELS, make_ke_time_computer, ke_spec
from cascaded_online_eval import make_net_param_func
import cascaded_eval
import dataclasses
import jax_utils
import re
import itertools
from systems.qg import utils as qg_utils, coarsen, spectral
import pyqg_jax
import functools
from generate_data import CONFIG_VARS
from ipywidgets import interact
import matplotlib


# In[3]:


matplotlib.use('agg')


# In[4]:

import argparse
parser = argparse.ArgumentParser(description="Generate plots")
parser.add_argument("run_type", type=str, default="eddyonly")
parser.add_argument("size", type=int, default=100)
parser.add_argument("scale", type=int, default=48)


args = parser.parse_args()
RUN_TYPE = args.run_type
SIZE = args.size
SCALE = args.scale


# In[5]:


DATA_PATH = f"/mnt/ceph/users/kotness/closure/data-{RUN_TYPE}/test/op1/data.hdf5"


# # Load Networks

# In[6]:


LoadedNetwork = dataclasses.make_dataclass("LoadedNetwork", ["net", "net_info", "net_data", "net_path", "model_params"])


# In[7]:


NET_PATHS = [
    f"/mnt/ceph/users/kotness/closure/run_outputs/run-varied-data-size-20230629-173901/{RUN_TYPE}/size{SIZE:d}-scale{SCALE:d}/net{net_id}"
    for net_id in range(3)
]
loaded_nets = []
for net_path in map(pathlib.Path, NET_PATHS):
    net, net_info = load_network(net_path / "weights" / "best_loss.eqx")
    loaded_nets.append(
        (
            net_path.name,
            LoadedNetwork(
                net=net,
                net_info=net_info,
                net_data=cascaded_eval.NetData(
                    input_channels=net_info['input_channels'],
                    output_channels=net_info["output_channels"],
                    processing_size=net_info["processing_size"]
                ),
                net_path=net_path,
                model_params=load_model_params(net_info["train_path"], eval_path=DATA_PATH),
            )
        )
    )
# Add ensemble net
loaded_nets.append(
    (
        "ensemble",
        LoadedNetwork(
            net=make_ensemble_net([net for name, net in loaded_nets]),
            net_info=loaded_nets[0][1].net_info,
            net_data=loaded_nets[0][1].net_data,
            net_path="ensemble net",
            model_params=loaded_nets[0][1].model_params,
        ),
    )
)
# Add null net
null_net_info = loaded_nets[0][1].net_info.copy()
null_net_info["input_channels"] = list(filter(lambda c: not re.match(SYS_INFO_CHANNELS, c), loaded_nets[0][1].net_info["input_channels"]))
null_net_data = cascaded_eval.NetData(
    input_channels=null_net_info["input_channels"],
    output_channels=null_net_info["output_channels"],
    processing_size=null_net_info["processing_size"],
)
loaded_nets.append(
    (
        "null",
        LoadedNetwork(
            net=lambda chunk: jnp.zeros_like(chunk),
            net_info=null_net_info,
            net_data=null_net_data,
            net_path="null net",
            model_params=loaded_nets[0][1].model_params,
        ),
    )
)
loaded_nets = {k: v for k, v in loaded_nets}


# In[8]:


required_channels = determine_required_fields(itertools.chain.from_iterable(v.net_data.input_channels for v in loaded_nets.values()))


# # Prepare Trajectory Step Function

# In[9]:


def make_step_traj(nets, num_steps=87400, subsample=20, num_warmup_steps=43200, dt=3600.0):
    assert "reference" not in nets
    base_net = next(iter(nets.values()))
    output_size = determine_channel_size(base_net.net_data.output_channels[0])
    big_model = base_net.model_params.qg_models["big_model"]
    small_model = base_net.model_params.qg_models[output_size]
    coarsen_cls = coarsen.COARSEN_OPERATORS[base_net.net_info["coarse_op_name"]]
    coarsener = coarsen_cls(big_model, output_size)

    def get_stepped_model_for_net(name, sys_params={}):
        model = big_model if name == "reference" else small_model
        model_params = qg_utils.qg_model_to_args(model)
        model_params.update(sys_params)
        model = pyqg_jax.qg_model.QGModel(**model_params)
        if name == "reference":
            param_model = pyqg_jax.parameterizations.noop.apply_parameterization(model)
        else:
            net = nets[name]
            expanded_sys_params = {k: jnp.full(shape=(1, 1, 1), fill_value=v) for k, v in sys_params.items()}
            param_model = pyqg_jax.parameterizations.ParameterizedModel(
                model,
                functools.partial(
                    make_net_param_func(
                        [net.net],
                        [net.net_data],
                        net.model_params,
                    ),
                    sys_params=expanded_sys_params,
                )
            )
        stepped_model = pyqg_jax.steppers.SteppedModel(
            param_model,
            pyqg_jax.steppers.AB3Stepper(dt=dt)
        )
        return stepped_model

    def pack_state(name, state):
        model = get_stepped_model_for_net(name)
        return model.initialize_stepper_state(
            model.model.initialize_param_state(state)
        )

    def single_step(state_dict, sys_params):
        return {k: get_stepped_model_for_net(k, sys_params=sys_params).step_model(v) for k, v in state_dict.items()}

    def warmup_scan(carry, _x, sys_params):
        new_carry = single_step(carry, sys_params)
        return new_carry, None

    def main_scan(carry, _x, sys_params):
        new_carry = single_step(carry, sys_params)
        out_steps = {k: v.state.model_state.q for k, v in new_carry.items()}
        out_steps["reference"] = coarsener.coarsen(out_steps["reference"])
        return new_carry, out_steps

    def step_traj(rng, sys_params):
        sys_params = sys_params.copy()
        for k in ["rek", "beta", "delta"]:
            sys_params.setdefault(k, getattr(big_model, k))
        # Take warmup steps
        state_dict, _ = jax.lax.scan(
            functools.partial(warmup_scan, sys_params=sys_params),
            {"reference": pack_state("reference", big_model.create_initial_state(rng))},
            None,
            length=num_warmup_steps
        )
        # Pack initial states
        big_init = state_dict["reference"].state.model_state
        small_init = small_model.create_initial_state(jax.random.PRNGKey(0)).update(q=coarsener.coarsen(big_init.q))
        small_packed = pack_state(next(iter(nets.keys())), small_init)
        state_dict = {
            "reference": pack_state("reference", big_init)
        }
        state_dict.update({k: pack_state(k, small_init) for k in nets.keys()})
        # Take main sequence steps
        _state_dict, traj_q = jax_utils.strided_scan(functools.partial(main_scan, sys_params=sys_params), state_dict, None, length=num_warmup_steps, stride=subsample)
        return traj_q

    return step_traj


# In[10]:


step_fn = jax.jit(make_step_traj(loaded_nets, subsample=20))


# In[11]:


eddy_trajs = {k: np.asarray(v) for k, v in step_fn(jax.random.PRNGKey(100), CONFIG_VARS["jet"]).items()}


# In[12]:


time_ke_computer = jax.jit(make_ke_time_computer(loaded_nets["null"].model_params.qg_models[eddy_trajs["null"].shape[-1]]))
ke_times = {k: np.asarray(time_ke_computer(v)) for k, v in eddy_trajs.items()}


# In[13]:


ke_spec_computer = jax.jit(jax_utils.chunked_vmap(functools.partial(ke_spec, small_model=loaded_nets["null"].model_params.qg_models[eddy_trajs["null"].shape[-1]]), 100))
ke_spec_values = {k: np.asarray(ke_spec_computer(v)) for k, v in eddy_trajs.items()}


# In[14]:


@jax.jit
def calc_ispec_fn(mean_kes):
    return jax.vmap(spectral.calc_ispec, in_axes=[None, 0], out_axes=(None, 0))(loaded_nets["null"].model_params.qg_models[eddy_trajs["null"].shape[-1]], mean_kes)


# In[15]:


def viz(t):
    net_names = ["reference", "null", "net0", "net1", "net2", "ensemble"]
    fig = plt.figure(layout="tight", figsize=(15, 9))
    gs = matplotlib.gridspec.GridSpec(4, 6, figure=fig)

    axs = np.array([fig.add_subplot(gs[i, j]) for i, j in itertools.product(range(2), range(6))])
    axs = axs.reshape((2, 6))
    for layer in range(2):
        for ni, net_name in enumerate(net_names):
            ax = axs[layer, ni]
            if layer == 0:
                ax.set_title(net_name)
            data = eddy_trajs[net_name][t, layer]
            ax.imshow(data)

    spec_ax = [fig.add_subplot(gs[-2:, :2]), fig.add_subplot(gs[-2:, 2:4])]
    net_styles = ["--", ":", "-."]
    for net_name in sorted(net_names):
        if net_name == "reference":
            color = "C0"
            linestyle = "-"
        elif net_name == "null":
            color = "C3"
            linestyle = "-"
        elif net_name == "ensemble":
            color = "C2"
            linestyle = "-"
        else:
            color = "C1"
            linestyle = net_styles.pop()
        data = np.mean(ke_spec_values[net_name][:t], axis=0)
        kr, ke_spec_vals = calc_ispec_fn(data)
        kr = np.asarray(kr)
        ke_spec_vales = np.asarray(ke_spec_vals)
        for layer in range(2):
            spec_ax[layer].loglog(kr, ke_spec_vals[layer], color=color, linestyle=linestyle, label=net_name)
    spec_ax[0].legend()
    spec_ax[0].grid(True)
    spec_ax[1].grid(True)
    spec_ax[0].set_ylim(10**-2, 10**2.5)
    spec_ax[1].set_ylim(10**-4, 10**2)


    ax = fig.add_subplot(gs[-2:, -2:])
    net_styles = ["--", ":", "-."]
    for net_name in sorted(net_names):
        if net_name == "reference":
            color = "C0"
            linestyle = "-"
        elif net_name == "null":
            color = "C3"
            linestyle = "-"
        elif net_name == "ensemble":
            color = "C2"
            linestyle = "-"
        else:
            color = "C1"
            linestyle = net_styles.pop()
        ax.plot(ke_times[net_name], label=net_name, color=color, linestyle=linestyle)
        ax.legend()
        ax.grid(True)
        ax.axvline(t, linestyle=":", color="black")

    fig.tight_layout()
    return fig


# In[ ]:


size_scale = loaded_nets["net0"].net_path.parent.name
net_type = loaded_nets["net0"].net_path.parent.parent.name

out_dir = pathlib.Path(f"sample_size{SIZE:d}-scale{SCALE:d}_{RUN_TYPE}")
out_dir.mkdir(exist_ok=True)
for frame, step in enumerate(range(4, eddy_trajs["reference"].shape[0] - 1, 4), start=1):
    print(f"frame {frame}")
    file_name = f"frame{frame:06d}.png"
    fig = viz(step)
    fig.savefig(out_dir / file_name, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")
print("done!")
