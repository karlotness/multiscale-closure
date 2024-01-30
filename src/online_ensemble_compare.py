import jax
import jax.numpy as jnp
import numpy as np
import jax_utils
import utils
import json
import functools
import math
import itertools
import pathlib
import logging
import sys
import re
import argparse
import dataclasses
import cascaded_eval
from cascaded_online_eval import make_parameterized_stepped_model
from eval import load_network, make_network_evaluator
from train import load_model_params, make_basic_coarsener, determine_required_fields, determine_channel_size
from systems.qg import utils as qg_utils, loader, spectral
import pyqg_jax
import matplotlib.pyplot as plt
import matplotlib

DATA_SUBSAMPLE_FACTOR = 8
SYS_INFO_CHANNELS = r"^(?:ejnorm_)?(rek|beta|delta)_\d+$"

parser = argparse.ArgumentParser(description="Compare offline and online metrics for individual networks vs ensemble")
parser.add_argument("out_dir", type=str, help="Directory to store evaluation results")
parser.add_argument("eval_set", type=str, help="Directory with evaluation samples")
parser.add_argument("nets", type=str, nargs="+", help="Trained network directory")
parser.add_argument("--net_type", type=str, default="best_loss", help="Trained network directory")
parser.add_argument("--dt", type=float, default=3600.0, help="Time step size")
parser.add_argument("--rollout_subsample", type=int, default=DATA_SUBSAMPLE_FACTOR, help="Stride to use when rolling out simulations")
parser.add_argument("--t_metric_start", type=float, default=155520000.0, help="Time at which we should start averaging")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])

LoadedNetwork = jax_utils.register_pytree_dataclass(
    dataclasses.make_dataclass("LoadedNetwork", ["net", "net_info", "net_data", "net_path", "model_params"])
)

def make_ensemble_net(nets):
    def ensemble_net(chunk):
        results = []
        for net_res in nets:
            results.append(net_res.net(chunk))
        results = jnp.stack(results, axis=0)
        return jnp.mean(results, axis=0)
    return ensemble_net


def compute_ke(state, model):
    full_state = model.get_full_state(state)
    return (full_state.u**2 + full_state.v**2) / 2


def delta_weighted_average(var, model):
    delta_weights = jnp.expand_dims(jnp.array([model.del1, model.del2]), axis=(-1, -2))
    return jnp.sum(var * delta_weights, axis=-3)


def KE_time(state, model):
    return jnp.mean(delta_weighted_average(compute_ke(state, model), model), axis=(-1, -2))


def make_ke_time_computer(model):
    def ke_computer(steps):
        packed_steps = jax.vmap(lambda s: model.create_initial_state(jax.random.key(0)).update(q=s))(steps)
        return jax_utils.chunked_vmap(lambda step: KE_time(step, model), 100)(packed_steps)
    return ke_computer


def ke_spec(q, small_model):
    state = small_model.create_initial_state(jax.random.PRNGKey(0)).update(q=q)
    full_state = small_model.get_full_state(state)
    return small_model.wv2 * jnp.abs(full_state.ph)**2 / small_model.M**2


def var_is_real(var):
    return var.shape[-1] == var.shape[-2]

def compute_real(var):
    if var_is_real(var):
        return var
    else:
        return pyqg_jax.state._generic_irfftn(var)


def compute_compl(var):
    if var_is_real(var):
        return pyqg_jax.state._generic_rfftn(var)
    else:
        return var

def compute_ddyh(f, model):
    return model.il * compute_compl(f)

def compute_ddxh(f, model):
    return model.ik * compute_compl(f)

def compute_curlh(x, y, model):
    return compute_ddxh(y, model) - compute_ddyh(x, model)

def compute_curl(a, b, model):
    return compute_real(compute_curlh(a, b, model))

def compute_relative_vorticity(state, model):
    full_state = model.get_full_state(state)
    return compute_curl(full_state.u, full_state.v, model)

def compute_ens(state, model):
    return 0.5 * (compute_relative_vorticity(state, model))**2

def make_ens_computer(model):
    def ens_computer(steps):
        packed_steps = jax.vmap(lambda s: model.create_initial_state(jax.random.key(0)).update(q=s))(steps)
        return jax.vmap(lambda step: compute_ens(step, model))(packed_steps)
    return ens_computer


def pdf_histogram(x, xmin=None, xmax=None, n_bins=30):
    N = x.shape[0]
    mean = jnp.mean(x)
    sigma = jnp.std(x)
    if xmin is None:
        xmin = mean - 4 * sigma
    if xmax is None:
        xmax = mean + 4 * sigma
    bandwidth = (xmax - xmin) / n_bins
    hist, bin_edges = jnp.histogram(x, range=(xmin, xmax), bins=n_bins)
    density = hist / N / bandwidth
    points = (bin_edges[0:-1] + bin_edges[1:]) * 0.5
    return points, density


def make_pdf_var(var_name, level, model):
    xmax_values = {
        "Ens": (1e-10, 1.5e-12),
        "KE": (1.5e-2, 5e-4),
    }

    def make_var(state):
        match var_name:
            case "KE":
                return jax.vmap(compute_ke, in_axes=(0, None))(state, model)
            case "Ens":
                return jax.vmap(compute_ens, in_axes=(0, None))(state, model)
            case "q" | "qh":
                return getattr(state, var_name)
            case _:
                return getattr(model.get_full_state(state), var_name)

    def compute_pdf(steps):
        state = jax.vmap(lambda s: model.create_initial_state(jax.random.key(0)).update(q=s))(steps)
        values = make_var(state)
        xmin = 0 if var_name in {"KE", "Ens"} else None
        xmax = xmax_values.get(var_name, (None, None))[level]
        points, density = pdf_histogram(values.ravel(), xmin=xmin, xmax=xmax)
        return points, density

    return compute_pdf



def make_compute_offline_net_stats(loaded_network, small_size):
    # Only for individual networks vs a baseline
    time_axis = -4

    def level_mse(est, true):
        scaler = loaded_network.model_params.scalers.q_total_forcing_scalers[small_size]
        est = scaler.scale_to_standard(est)
        true = scaler.scale_to_standard(true)
        return jnp.mean((est - true)**2, axis=(-4, -2, -1))

    def pearson_corr_spatial(est, true):
        # Dim for both [batch, level, x, y]
        # Code based on xarray implementation
        demean_est = est - jnp.mean(est, axis=time_axis)
        demean_true = true - jnp.mean(true, axis=time_axis)
        cov = jnp.mean(demean_true.conj() * demean_est, axis=time_axis)
        return cov / (jnp.std(true, axis=time_axis) * jnp.std(est, axis=time_axis))

    def r2_spatial(est, true):
        # Dim for both [batch, level, x, y]
        err = jnp.mean((est - true)**2, axis=time_axis)
        base = jnp.var(true, axis=time_axis)
        return 1 - err / base

    def make_chunk_evaluator(net_eval):

        def do_batch(batch):
            expanded = jax.tree_util.tree_map(lambda a: jnp.expand_dims(a, 0), batch)
            result = net_eval(expanded)
            return jax.tree_util.tree_map(lambda a: jnp.squeeze(a, 0), result)

        return do_batch

    def compute_stats(batch):
        network_evaluator = make_network_evaluator(
            net=loaded_network.net,
            net_info=loaded_network.net_info,
            model_params=loaded_network.model_params,
        )
        evaluated = jax_utils.chunked_vmap(make_chunk_evaluator(network_evaluator), 100)(batch)
        est = evaluated[f"q_total_forcing_{small_size}"]
        true = batch.q_total_forcings[small_size]
        # Compute r2 and correlation coefficient
        r2_val_spatial = r2_spatial(est=est, true=true)
        corr_val_spatial = pearson_corr_spatial(est=est, true=true)
        mse_val_level = level_mse(est=est, true=true)
        return {
            "r2_spatial": r2_val_spatial,
            "corr_spatial": corr_val_spatial,
            "mse_level": mse_val_level,
        }

    return compute_stats


def get_net_style(net_type):
    match net_type:
        case "net":
            return {"color": "C1", "linestyle": "--"}
        case "null":
            return {"color": "C3", "linestyle": "-"}
        case "ensemble":
            return {"color": "C2", "linestyle": "-"}
        case "ref":
            return {"color": "C0", "linestyle": "-"}
        case _:
            raise ValueError(f"unknown net type {net_type}")


def main():
    args = parser.parse_args()
    # Prepare our particular output directory
    out_dir = pathlib.Path(args.out_dir) / args.net_type
    # Make our directory, but only the last two levels (we don't want to create net_dir if it doesn't exist)
    out_dir.mkdir(exist_ok=True, parents=True)
    # Set up logging
    utils.set_up_logging(level=args.log_level, out_file=out_dir/"run.log")
    logger = logging.getLogger("main")
    logger.info("Arguments: %s", vars(args))
    git_info = utils.get_git_info(base_logger=logger)
    if git_info is not None:
        logger.info(
            "Running on commit %s (%s worktree)",
            git_info.hash,
            "clean" if git_info.clean_worktree else "dirty"
        )
    if not utils.check_environment_variables(base_logger=logger):
        sys.exit(1)
    logger.info("Saving results to: %s", out_dir)
    # Find the training data file
    eval_file = pathlib.Path(args.eval_set) / "data.hdf5"
    # Load the networks
    loaded_nets = []
    logger.info("Loading networks")
    for net_path in map(pathlib.Path, args.nets):
        logger.info("Loading network %s", net_path)
        net, net_info = load_network(net_path / "weights" / f"{args.net_type}.eqx")
        loaded_nets.append(
            LoadedNetwork(
                net=net,
                net_info=net_info,
                net_data=cascaded_eval.NetData(
                    input_channels=net_info['input_channels'],
                    output_channels=net_info["output_channels"],
                    processing_size=net_info["processing_size"]
                ),
                net_path=net_path,
                model_params=load_model_params(net_info["train_path"], eval_path=eval_file),
            )
        )
    # Build parameterized models and ensemble
    logger.info("Constructing parameterized models for individual networks")
    individual_net_models = []
    for ln in loaded_nets:
        small_size = determine_channel_size(ln.net_info["output_channels"][0])
        individual_net_models.append(
            make_parameterized_stepped_model(
                nets=[ln.net],
                net_data=[ln.net_data],
                model_params=ln.model_params,
                qg_model_args=qg_utils.qg_model_to_args(ln.model_params.qg_models[small_size]),
                dt=args.dt,
            )
        )
    logger.info("Constructing parameterized model for ensemble")
    ensemble_loaded_network = LoadedNetwork(
        net=make_ensemble_net(loaded_nets),
        net_info=loaded_nets[0].net_info,
        net_data=loaded_nets[0].net_data,
        net_path="ensemble net",
        model_params=loaded_nets[0].model_params,
    )
    ensemble_model = make_parameterized_stepped_model(
        nets=[ensemble_loaded_network.net],
        net_data=[ensemble_loaded_network.net_data],
        model_params=ensemble_loaded_network.model_params,
        qg_model_args=qg_utils.qg_model_to_args(ensemble_loaded_network.model_params.qg_models[small_size]),
        dt=args.dt,
    )
    null_net_info = loaded_nets[0].net_info.copy()
    null_net_info["input_channels"] = list(filter(lambda c: not re.match(SYS_INFO_CHANNELS, c), loaded_nets[0].net_info["input_channels"]))
    null_net_data = cascaded_eval.NetData(
        input_channels=null_net_info["input_channels"],
        output_channels=null_net_info["output_channels"],
        processing_size=null_net_info["processing_size"],
    )
    null_loaded_network = LoadedNetwork(
        net=lambda chunk: jnp.zeros_like(chunk),
        net_info=null_net_info,
        net_data=null_net_data,
        net_path="null net",
        model_params=loaded_nets[0].model_params,
    )
    null_model = make_parameterized_stepped_model(
        nets=[null_loaded_network.net],
        net_data=[null_loaded_network.net_data],
        model_params=null_loaded_network.model_params,
        qg_model_args=qg_utils.qg_model_to_args(null_loaded_network.model_params.qg_models[small_size]),
        dt=args.dt,
    )
    individual_offline_stats = [jax.jit(make_compute_offline_net_stats(loaded_network=ln, small_size=small_size)) for ln in loaded_nets]
    ensemble_offline_stats_fn = jax.jit(make_compute_offline_net_stats(loaded_network=ensemble_loaded_network, small_size=small_size))
    dt_name_slug = str(int(args.dt))
    logger.info("Using time step dt=%f", args.dt)
    # Load data
    required_fields = determine_required_fields(loaded_nets[0].net_data.input_channels)
    required_fields.update(["q", f"q_total_forcing_{small_size}", "rek", "beta", "delta"])
    with loader.SimpleQGLoader(eval_file, fields=required_fields) as data_loader:
        logger.info("Using data set with %d trajectories of %d steps", data_loader.num_trajs, data_loader.num_steps)
        # OFFLINE TESTS
        logger.info("Running offline tests")
        offline_results = {}
        for (loaded_net, os_fn) in itertools.chain(
                zip(loaded_nets, individual_offline_stats, strict=True),
                [(ensemble_loaded_network, ensemble_offline_stats_fn)]
        ):
            logger.info("Evaluating network %s", loaded_net.net_path)
            offline_results[str(loaded_net.net_path)] = []
            for traj in range(data_loader.num_trajs):
                logger.info("Testing trajectory %d", traj)
                offline_stats = os_fn(data_loader.get_trajectory(traj))
                corr_spatial = np.asarray(jnp.mean(offline_stats["corr_spatial"], axis=(-1, -2))).tolist()
                r2_spatial = np.asarray(jnp.mean(offline_stats["r2_spatial"], axis=(-1, -2))).tolist()
                mse_level = np.asarray(offline_stats["mse_level"]).tolist()
                logger.info("correlation:     %s", corr_spatial)
                logger.info("r2 spatial mean: %s", r2_spatial)
                logger.info("levelwise mse:   %s", mse_level)
                offline_results[str(loaded_net.net_path)].append(
                    {
                        "corr_spatial": corr_spatial,
                        "r2_spatial": r2_spatial,
                        "mse_level": mse_level,
                    }
                )
                with utils.rename_save_file(out_dir / f"offline_stats_dt{dt_name_slug}.json", "w") as stats_json_file:
                    json.dump(offline_results, stats_json_file)
        # ONLINE TESTS
        time_ke_computer = jax.jit(make_ke_time_computer(loaded_nets[0].model_params.qg_models[small_size]))
        logger.info("Running online tests")
        for traj in range(data_loader.num_trajs):
            logger.info("Testing trajectory %d", traj)
            traj_data = data_loader.get_trajectory(traj)
            data_q = traj_data.q
            traj_sys_params = jax.tree_map(lambda d: d[0, 0, 0, 0], traj_data.sys_params)
            ref_subsample = math.ceil(args.rollout_subsample / (DATA_SUBSAMPLE_FACTOR * (args.dt / 3600.0)))
            num_steps = math.ceil(data_loader.num_steps * DATA_SUBSAMPLE_FACTOR / (args.dt / 3600.0))
            skip_steps = math.ceil(args.t_metric_start / args.dt)
            coarsener = make_basic_coarsener(data_q.shape[-1], small_size, loaded_nets[0].model_params)
            # Reference trajectory
            ref_traj = jax.vmap(coarsener)(data_q[(skip_steps // DATA_SUBSAMPLE_FACTOR)::ref_subsample])
            # Roll out trajectories with network corrections
            net_rollouts = [("ref", ref_traj)]
            for (loaded_net, net_rollout_fn, net_label) in itertools.chain(
                zip(loaded_nets, individual_net_models, itertools.repeat("net", len(loaded_nets)), strict=True),
                [
                    (ensemble_loaded_network, ensemble_model, "ensemble"),
                    (null_loaded_network, null_model, "null"),
                ]
            ):
                logger.info("Rolling out with network %s", loaded_net.net_path)
                rolled_out_traj = net_rollout_fn(coarsener(data_q[0]), num_steps=num_steps, subsampling=args.rollout_subsample, sys_params=traj_sys_params, skip_steps=skip_steps)
                net_rollouts.append((net_label, rolled_out_traj.q))
            # PRODUCE PLOTS
            # 1. KE OVER TIME

            with utils.rename_save_file(out_dir / f"ke_over_time_traj{traj:05}_dt{dt_name_slug}.png", "wb") as ke_plot_fig:
                for label, rollout in net_rollouts:
                    kes = time_ke_computer(rollout)
                    net_times = np.linspace(args.t_metric_start, args.dt * data_loader.num_steps, kes.shape[0])
                    plt.plot(net_times, kes, **get_net_style(label), label=label)
                plt.legend()
                plt.grid(True)
                plt.xlabel("Time step")
                plt.ylabel("KE")
                plt.tight_layout()
                plt.savefig(ke_plot_fig, format="png", facecolor="white", dpi=150, bbox_inches="tight")
                plt.clf()

            # 2. SPECTRA

            ke_spec_computer = jax.jit(jax_utils.chunked_vmap(functools.partial(ke_spec, small_model=loaded_nets[0].model_params.qg_models[small_size]), 100))
            ke_values = [(net_label, ke_spec_computer(d)) for (net_label, d) in net_rollouts]
            for horizon in filter(lambda v: v <= min(ref_traj.shape[0], net_rollouts[0][1].shape[0]), [250, 1000, 2500, 5400]):
                logger.info("KE spectrum horizon %d", horizon)
                with utils.rename_save_file(out_dir / f"ke_spec_horizon{horizon:05}_traj{traj:05}_dt{dt_name_slug}.png", "wb") as ke_spec_fig:
                    fig, axs = plt.subplots(1, 2, facecolor="white")
                    for label, ke_vals in ke_values:
                        mean_ke = jnp.mean(ke_vals[:horizon], axis=0)
                        kr, ke_spec_vals = jax.vmap(spectral.calc_ispec, in_axes=[None, 0], out_axes=(None, 0))(loaded_nets[0].model_params.qg_models[small_size], mean_ke)
                        for i in range(2):
                            axs[i].loglog(kr, ke_spec_vals[i], label=label, **get_net_style(label))
                    for i in range(2):
                        axs[i].legend()
                        axs[i].grid(True)
                    axs[0].set_ylim(10**-2, 10**2.5)
                    axs[1].set_ylim(10**-4, 10**1.5)
                    fig.suptitle(f"KE Power spec horizon {horizon} steps")
                    fig.tight_layout()
                    plt.savefig(ke_spec_fig, format="png", facecolor="white", dpi=150, bbox_inches="tight")
                    plt.close(fig)

            # 3. VARIABLE PDFs
            for var in ["KE", "Ens"]:
                for level in range(2):
                    logger.info("PDF variable %s level %d", var, level)
                    var_func = jax.jit(make_pdf_var(var, level, loaded_nets[0].model_params.qg_models[small_size]))
                    with utils.rename_save_file(out_dir / f"varpdf_{var}_traj{traj:05}_lev{level}_dt{dt_name_slug}.png", "wb") as var_pdf_fig:
                        for label, rollout in net_rollouts:
                            pdf_points, pdf_density = var_func(rollout)
                            plt.semilogy(pdf_points, pdf_density, label=label, **get_net_style(label))
                        plt.legend()
                        plt.grid(True)
                        plt.title(f"PDF {var} level {level} traj {traj}")
                        plt.tight_layout()
                        plt.savefig(var_pdf_fig, format="png", facecolor="white", dpi=150, bbox_inches="tight")
                        plt.clf()

    logger.info("Finished evaluation")


if __name__ == "__main__":
    main()
