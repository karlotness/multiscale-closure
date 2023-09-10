import sys
import argparse
import functools
import logging
import pathlib
import re
import itertools
import h5py
import jax.numpy as jnp
import jax
import equinox as eqx
import numpy as np
import operator
import utils
import jax_utils
import importlib
from eval import load_network
import cascaded_eval
from cascaded_train import name_remove_residual
from systems.qg import loader, spectral, utils as qg_utils, coarsen
from online_ke_data import model_rollout
from train import determine_required_fields, determine_channel_size, load_model_params, determine_output_size
from online_ensemble_compare import make_ke_time_computer, LoadedNetwork, ke_spec, SYS_INFO_CHANNELS
from cascaded_online_eval import make_net_param_func
import pyqg_jax

HORIZONS = [250, 1000, 2500, 5400, 10800]

parser = argparse.ArgumentParser(description="Compute statistics for poster, etc.")
parser.add_argument("out_file", type=str, help="File to store results")
parser.add_argument("eval_file", type=str, help="File to use for evaluation reference data")
parser.add_argument("net_weights", type=str, nargs="+", help="Network weights to evaluate")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
parser.add_argument("--corr_seed", type=int, default=123, help="RNG seed for correlation coefficient noise")
parser.add_argument("--corr_num_samples", type=int, default=3, help="Number of correlation samples to draw (UNUSED)")
parser.add_argument("--trajectory_batch_size", type=int, default=16, help="Number of trajectories to batch together")


def batched(iterable, n):
    # Following recipe from itertools documentation
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def pearsonr(x, y):
    # Based on SciPy implementation
    assert x.ndim == 1
    assert y.ndim == 1
    xmean = jnp.mean(x)
    ymean = jnp.mean(y)
    xm = x - xmean
    ym = y - ymean
    normxm = jnp.linalg.norm(xm)
    normym = jnp.linalg.norm(ym)
    r = jnp.dot(xm / normxm, ym / normym)
    r = jnp.clip(r, -1.0, 1.0)
    return r


def pearson_step(step_x, step_y):
    return pearsonr(step_x.ravel(), step_y.ravel())


def compute_step_ke(traj_q, qg_model):
    time_ke_computer = make_ke_time_computer(qg_model)
    traj_ke = time_ke_computer(traj_q)
    return traj_ke


def compute_traj_ke_spec(traj_q, qg_model, horizons):
    ke_spec_computer = jax_utils.chunked_vmap(functools.partial(ke_spec, small_model=qg_model), 100)
    ke_vals = ke_spec_computer(traj_q)
    result = {}
    for horizon in horizons:
        mean_ke = jnp.mean(ke_vals[:horizon], axis=0)
        kr, ke_spec_vals = jax.vmap(spectral.calc_ispec, in_axes=[None, 0], out_axes=(None, 0))(qg_model, mean_ke)
        result[horizon] = ke_spec_vals
    return kr, result


def compute_traj_losses(traj_q, ref_q):
    assert traj_q.shape == ref_q.shape
    assert traj_q.ndim == 4
    # Pearson corrleation
    corr = jax.vmap(pearson_step)(ref_q, traj_q)
    # MSE
    err = traj_q - ref_q
    mean_err = jnp.mean(jnp.abs(err), axis=(-1, -2, -3))
    mse = jnp.mean(err**2, axis=(-1, -2, -3))
    rel_l1_err = mean_err / jnp.mean(jnp.abs(ref_q), axis=(-1, -2, -3))
    return {
        "corr": corr,
        "mean_err": mean_err,
        "mse": mse,
        "rel_l1_err": rel_l1_err,
    }



def make_parameterized_stepped_model(nets, net_data, model_params, qg_model_args, dt):

    def model_stepper(initial_q, subsampling=1, sys_params={}, skip_steps=0):
        assert all(v.ndim == 0 for v in sys_params.values())
        new_model_params = qg_model_args.copy()
        new_model_params.update(sys_params)
        fixed_sys_params = {k: jnp.full((1, 1, 1), fill_value=v) for k, v in sys_params.items()}
        model = pyqg_jax.steppers.SteppedModel(
            pyqg_jax.parameterizations.ParameterizedModel(
                pyqg_jax.qg_model.QGModel(
                    **new_model_params,
                ),
                param_func=functools.partial(
                    make_net_param_func(
                        nets=nets,
                        net_data=net_data,
                        model_params=model_params,
                    ),
                    sys_params=fixed_sys_params,
                )
            ),
            pyqg_jax.steppers.AB3Stepper(dt=dt),
        )

        # Wrap in model states
        inner_state = model.model.model.create_initial_state(jax.random.PRNGKey(0))
        inner_state = inner_state.update(q=initial_q)
        state = model.initialize_stepper_state(
            model.model.initialize_param_state(inner_state)
        )

        # Step through time
        def one_step_state(carry, _x):
            old_state = carry
            new_state = model.step_model(old_state)
            out_value = old_state.state.model_state
            return new_state, out_value

        def subsample_step_state(state):
            new_state, _ys = jax.lax.scan(
                one_step_state,
                state,
                None,
                length=subsampling,
            )
            return new_state

        # Step through time
        def step_state(carry, _x):
            old_state = carry
            new_state = model.step_model(old_state)
            out_value = old_state.state.model_state
            return new_state, out_value

        def skip_states(carry, _x):
            new_state, _y = step_state(carry, _x)
            return new_state, None

        # Skip the warmup steps, if any
        if skip_steps > 0:
            state, _states = jax.lax.scan(
                skip_states,
                state,
                None,
                length=skip_steps,
            )

        return state, subsample_step_state

    return model_stepper


def values_from_step(i, carry, step_state, qg_model, horizons):
    state, results = carry
    results = results.copy()
    # Process the current state into results
    step_ke = compute_step_ke(jnp.expand_dims(state.state.model_state.q, 0), qg_model=qg_model)
    results["ke_times"] = jax.lax.dynamic_update_index_in_dim(
        results["ke_times"],
        step_ke,
        index=i,
        axis=0,
    )
    # Spectrum
    ke_spec_val = ke_spec(state.state.model_state.q, small_model=qg_model)
    # Include specrum in averages as needed
    for horizon in horizons:
        results["ke_spec"][horizon] = results["ke_spec"][horizon] + jax.lax.select(
            i < horizon,
            ke_spec_val,
            jnp.zeros_like(ke_spec_val),
        )
    # Step time forward
    new_state = step_state(state)
    return new_state, results


def make_traj_evaluator(dt, subsampling, num_steps, small_model, ke_horizons, corr_num_samples):

    def eval_traj(batch, loaded_net, corr_rng):
        processing_size = loaded_net.net_info["processing_size"]
        q_size = batch.q.shape[-1]
        ref_q = batch.q
        if q_size != processing_size:
            coarse_op = coarsen.COARSEN_OPERATORS[loaded_net.model_params.scale_operator](
                big_model=loaded_net.model_params.qg_models[q_size],
                small_nx=processing_size,
            )
            ref_q = jax.vmap(coarse_op.coarsen)(ref_q)
        init_q = ref_q[0]
        sys_params = jax.tree_map(lambda d: jnp.asarray(d[0, 0, 0, 0]), batch.sys_params)
        # Prepare state
        phase1 = make_parameterized_stepped_model(
            nets=loaded_net.net,
            net_data=loaded_net.net_data,
            model_params=loaded_net.model_params,
            qg_model_args=qg_utils.qg_model_to_args(small_model),
            dt=dt,
        )
        state, step_state = phase1(
            initial_q=init_q,
            subsampling=subsampling,
            sys_params=sys_params,
            skip_steps=1,
        )
        # Loop over the states
        init_carry = (
            state,
            {
                "ke_times": jnp.zeros((num_steps,), dtype=jnp.float64),
                "ke_spec": {
                    horizon: jnp.zeros((small_model.nz, small_model.nl, small_model.nk), dtype=jnp.float64)
                    for horizon in HORIZONS
                },
            },
        )
        final_state, result = jax.lax.fori_loop(
            0,
            num_steps,
            functools.partial(values_from_step, step_state=step_state, qg_model=small_model, horizons=HORIZONS),
            init_carry,
        )
        out_result = {
            "ke_times": result["ke_times"],
            "ke_spec": {"specs": {}}
        }
        for k in result["ke_spec"].keys():
            mean_ke = result["ke_spec"][k] / k
            kr, ke_spec_vals = jax.vmap(spectral.calc_ispec, in_axes=[None, 0], out_axes=(None, 0))(small_model, mean_ke)
            out_result["ke_spec"]["specs"][k] = ke_spec_vals
        out_result["ke_spec"]["kr"] = kr
        return out_result

    return eval_traj


def batch_traj_evaluator(traj_eval_fn):

    def eval_traj(batch_of_batches, loaded_net, corr_rng):
        batch_size = set(jnp.shape(leaf)[0] for leaf in jax.tree_util.tree_leaves(batch_of_batches))
        return jax.vmap(lambda batch: traj_eval_fn(batch, loaded_net, corr_rng))(batch_of_batches)

    return eval_traj



@jax.jit
def traj_stack(traj_batches):
    return jax.tree_util.tree_map(lambda *args: jnp.stack(args), *traj_batches)


def load_networks(net_paths, eval_file, logger=None):
    if logger is None:
        logger = logging.getLogger("load-net")
    loaded_nets = []
    logger.info("Loading networks")
    for net_path in map(pathlib.Path, net_paths):
        logger.info("Loading network %s", net_path)
        try:
            net, net_info = load_network(net_path)
            loaded_nets.append(
                LoadedNetwork(
                    net=[net],
                    net_info=net_info,
                    net_data=[
                        cascaded_eval.NetData(
                            input_channels=net_info["input_channels"],
                            output_channels=net_info["output_channels"],
                            processing_size=net_info["processing_size"]
                        ),
                    ],
                    net_path=str(net_path.resolve()),
                    model_params=load_model_params(net_info["train_path"], eval_path=eval_file),
                )
            )
        except Exception:
            # This might be a cascaded network
            net, net_info, net_data = importlib.import_module("cascaded_eval").load_networks(net_path)
            # Patch up net_info
            net_info["input_channels"] = net_info["networks"][0]["input_channels"].copy()
            net_info["output_channels"] = net_info["networks"][-1]["output_channels"].copy()
            net_info["processing_size"] = determine_output_size(net_info["output_channels"])
            loaded_nets.append(
                LoadedNetwork(
                    net=net,
                    net_info=net_info,
                    net_data=net_data,
                    net_path=str(net_path.resolve()),
                    model_params=load_model_params(net_info["train_path"], eval_path=eval_file),
                )
            )
        del loaded_nets[-1].model_params.qg_models["big_model"]
    # Add null network
    null_net_info = loaded_nets[0].net_info.copy()
    null_net_info["input_channels"] = list(map(name_remove_residual, filter(lambda c: not re.match(SYS_INFO_CHANNELS, c), loaded_nets[0].net_info["input_channels"])))
    null_net_info["output_channels"] = list(map(name_remove_residual, filter(lambda c: not re.match(SYS_INFO_CHANNELS, c), loaded_nets[0].net_info["output_channels"])))
    null_net_data = cascaded_eval.NetData(
        input_channels=null_net_info["input_channels"],
        output_channels=null_net_info["output_channels"],
        processing_size=null_net_info["processing_size"],
    )
    null_loaded_network = LoadedNetwork(
        net=[lambda chunk: jnp.zeros_like(chunk)],
        net_info=null_net_info,
        net_data=[null_net_data],
        net_path="null net",
        model_params=loaded_nets[0].model_params,
    )
    loaded_nets.append(null_loaded_network)
    return loaded_nets


def main():
    args = parser.parse_args()
    # Set up logging
    out_file = pathlib.Path(args.out_file)
    eval_file = pathlib.Path(args.eval_file)
    log_file = out_file.parent / f"run-{out_file.stem}.log"
    utils.set_up_logging(level=args.log_level, out_file=log_file)
    logger = logging.getLogger("main")
    # Log basic information
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
    if not eval_file.is_file():
        logger.error("Eval file %s does not exist", eval_file)
        sys.exit(2)
    if out_file.is_file():
        logger.error("Output file %s already exists", out_file)
        sys.exit(3)

    # Load networks
    logger.info("Loading %d networks", len(args.net_weights))
    loaded_nets = load_networks(net_paths=args.net_weights, eval_file=eval_file, logger=logger.getChild("load-nets"))
    net_ids = []
    counter = 0
    for ln in loaded_nets:
        if ln.net_path == "null net":
            net_ids.append("null")
        else:
            net_ids.append(f"net{counter}")
            counter += 1
    logger.info("Finished loading %d networks", len(loaded_nets) - 1)
    small_size = determine_output_size(loaded_nets[0].net_data[-1].output_channels)
    logger.info("Small size: %d", small_size)

    # Determine required fields
    required_fields = determine_required_fields(
        itertools.chain.from_iterable(
            itertools.chain(ln.net_data[0].input_channels, ln.net_data[-1].output_channels) for ln in loaded_nets
        )
    )
    required_fields.update(["rek", "beta", "delta"])
    logger.info("Required fields: %s", required_fields)

    # Determine required eval parameters
    with h5py.File(eval_file, "r") as data_file:
        dt = data_file["params"]["dt"][()].item()
        subsample = data_file["params"]["subsample"][()].item()
        num_steps = data_file["trajs"]["traj00000_q"].shape[0]
        logger.info("Eval set dt=%f", dt)
        logger.info("Eval set subsample=%d", subsample)
        logger.info("Eval set num_steps=%d", num_steps)

    # Args: batch, loaded_net, corr_rng
    if small_size in loaded_nets[0].model_params.qg_models:
        small_model = loaded_nets[0].model_params.qg_models[small_size]
    else:
        small_model = loaded_nets[0].model_params.qg_models[max(filter(lambda v: isinstance(v, int), loaded_nets[0].model_params.qg_models.keys()))]
        small_model = qg_utils.qg_model_to_args(small_model)
        small_model["nx"] = small_size
        small_model["ny"] = small_size
        small_model = pyqg_jax.qg_model.QGModel(**small_model)
    traj_eval_fn = eqx.filter_jit(
        batch_traj_evaluator(
            make_traj_evaluator(
                dt=dt,
                subsampling=subsample,
                num_steps=num_steps,
                small_model=small_model,
                ke_horizons=HORIZONS,
                corr_num_samples=args.corr_num_samples,
            )
        )
    )
    corr_rng = jax.random.PRNGKey(args.corr_seed)

    # Open output file
    with h5py.File(out_file, "w") as out_file:
        logger.info("Storing basic parameters")
        params_group = out_file.create_group("params")
        params_group.create_dataset("eval_file", data=str(eval_file.resolve()))
        paths_group = params_group.create_group("paths")
        for net_id, ln in zip(net_ids, loaded_nets, strict=True):
            paths_group.create_dataset(net_id, data=str(ln.net_path))
        logger.info("Finished storing parameters")
        # Do network processing
        with loader.SimpleQGLoader(eval_file, fields=required_fields) as data_set:
            results_group = out_file.create_group("results")
            assert data_set.num_steps == num_steps

            for traj_num in range(data_set.num_trajs):
                results_group.create_group(f"traj{traj_num:05d}")

            for net_id, ln in zip(net_ids, loaded_nets, strict=True):
                for traj_batch in batched(range(data_set.num_trajs), args.trajectory_batch_size):
                    # Load data
                    logger.info("Loading trajectories %d - %d", min(traj_batch), max(traj_batch))
                    traj_data = []
                    for traj_num in traj_batch:
                        traj_data.append(data_set.get_trajectory(traj_num, 0, 1))
                    traj_data = traj_stack(traj_data)
                    # Compute results
                    logger.info("Processing trajectories %d - %d and net %s", min(traj_batch), max(traj_batch), net_id)
                    batch_results = traj_eval_fn(traj_data, ln, corr_rng)
                    # Store results
                    for idx, traj_num in enumerate(traj_batch):
                        results = jax.tree_util.tree_map(operator.itemgetter(idx), batch_results)
                        traj_group = results_group[f"traj{traj_num:05d}"]
                        net_group = traj_group.create_group(net_id)
                        # ke times
                        net_group.create_dataset("ke_times", data=np.asarray(results["ke_times"]))
                        # ke specta
                        spectra_group = net_group.create_group("spectra")
                        spectra_group.create_dataset("kr", data=np.asarray(results["ke_spec"]["kr"]))
                        for horizon, spectrum in results["ke_spec"]["specs"].items():
                            spectra_group.create_dataset(f"horiz{horizon:05d}", data=np.asarray(spectrum))
                        results = None
                    batch_results = None

    logger.info("Finished computing stats")


if __name__ == "__main__":
    main()
