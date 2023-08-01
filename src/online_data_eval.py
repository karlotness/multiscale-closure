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
import utils
import jax_utils
from eval import load_network
import cascaded_eval
from systems.qg import loader, spectral
from online_ke_data import model_rollout
from train import determine_required_fields, determine_channel_size, load_model_params
from online_ensemble_compare import make_ke_time_computer, LoadedNetwork, ke_spec, SYS_INFO_CHANNELS

HORIZONS = [250, 1000, 2500, 5400, 10800]

parser = argparse.ArgumentParser(description="Compute statistics for poster, etc.")
parser.add_argument("out_file", type=str, help="File to store results")
parser.add_argument("eval_file", type=str, help="File to use for evaluation reference data")
parser.add_argument("net_weights", type=str, nargs="+", help="Network weights to evaluate")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
parser.add_argument("--corr_seed", type=int, default=123, help="RNG seed for correlation coefficient noise")
parser.add_argument("--corr_num_samples", type=int, default=3, help="Number of correlation samples to draw")


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


def compute_traj_values(init_q, loaded_net, sys_params, dt, subsampling, num_steps):
    net_num_steps = num_steps * subsampling
    traj_q = model_rollout(
        loaded_net=loaded_net,
        initial_q=init_q,
        sys_params=sys_params,
        dt=dt,
        num_steps=net_num_steps + 1,
        subsampling=subsampling,
        skip_steps=1,
    )
    assert traj_q.shape[0] == num_steps
    return traj_q



def make_traj_evaluator(dt, subsampling, num_steps, small_model, ke_horizons, corr_num_samples):

    def eval_traj(batch, loaded_net, corr_rng):
        ref_q = batch.q
        init_q = ref_q[0]
        sys_params = jax.tree_map(lambda d: jnp.asarray(d[0, 0, 0, 0]), batch.sys_params)
        # Compute base net_trajectory
        net_traj = compute_traj_values(
            init_q=init_q,
            loaded_net=loaded_net,
            sys_params=sys_params,
            dt=dt,
            subsampling=subsampling,
            num_steps=num_steps,
        )
        # Compute losses
        losses = compute_traj_losses(
            traj_q=net_traj,
            ref_q=ref_q,
        )
        # KE Spec
        kr, ke_specs = compute_traj_ke_spec(
            traj_q=net_traj,
            qg_model=small_model,
            horizons=ke_horizons,
        )
        # KE Times
        ke_times = compute_step_ke(traj_q=net_traj, qg_model=small_model)
        # Compute noise correlations
        noise_mask = jax.random.normal(corr_rng, shape=((corr_num_samples,) + init_q.shape), dtype=init_q.dtype) * 1e-10
        noise_init_q = jnp.expand_dims(init_q, 0) + noise_mask
        noise_net_traj = jax.vmap(
            functools.partial(
                compute_traj_values,
                loaded_net=loaded_net,
                sys_params=sys_params,
                dt=dt,
                subsampling=subsampling,
                num_steps=num_steps,
            )
        )(noise_init_q)
        corr_values = jax.vmap(lambda net_q: jax.vmap(pearson_step)(ref_q, net_q))(noise_net_traj)
        return {
            "losses": losses,
            "ke_spec": {
                "kr": kr,
                "specs": ke_specs,
            },
            "ke_times": ke_times,
            "corr_values": corr_values,
        }

    return eval_traj


def load_networks(net_paths, eval_file, logger=None):
    if logger is None:
        logger = logging.getLogger("load-net")
    loaded_nets = []
    logger.info("Loading networks")
    for net_path in map(pathlib.Path, net_paths):
        logger.info("Loading network %s", net_path)
        net, net_info = load_network(net_path)
        loaded_nets.append(
            LoadedNetwork(
                net=net,
                net_info=net_info,
                net_data=cascaded_eval.NetData(
                    input_channels=net_info["input_channels"],
                    output_channels=net_info["output_channels"],
                    processing_size=net_info["processing_size"]
                ),
                net_path=str(net_path.resolve()),
                model_params=load_model_params(net_info["train_path"], eval_path=eval_file),
            )
        )
        del loaded_nets[-1].model_params.qg_models["big_model"]
    # Add null network
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
        logger.error("Output file %s already exists", eval_file)
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
    small_size = determine_channel_size(loaded_nets[0].net_data.output_channels[0])
    logger.info("Small size: %d", small_size)

    # Determine required fields
    required_fields = determine_required_fields(
        itertools.chain.from_iterable(
            itertools.chain(ln.net_data.input_channels, ln.net_data.output_channels) for ln in loaded_nets
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
    traj_eval_fn = eqx.filter_jit(
        make_traj_evaluator(
            dt=dt,
            subsampling=subsample,
            num_steps=num_steps,
            small_model=loaded_nets[0].model_params.qg_models[small_size],
            ke_horizons=HORIZONS,
            corr_num_samples=args.corr_num_samples,
        )
    )
    corr_rng_ctr = jax.random.PRNGKey(args.corr_seed)

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
                traj_group = results_group.create_group(f"traj{traj_num:05d}")
                logger.info("Starting trajectory %d", traj_num)
                corr_rng_ctr, corr_rng = jax.random.split(corr_rng_ctr, 2)
                traj_data = data_set.get_trajectory(traj_num)
                for net_id, ln in zip(net_ids, loaded_nets, strict=True):
                    net_group = traj_group.create_group(net_id)
                    logger.info("Processing trajectory %d and net %s", traj_num, net_id)
                    results = traj_eval_fn(traj_data, ln, corr_rng)
                    # Store results
                    # losses
                    loss_group = net_group.create_group("losses")
                    for loss_k, loss_v in results["losses"].items():
                        loss_group.create_dataset(loss_k, data=np.asarray(loss_v))
                    # ke times
                    net_group.create_dataset("ke_times", data=np.asarray(results["ke_times"]))
                    # corr values
                    net_group.create_dataset("corr_values", data=np.asarray(results["corr_values"]))
                    # ke specta
                    spectra_group = net_group.create_group("spectra")
                    spectra_group.create_dataset("kr", data=np.asarray(results["ke_spec"]["kr"]))
                    for horizon, spectrum in results["ke_spec"]["specs"].items():
                        spectra_group.create_dataset(f"horiz{horizon:05d}", data=np.asarray(spectrum))
                    results = None

    logger.info("Finished computing stats")


if __name__ == "__main__":
    main()
