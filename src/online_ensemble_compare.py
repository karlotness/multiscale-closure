import jax
import jax.numpy as jnp
import jax_utils
import utils
import itertools
import pathlib
import logging
import sys
import argparse
import dataclasses
import cascaded_eval
from cascaded_online_eval import make_parameterized_stepped_model
from eval import load_network, make_network_evaluator
from train import load_model_params
from systems.qg import utils as qg_utils, loader

DATA_SUBSAMPLE_FACTOR = 8

parser = argparse.ArgumentParser(description="Compare offline and online metrics for individual networks vs ensemble")
parser.add_argument("out_dir", type=str, help="Directory to store evaluation results")
parser.add_argument("eval_set", type=str, help="Directory with evaluation samples")
parser.add_argument("nets", type=str, nargs="+", help="Trained network directory")
parser.add_argument("--net_type", type=str, default="best_loss", help="Trained network directory")
parser.add_argument("--dt", type=float, default=3600.0, help="Time step size")
parser.add_argument("--rollout_subsample", type=int, default=4, help="Stride to use when rolling out simulations")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])

LoadedNetwork = dataclasses.make_dataclass("LoadedNetwork", ["net", "net_info", "net_data", "net_path", "model_params"])

def make_ensemble_net(nets):
    def ensemble_net(chunk):
        results = []
        for net_res in nets:
            results.append(net_res.net(chunk))
        results = jnp.stack(results, axis=0)
        return jnp.mean(results, axis=0)
    return ensemble_net


def make_compute_offline_net_stats(loaded_network, small_size):
    # Only for individual networks vs a baseline
    time_axis = -4

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
        return {
            "r2_spatial": r2_val_spatial,
            "corr_spatial": corr_val_spatial,
        }

    return compute_stats


def make_compute_online_net_stats(net_evaluator, num_steps, subsampling):

    def compute_stats(batch):

        initial_q = batch.q[0]
        pass

    return compute_stats


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
                model_params=load_model_params(net_info["train_path"]),
            )
        )
    # Build parameterized models and ensemble
    logger.info("Constructing parameterized models for individual networks")
    individual_net_models = []
    for ln in loaded_nets:
        small_size = int(ln.net_info["output_channels"][0][-2:])
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
        nets=[ensemble_loaded_network],
        net_data=[ensemble_loaded_network.net_data],
        model_params=ensemble_loaded_network.model_params,
        qg_model_args=qg_utils.qg_model_to_args(ensemble_loaded_network.model_params.qg_models[small_size]),
        dt=args.dt,
    )
    individual_offline_stats = [jax.jit(make_compute_offline_net_stats(loaded_network=ln, small_size=small_size)) for ln in loaded_nets]
    ensemble_offline_stats_fn = jax.jit(make_compute_offline_net_stats(loaded_network=ensemble_loaded_network, small_size=small_size))
    dt_name_slug = f"{args.dt:.1f}"
    logger.info("Using time step dt=%f", args.dt)
    # Load data
    with loader.SimpleQGLoader(eval_file, fields=["q", f"q_total_forcing_{small_size}"]) as data_loader:
        logger.info("Using data set with %d trajectories of %d steps", data_loader.num_trajs, data_loader.num_steps)
        # Run tests (include online AND offline!) and save plots
        for (loaded_net, os_fn) in itertools.chain(
                zip(loaded_nets, individual_offline_stats, strict=True),
                [(ensemble_loaded_network, ensemble_offline_stats_fn)]
        ):
            logger.info("Evaluating network %s", loaded_net.net_path)
            for traj in range(data_loader.num_trajs):
                logger.info("Testing trajectory %d", traj)
                offline_stats = os_fn(data_loader.get_trajectory(traj))
                logger.info("correlation:     %s", jnp.mean(offline_stats["corr_spatial"], axis=(-1, -2)))
                logger.info("r2 spatial mean: %s", jnp.mean(offline_stats["r2_spatial"], axis=(-1, -2)))


if __name__ == "__main__":
    main()
