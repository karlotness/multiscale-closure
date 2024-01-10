import argparse
import pathlib
import json
import functools
import itertools
import logging
import random
import operator
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from methods import get_net_constructor
from cascaded_train import NetData, make_validation_stats_function
from train import determine_required_fields, load_model_params
from eval import check_coarse_op
from systems.qg.loader import SimpleQGLoader
import utils

parser = argparse.ArgumentParser(description="Evaluate cascaded neural networks for closure")
parser.add_argument("net_dir", type=str, help="Directory with saved network weights")
parser.add_argument("eval_set", type=str, help="Directory with evaluation examples")
parser.add_argument("net_type", type=str, help="Which saved network weights to load")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
parser.add_argument("--seed", type=int, default=None, help="Seed to use with RNG (if None, select automatically)")
# Settings for generative spectrum tests
parser.add_argument("--num_samples", type=int, default=100, help="Number of random steps to test for generative statistics")
parser.add_argument("--sample_seed", type=int, default=0, help="Seed used to draw samples for generative statistics")
parser.add_argument("--min_sample_step", type=int, default=0, help="Starting step index for sampling (can be set to exclude warmup)")


def load_networks(weight_file):
    weight_file = pathlib.Path(weight_file)
    # Load network info
    with open(weight_file.parent / "network_info.json", "r", encoding="utf8") as net_info_file:
        net_info = json.load(net_info_file)
    # Recreate networks
    nets = []
    net_data = []
    for net_params in net_info["networks"]:
        nets.append(
            get_net_constructor(net_params["arch"])(**net_params["args"], key=jax.random.PRNGKey(0))
        )
        net_data.append(
            NetData(
                input_channels=frozenset(net_params["input_channels"]),
                output_channels=frozenset(net_params["output_channels"]),
                processing_size=net_params["processing_size"],
            )
        )
    # Ensure all weights are float32
    def leaf_map(leaf):
        if isinstance(leaf, jnp.ndarray):
            if leaf.dtype == jnp.dtype(jnp.float64):
                return leaf.astype(jnp.float32)
            if leaf.dtype == jnp.dtype(jnp.complex128):
                return leaf.astype(jnp.complex64)
        return leaf
    nets = jax.tree_util.tree_map(leaf_map, nets)
    nets = eqx.tree_deserialise_leaves(weight_file, tuple(nets))
    return nets, net_info, tuple(net_data)


def run_cascaded_offline_stats(nets, eval_file, num_samples, sample_seed, base_logger, model_params, net_data, min_sample_step, processing_scales):
    sample_stats_fn = jax.jit(
        functools.partial(
            make_validation_stats_function(
                net_data=net_data,
                model_params=model_params,
                processing_scales=processing_scales
            ),
            nets=nets,
        )
    )
    required_fields = sorted(
        determine_required_fields(
            itertools.chain.from_iterable(
                itertools.chain(data.input_channels, data.output_channels)
                for data in net_data
            )
        )
    )
    logger = base_logger.getChild("cascaded_offline")
    np_rng = np.random.default_rng(seed=sample_seed)
    logger.info("Starting to load evaluation data")
    with SimpleQGLoader(eval_file, fields=required_fields) as eval_loader:
        # First, sample the (traj, index) pairs used for "offline" statistics
        valid_step_count = eval_loader.num_steps - min_sample_step
        total_steps = eval_loader.num_trajs * valid_step_count
        sample_idxs = np_rng.integers(0, total_steps, size=num_samples, dtype=np.uint64)
        sample_trajs, sample_steps = np.divmod(sample_idxs, valid_step_count)
        sample_steps = sample_steps + min_sample_step
        # For each of these, load the relevant data and stack
        batch = jax.tree_util.tree_map(
            lambda *args: jnp.concatenate(args, axis=0),
            *(eval_loader.get_trajectory(traj=operator.index(t), start=operator.index(s), end=operator.index(s)+1) for t, s in zip(sample_trajs, sample_steps, strict=True))
        )
    logger.info("Finished loading evaluation data")
    # With loaded data on device, compute statistics
    logger.info("Starting statistics computation")
    stats_report = jax.device_get(sample_stats_fn(batch))
    logger.info("Finished statistics computation")
    logger.info(f"Stat standard_mse: {stats_report['standard_mse']:<15.5g}")
    logger.info(f"Stat l2_mean: {stats_report['l2_mean']:<15.5g}")
    logger.info(f"Stat l2_total: {stats_report['l2_total']:<15.5g}")
    # Package stats
    json_results = {
        "standard_mse": stats_report["standard_mse"].item(),
        "l2_mean": stats_report["l2_mean"].item(),
        "l2_total": stats_report["l2_total"].item(),
    }
    bulk_results = {
        "standard_mse": stats_report["standard_mse"],
        "l2_mean": stats_report["l2_mean"],
        "l2_total": stats_report["l2_total"],
        "trajs": sample_trajs,
        "steps": sample_steps,
    }
    return json_results, bulk_results


def main():
    args = parser.parse_args()
    # Prepare directory paths
    net_dir = pathlib.Path(args.net_dir)
    weight_file = net_dir / "weights" / f"{args.net_type}.eqx"
    # Check that we can find both weight files
    if not weight_file.is_file():
        raise ValueError(f"weight file {args.net_type} does not exist")
    # Prepare our particular output directory
    out_dir = net_dir / "eval" / args.net_type
    # Make our directory, but only the last two levels (we don't want to create net_dir if it doesn't exist)
    out_dir.parent.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
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
    # Set up seed
    if args.seed is None:
        seed = random.SystemRandom().randint(0, 2**32)
    else:
        seed = args.seed
    logger.info("Using seed %d", seed)
    np_rng = np.random.default_rng(seed=seed)
    rng_ctr = jax.random.PRNGKey(seed=np_rng.integers(2**32).item())
    # Find the training data file
    eval_file = pathlib.Path(args.eval_set) / "data.hdf5"
    # Load the networks
    nets, net_info, net_data = load_networks(weight_file=weight_file)
    coarse_op_name = net_info.get("coarse_op_name", None)
    train_set = pathlib.Path(net_info["train_path"])
    logger.info("Parameters from training set %s", train_set)
    if coarse_op_name is not None:
        logger.info("Coarsening operator: %s", coarse_op_name)
    if not check_coarse_op(eval_file, coarse_op_name):
        logger.error("Invalid eval file for operator %s", coarse_op_name)
        sys.exit(2)
    if not check_coarse_op(train_set, coarse_op_name):
        logger.error("Invalid train file for operator %s", coarse_op_name)
        sys.exit(2)
    model_params = load_model_params(train_set)

    # Set up results files
    json_results = {}
    bulk_results = {}

    # Handle validation for basic CNN mode
    cascaded_json_res, cascaded_bulk_res = run_cascaded_offline_stats(
        nets=nets,
        eval_file=eval_file,
        num_samples=args.num_samples,
        sample_seed=args.sample_seed,
        base_logger=logger,
        model_params=model_params,
        net_data=net_data,
        min_sample_step=args.min_sample_step,
        processing_scales=set(net_info["processing_scales"]),
    )
    json_results["cascaded_offline_stats"] = cascaded_json_res
    bulk_results.update({f"cascaded_offline_stats_{k}": v for k, v in cascaded_bulk_res.items()})
    # Save results
    np.savez(out_dir / "results.npz", **bulk_results)
    with open(out_dir / "results.json", "w", encoding="utf8") as json_file:
        json.dump(json_results, json_file)
    logger.info("Finished evaluation")

if __name__ == "__main__":
    main()
