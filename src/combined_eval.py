import pathlib
import dataclasses
import itertools
import argparse
import utils
import logging
import json
import sys
import random
import operator
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from systems.qg import diagnostics as qg_spec_diag
from systems.qg.loader import SimpleQGLoader
import utils
from train import load_model_params, determine_required_fields, determine_output_size, make_basic_coarsener, make_chunk_from_batch, remove_residual_from_output_chunk, make_non_residual_chunk_from_batch
from eval import load_network, check_coarse_op
import jax_utils


parser = argparse.ArgumentParser(description="Evaluate neural networks for closure")
parser.add_argument("out_dir", type=str, help="Directory with saved network weights")
parser.add_argument("eval_set", type=str, help="Directory with evaluation examples")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
parser.add_argument("--seed", type=int, default=None, help="Seed to use with RNG (if None, select automatically)")
# Specification for networks to load
parser.add_argument("--downscale_nets", type=str, nargs="+", help="Paths to downscale network weights")
parser.add_argument("--buildup_nets", type=str, nargs="+", help="Paths to buildup network weights")
# Settings for generative spectrum tests
parser.add_argument("--num_samples", type=int, default=100, help="Number of random steps to test for generative statistics")
parser.add_argument("--sample_seed", type=int, default=0, help="Seed used to draw samples for generative statistics")
parser.add_argument("--min_sample_step", type=int, default=0, help="Starting step index for sampling (can be set to exclude warmup)")


@jax_utils.register_pytree_dataclass
@dataclasses.dataclass
class LoadedNetwork:
    net: object
    net_info: dict
    coarse_op_name: str
    model_params: object
    weights_path: str


def load_eval_network(weights_path, eval_file_path=None):
    weights_path = pathlib.Path(weights_path)
    net, net_info = load_network(weights_path)
    coarse_op_name = net_info.get("coarse_op_name", None)
    if eval_file_path is not None and not check_coarse_op(eval_file_path, coarse_op_name):
        raise ValueError(f"invalid evaluation file coarsening for network {weights_path}")
    train_set = pathlib.Path(net_info["train_path"])
    model_params = load_model_params(train_set)
    return LoadedNetwork(
        net=net,
        net_info=net_info,
        coarse_op_name=coarse_op_name,
        model_params=model_params,
        weights_path=str(weights_path.resolve())
    )


@jax_utils.register_pytree_dataclass
@dataclasses.dataclass
class MaskedNetwork:
    net: object
    model_params: object
    processing_size: int
    input_channels: tuple[str]
    output_channels: tuple[str]


def make_masked_network(loaded_network):
    return MaskedNetwork(
        net=loaded_network.net,
        model_params=loaded_network.model_params,
        processing_size=operator.index(loaded_network.net_info["processing_size"]),
        input_channels=tuple(sorted(loaded_network.net_info["input_channels"])),
        output_channels=tuple(sorted(loaded_network.net_info["output_channels"])),
    )


def get_network_pytree_mask():
    return MaskedNetwork(
        net=eqx.is_array,
        model_params=eqx.is_array,
        processing_size=False,
        input_channels=False,
        output_channels=False,
    )

def check_network_consistency(nets):
    nets = list(nets)
    net0 = nets[0]
    for net in nets:
        if net.net_info["input_channels"] != net0.net_info["input_channels"]:
            return False
        if net.net_info["output_channels"] != net0.net_info["output_channels"]:
            return False
    return True


def check_cross_task_consistency(downscale_net, buildup_net):
    if len(downscale_net.net_info["output_channels"]) > 1:
        return False
    if downscale_net.net_info["output_channels"][0] not in buildup_net.net_info["input_channels"]:
        return False
    return True


def load_eval_data(eval_file, num_samples, sample_seed, min_sample_step, required_fields):
    np_rng = np.random.default_rng(seed=sample_seed)
    with SimpleQGLoader(eval_file, fields=required_fields) as eval_loader:
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
        return batch, sample_trajs, sample_steps


@eqx.filter_jit(args=(eqx.is_array, get_network_pytree_mask(), get_network_pytree_mask()))
def compute_network_stats(batch, masked_downscale, masked_buildup):
    downscale_input = make_chunk_from_batch(
        channels=masked_downscale.input_channels,
        batch=batch,
        model_params=masked_downscale.model_params,
        processing_size=masked_downscale.processing_size,
    )
    # Run inputs through downscale network
    downscale_output = jax.vmap(
        make_basic_coarsener(masked_downscale.processing_size, determine_output_size(masked_downscale.output_channels), masked_downscale.model_params)
    )(jax.vmap(masked_downscale.net)(downscale_input))
    # Now register these as an alt_source
    assert len(masked_downscale.output_channels) == 1
    alt_source = {
        masked_downscale.output_channels[0]: downscale_output
    }
    # Create input chunk for buildup network
    buildup_input = make_chunk_from_batch(
        channels=masked_buildup.input_channels,
        batch=batch,
        model_params=masked_buildup.model_params,
        processing_size=masked_buildup.processing_size,
        alt_source=alt_source,
    )
    # Get buildup residual predictions
    output_size = determine_output_size(masked_buildup.output_channels)
    residual_pred = jax.vmap(
        make_basic_coarsener(masked_buildup.processing_size, output_size, masked_buildup.model_params)
    )(jax.vmap(masked_buildup.net)(buildup_input))
    samples = remove_residual_from_output_chunk(
        output_channels=masked_buildup.output_channels,
        output_chunk=residual_pred,
        batch=batch,
        model_params=masked_buildup.model_params,
        processing_size=output_size,
        alt_source=alt_source,
    )
    targets = make_non_residual_chunk_from_batch(
        channels=masked_buildup.output_channels,
        batch=batch,
        model_params=masked_buildup.model_params,
        processing_size=output_size,
    )
    # Compute statistics
    err = jnp.abs(targets - samples)
    mse = jnp.mean(err**2)
    stats = qg_spec_diag.subgrid_scores(
        true=jnp.expand_dims(targets, 1),
        mean=jnp.expand_dims(samples, 1),
        gen=jnp.expand_dims(samples, 1),
    )
    return {
        "standard_mse": mse,
        "l2_mean": stats.l2_mean,
        "l2_total": stats.l2_total,
    }


def main():
    args = parser.parse_args()
    # Prepare our particular output directory
    out_dir = pathlib.Path(args.out_dir).resolve()
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
    # Process eval data path
    eval_path = pathlib.Path(args.eval_set) / "data.hdf5"
    # Load the various networks
    downscale_nets = []
    buildup_nets = []
    for net_path in args.downscale_nets:
        logger.info("Loading downscale network %s", net_path)
        downscale_nets.append(load_eval_network(weights_path=net_path, eval_file_path=eval_path))
    for net_path in args.buildup_nets:
        logger.info("Loading buildup network %s", net_path)
        buildup_nets.append(load_eval_network(weights_path=net_path, eval_file_path=eval_path))
    if len(downscale_nets) < 1 or len(buildup_nets) < 1:
        logger.error("Must specify at least one network to test")
        sys.exit(1)
    logger.info("Finished loading networks")
    # Check network information for consistency (all same input and output channels)
    if not check_network_consistency(downscale_nets) or not check_network_consistency(buildup_nets):
        logger.error("Network inputs or outputs inconsistent. Make sure to specify only networks performing the same task")
        sys.exit(1)
    if not check_cross_task_consistency(downscale_nets[0], buildup_nets[0]):
        logger.error("Network types do not correspond in outputs")
        sys.exit(1)
    # Determine required fields and load evaluation data
    required_fields = sorted(
        determine_required_fields(
            itertools.chain.from_iterable(
                itertools.chain(
                    net.net_info["input_channels"],
                    net.net_info["output_channels"],
                )
                for net in itertools.chain(
                        downscale_nets,
                        buildup_nets
                )
            )
        )
    )
    logger.info("Required fields: %s", required_fields)
    # Load the evaluation data
    logger.info("Loading eval data")
    eval_data, sample_trajs, sample_steps = load_eval_data(
        eval_file=eval_path,
        num_samples=args.num_samples,
        sample_seed=args.sample_seed,
        min_sample_step=args.min_sample_step,
        required_fields=required_fields,
    )
    logger.info("Finished loading eval data")
    stats_reports = {"cross_nets": []}
    for (downscale_i, downscale_net), (buildup_i, buildup_net) in itertools.product(enumerate(downscale_nets), enumerate(buildup_nets)):
        logger.info("Running evaluation on downscale net %d and buildup net %d", downscale_i + 1, buildup_i + 1)
        single_stats_report = jax.device_get(
            compute_network_stats(
                eval_data,
                make_masked_network(downscale_net),
                make_masked_network(buildup_net),
            )
        )
        single_stats_report = jax_utils.make_json_serializable(single_stats_report)
        logger.info(f"Stat standard_mse: {single_stats_report['standard_mse']:<15.5g}")
        logger.info(f"Stat l2_mean: {single_stats_report['l2_mean']:<15.5g}")
        logger.info(f"Stat l2_total: {single_stats_report['l2_total']:<15.5g}")
        logger.info("Finished evaluation on downscale net %d and buildup net %d", downscale_i + 1, buildup_i + 1)
        # Store results
        stats_reports["cross_nets"].append(
            {
                "downscale_net": {
                    "index": downscale_i,
                    "path": downscale_net.weights_path,
                },
                "buildup_net": {
                    "index": buildup_i,
                    "path": buildup_net.weights_path,
                },
                "stats_report": single_stats_report,
            }
        )
        with open(out_dir / "eval_results.json", "w") as report_file:
            json.dump(stats_reports, report_file)
    # Compute means of all measurements
    logger.info("Reporting means of all current runs")
    mean_standard_mse = np.mean([rep["stats_report"]["standard_mse"] for rep in stats_reports["cross_nets"]])
    mean_l2_mean = np.mean([rep["stats_report"]["l2_mean"] for rep in stats_reports["cross_nets"]])
    mean_l2_total = np.mean([rep["stats_report"]["l2_total"] for rep in stats_reports["cross_nets"]])
    # Store means
    stats_reports["means"] = {
        "standard_mse": mean_standard_mse,
        "l2_mean": mean_l2_mean,
        "l2_total": mean_l2_total,
    }
    with open(out_dir / "eval_results.json", "w") as report_file:
        json.dump(stats_reports, report_file)
    logger.info(f"Stat standard_mse: {mean_standard_mse:<15.5g}")
    logger.info(f"Stat l2_mean: {mean_l2_mean:<15.5g}")
    logger.info(f"Stat l2_total: {mean_l2_total:<15.5g}")
    logger.info("Finished evaluation")

if __name__ == "__main__":
    main()
