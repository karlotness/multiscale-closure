import pathlib
import argparse
import functools
import utils
import logging
import types
import json
import sys
import random
import operator
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from systems.qg.loader import SimpleQGLoader, qg_model_from_hdf5
from systems.qg import utils as qg_utils, diagnostics as qg_spec_diag
import utils
import math
import dataclasses
from methods.gz_fcnn import GZFCNN
from methods import ARCHITECTURES
from train_sdegm import determine_required_fields, build_fixed_input_from_batch, make_scalers, make_coarseners, make_sampler
import jax_utils


parser = argparse.ArgumentParser(description="Evaluate neural networks for closure")
parser.add_argument("net_dir", type=str, help="Directory with saved network weights")
parser.add_argument("eval_set", type=str, help="Directory with evaluation examples")
parser.add_argument("net_type", type=str, help="Which saved network weights to load")
parser.add_argument("train_set", type=str, help="Directory with training data (used for scalers)")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
parser.add_argument("--sampling_dt", type=float, default=0.01, help="Time step size for the SDE generating samples")
parser.add_argument("--seed", type=int, default=None, help="Seed to use with RNG (if None, select automatically)")
# Settings for generative spectrum tests
parser.add_argument("--num_spectrum_samples", type=int, default=100, help="Number of random steps to test for generative statistics")
parser.add_argument("--spectrum_seed", type=int, default=0, help="Seed used to draw samples for generative statistics")
parser.add_argument("--num_mean_samples", type=int, default=10, help="Number of samples to compute mean of generated forcing")
parser.add_argument("--spectrum_min_sample_step", type=int, default=0, help="Starting step index for sampling (can be set to exclude warmup)")


def load_network(weight_file):
    weight_file = pathlib.Path(weight_file)
    # Load network info
    with open(weight_file.parent / "network_info.json", "r", encoding="utf8") as net_info_file:
        net_info = json.load(net_info_file)
    net = ARCHITECTURES[net_info["arch"]](**net_info["args"], key=jax.random.PRNGKey(0))
    # Ensure all weights are float32
    def leaf_map(leaf):
        if isinstance(leaf, jnp.ndarray):
            if leaf.dtype == jnp.dtype(jnp.float64):
                return leaf.astype(jnp.float32)
            if leaf.dtype == jnp.dtype(jnp.complex128):
                return leaf.astype(jnp.complex64)
        return leaf
    net = jax.tree_util.tree_map(leaf_map, net)
    net = eqx.tree_deserialise_leaves(weight_file, net)
    return net, net_info


def make_generative_sampler(net, num_mean_samples, sample_dt, scalers, coarseners, input_channels):

    def compute_samples_mean(batch, rng):
        # Function of batch, net, rng
        sample_fn = functools.partial(
            make_sampler(
                scalers=scalers,
                coarseners=coarseners,
                input_channels=input_channels,
                dt=sample_dt,
            ),
            batch=batch,
            net=net,
        )
        rngs = jax.random.split(rng, num_mean_samples)
        # Scan instead of vmap to save memory at the cost of parallelism
        _carry, mean_samples = jax.lax.scan(
            lambda _carry, x: (None, sample_fn(rng=x)[0]),
            None,
            rngs,
            length=num_mean_samples,
        )
        sample = mean_samples[0]
        mean = jnp.mean(mean_samples, axis=0)
        return sample, mean

    return compute_samples_mean


def make_generative_stat_computer(net, num_mean_samples, sample_dt, scalers, coarseners, input_channels):

    def map_gen_sampler(_carry, x):
        batch_elem, rng = x
        # Function of (batch, rng)
        sampler_fn = make_generative_sampler(
            net=net,
            num_mean_samples=num_mean_samples,
            sample_dt=sample_dt,
            scalers=scalers,
            coarseners=coarseners,
            input_channels=input_channels,
        )
        sample, mean = sampler_fn(
            batch=jax.tree_map(lambda arr: jnp.expand_dims(arr, 0), batch_elem),
            rng=rng,
        )
        sample = jnp.squeeze(sample, 0)
        mean = jnp.squeeze(mean, 0)
        return None, (sample, mean)

    def compute_stats(batch, rng):
        batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0]
        rng_ctr, *rngs = jax.random.split(rng, batch_size + 1)
        rngs = jnp.stack(rngs)
        _carry, (samples, means) = jax.lax.scan(
            map_gen_sampler,
            None,
            (batch, rngs),
        )
        # Add time dimensions and compute stats
        output_size = samples.shape[-1]
        true_forcings = batch.q_total_forcings[output_size]
        stats = qg_spec_diag.subgrid_scores(
            true=jnp.expand_dims(true_forcings, 1),
            mean=jnp.expand_dims(means, 1),
            gen=jnp.expand_dims(samples, 1),
        )
        return stats, rng_ctr

    return compute_stats


def run_generative_offline_stats(net, eval_file, num_step_samples, spectrum_seed, num_mean_samples, base_logger, rng_ctr, sample_dt, scalers, coarseners, input_channels, output_size, min_sample_step=0):
    logger = base_logger.getChild("spec_offline")
    np_rng = np.random.default_rng(seed=spectrum_seed)
    logger.info("Starting to load evaluation data")
    with SimpleQGLoader(eval_file, fields=determine_required_fields(input_channels=input_channels, output_size=output_size)) as eval_loader:
        # First, sample the (traj, index) pairs used for "offline" statistics
        valid_step_count = eval_loader.num_steps - min_sample_step
        total_steps = eval_loader.num_trajs * valid_step_count
        sample_idxs = np_rng.integers(0, total_steps, size=num_step_samples, dtype=np.uint64)
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
    stats_fn = jax.jit(
        make_generative_stat_computer(
            net=net,
            num_mean_samples=num_mean_samples,
            sample_dt=sample_dt,
            scalers=scalers,
            coarseners=coarseners,
            input_channels=input_channels,
        )
    )
    stats, rng_ctr = stats_fn(batch, rng_ctr)
    stats = jax.device_get(stats)
    logger.info("Finished statistics computation")
    logger.info(f"Stat l2_mean: {stats.l2_mean:<15.5g}")
    logger.info(f"Stat l2_total: {stats.l2_total:<15.5g}")
    logger.info(f"Stat l2_residual: {stats.l2_residual:<15.5g}")
    logger.info(f"Stat var_ratio: {stats.var_ratio}")
    # Split results into json values and bulk values
    json_results = {
        "l2_mean": stats.l2_mean.item(),
        "l2_total": stats.l2_total.item(),
        "l2_residual": stats.l2_residual.item(),
        "var_ratio": stats.var_ratio.tolist(),
    }
    bulk_results = {
        "l2_mean": stats.l2_mean,
        "l2_total": stats.l2_total,
        "l2_residual": stats.l2_residual,
        "var_ratio": stats.var_ratio,
        "trajs": sample_trajs,
        "steps": sample_steps,
    }
    return json_results, bulk_results, rng_ctr

# TODO:
# 1. Draw sample steps, compute forcings, compare with Pavel's offline metrics for true forcings
# 2. Compute and store spectra as in the Ross paper for some steps (all steps for rolled out trajectories? Maybe a sub sample? should we aggregate across steps?)
# 3. Store the spectra and forcing statistics in the output stats file and log (Ross paper does time-averaged KE spectra, compute these)


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
    # Load the network
    net, net_info = load_network(weight_file=weight_file)
    input_channels = net_info["input_channels"]
    output_size = net_info["output_size"]
    required_fields = determine_required_fields(
        input_channels=input_channels,
        output_size=output_size,
    )
    # Make scalers and coarseners
    scalers = make_scalers(
        source_data=pathlib.Path(args.train_set) / "shuffled.hdf5",
        target_size=output_size,
    )
    coarseners = make_coarseners(
        source_data=pathlib.Path(args.train_set) / "shuffled.hdf5",
        target_size=output_size,
        input_channels=input_channels,
    )

    # Set up results files
    json_results = {}
    bulk_results = {}

    # GENERATIVE OFFLINE STATISTICS
    # Sample a few snapshots from the file, sample forcings and compute statistics vs the true forcing
    gen_json_res, gen_bulk_res, rng_ctr= run_generative_offline_stats(
        net=net,
        eval_file=eval_file,
        num_step_samples=args.num_spectrum_samples,
        spectrum_seed=args.spectrum_seed,
        num_mean_samples=args.num_mean_samples,
        base_logger=logger,
        rng_ctr=rng_ctr,
        sample_dt=args.sampling_dt,
        scalers=scalers,
        coarseners=coarseners,
        input_channels=input_channels,
        output_size=output_size,
        min_sample_step=args.spectrum_min_sample_step,
    )
    # Add results to pending data
    json_results["gen_offline_stats"] = gen_json_res
    bulk_results.update({f"gen_offline_stats_{k}": v for k, v in gen_bulk_res.items()})

    # Save results
    np.savez(out_dir / "results.npz", **bulk_results)
    with open(out_dir / "results.json", "w", encoding="utf8") as json_file:
        json.dump(json_results, json_file)
    logger.info("Finished evaluation")

if __name__ == "__main__":
    main()
