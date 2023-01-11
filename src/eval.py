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
from train_sdegm import make_sampler, make_scalers
import jax_utils


parser = argparse.ArgumentParser(description="Evaluate neural networks for closure")
parser.add_argument("net_dir", type=str, help="Directory with saved network weights")
parser.add_argument("eval_set", type=str, help="Directory with evaluation examples")
parser.add_argument("net_type", type=str, help="Which saved network weights to load")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
parser.add_argument("--sampling_dt", type=float, default=0.01, help="Time step size for the SDE generating samples")
parser.add_argument("--seed", type=int, default=None, help="Seed to use with RNG (if None, select automatically)")
# Settings for generative spectrum tests
parser.add_argument("--num_spectrum_samples", type=int, default=100, help="Number of random steps to test for generative statistics")
parser.add_argument("--spectrum_seed", type=int, default=0, help="Seed used to draw samples for generative statistics")
parser.add_argument("--num_mean_samples", type=int, default=10, help="Number of samples to compute mean of generated forcing")
parser.add_argument("--spectrum_min_sample_step", type=int, default=0, help="Starting step index for sampling (can be set to exclude warmup)")

def mse_loss(real_step, est_step):
    err = jnp.abs(est_step - real_step)
    return jnp.mean(err ** 2)


def relerr_loss(real_step, est_step):
    err = jnp.abs(est_step - real_step)
    return jnp.mean(err / jnp.abs(real_step))


def load_network(weight_file):
    net = GZFCNN(
        img_size=64,
        n_layers_in=4,
        n_layers_out=2,
        key=jax.random.PRNGKey(0),
    )
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
    return net


def make_eval_traj_computer(small_model, num_steps, observe_interval=1, q_param_func=None):
    total_steps = num_steps * observe_interval

    def state_scan(carry, x):
        state, param_state = carry

        def wrap_q_param(system_state):
            nonlocal param_state
            dq, param_state = q_param_func(system_state, param_state)
            return dq

        new_state = small_model.step_forward(state, q_param_func=wrap_q_param if q_param_func is not None else None)
        return (new_state, param_state), new_state

    def do_eval(initial_state, param_state=None):
        (final_state, final_param_state), observed_states = jax_utils.strided_scan(
            state_scan,
            init=(initial_state, param_state),
            xs=None,
            length=num_steps,
            stride=observe_interval,
        )
        return observed_states

    return do_eval


def make_simple_sampler(net, dt=0.01, sample_shape=(2, 64, 64)):
    q_scaler, forcing_scaler = make_scalers()
    base_sampler = make_sampler(
        dt=dt,
        sample_shape=sample_shape,
        q_scaler=q_scaler,
        forcing_scaler=forcing_scaler,
    )
    train_state = types.SimpleNamespace(net=net)

    def draw_sample(q, rng):
        assert q.ndim == 3
        # Expand to have batch and time dimensions
        batch_q = jnp.expand_dims(q, (0, 1))
        sample, _new_rng = base_sampler(train_state, batch_q, rng)
        return jnp.squeeze(sample, 0)

    return draw_sample


def make_net_eval_traj_computer(net, small_model, num_steps, observe_interval=1, dt=0.01):
    q_scaler, forcing_scaler = make_scalers()
    draw_samples = make_simple_sampler(
        net=net,
        dt=dt,
    )

    def q_param(state, key):
        # Add batch and time dimensions
        rng_ctr, rng = jax.random.split(key, 2)
        # Draw single sample for q subgrid forcing
        sample = draw_samples(state.q, rng)
        return sample, rng_ctr

    return make_eval_traj_computer(
        small_model=small_model,
        num_steps=num_steps,
        observe_interval=observe_interval,
        q_param_func=q_param,
    )


def run_generative_offline_stats(net, eval_file, num_step_samples, spectrum_seed, num_mean_samples, base_logger, rng_ctr, sample_dt, min_sample_step=0):
    sampler = make_simple_sampler(
        net=net,
        dt=sample_dt,
    )

    def inner_sample(q, rng_ctr):
        rng = jax.random.split(rng_ctr, num_mean_samples)
        return jax.vmap(sampler, in_axes=(None, 0))(q, rng)

    @jax.jit
    def compute_stats(q_stack, forcing_stack, rng_ctr):
        assert q_stack.shape == forcing_stack.shape
        num_qs = q_stack.shape[0]
        rng = jax.random.split(rng_ctr, num_qs)
        mean_samples = jax.vmap(inner_sample)(q_stack, rng)
        # Mean samples has shape (num_qs, num_means, forcing...)
        single_samples = mean_samples[:, 0]
        means = jnp.mean(mean_samples, axis=1)
        stats = qg_spec_diag.subgrid_scores(
            true=jnp.expand_dims(forcing_stack, 1),
            mean=jnp.expand_dims(means, 1),
            gen=jnp.expand_dims(single_samples, 1),
        )
        return stats

    logger = base_logger.getChild("spec_offline")
    np_rng = np.random.default_rng(seed=spectrum_seed)
    logger.info("Starting to load evaluation data")
    with SimpleQGLoader(eval_file, fields=["q", "q_total_forcing"]) as eval_loader:
        # First, sample the (traj, index) pairs used for "offline" statistics
        valid_step_count = eval_loader.num_steps - min_sample_step
        total_steps = eval_loader.num_trajs * valid_step_count
        sample_idxs = np_rng.integers(0, total_steps, size=num_step_samples, dtype=np.uint64)
        sample_trajs, sample_steps = np.divmod(sample_idxs, valid_step_count)
        sample_steps = sample_steps + min_sample_step
        # For each of the selected steps and trajectories, load the q component and true forcing
        # Ship them to the device and stack
        q_stack = []
        forcing_stack = []
        for traj, step in zip(sample_trajs, sample_steps):
            loaded = eval_loader.get_trajectory(traj=operator.index(traj), start=operator.index(step), end=operator.index(step) + 1)
            q_stack.append(loaded.q)
            forcing_stack.append(loaded.q_total_forcing)
        loaded = None
    q_stack = jnp.concatenate(q_stack, axis=0)
    forcing_stack = jnp.concatenate(forcing_stack, axis=0)
    logger.info("Finished loading evaluation data")
    # With loaded data on device, compute statistics
    logger.info("Starting statistics computation")
    stats = jax.device_get(
        compute_stats(
            q_stack=q_stack,
            forcing_stack=forcing_stack,
            rng_ctr=rng_ctr,
        )
    )
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
    return json_results, bulk_results

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
    # Create the model
    eval_small_model = qg_model_from_hdf5(file_path=eval_file, model="small")
    # Load the network
    net = load_network(weight_file=weight_file)
    # Set up results files
    json_results = {}
    bulk_results = {}

    # GENERATIVE OFFLINE STATISTICS
    # Sample a few snapshots from the file, sample forcings and compute statistics vs the true forcing
    rng_ctr, rng = jax.random.split(rng_ctr, 2)
    gen_json_res, gen_bulk_res = run_generative_offline_stats(
        net=net,
        eval_file=eval_file,
        num_step_samples=args.num_spectrum_samples,
        spectrum_seed=args.spectrum_seed,
        num_mean_samples=args.num_mean_samples,
        base_logger=logger,
        rng_ctr=rng,
        sample_dt=args.sampling_dt,
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
