import argparse
import pathlib
import math
import re
import os
import random
import itertools
import contextlib
import h5py
import json
import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
import optax
import numpy as np
import logging
import time
import functools
from typing import Callable
from systems.qg.qg_model import QGModel
from systems.qg import utils as qg_utils
from systems.qg.loader import ThreadedQGLoader, SimpleQGLoader, qg_model_from_hdf5
from methods.unet import UNet
import jax_utils
import utils


parser = argparse.ArgumentParser(description="Train neural networks for closure")
parser.add_argument("out_dir", type=str, help="Directory to store output (created if non-existing)")
parser.add_argument("train_set", type=str, help="Directory with training examples")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
parser.add_argument("--save_interval", type=int, default=1, help="Number of epochs between saves")
parser.add_argument("--seed", type=int, default=None, help="Seed to use with RNG (if None, select automatically)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batches_per_epoch", type=int, default=100, help="Training batches per epoch")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for optimizer")
parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
parser.add_argument("--dt", type=float, default=0.01, help="Time step size when running diffusion")
parser.add_argument("--num_epoch_samples", type=int, default=15, help="Number of samples to draw after each epoch")
parser.add_argument("--num_hutch_samples", type=int, default=100, help="Number of samples to use when estimating the Jacobian")


def save_network(output_name, output_dir, state, base_logger=None):
    if base_logger is None:
        logger = logging.getLogger("save")
    else:
        logger = base_logger.getChild("save")
    output_dir = pathlib.Path(output_dir)
    eqx.tree_serialise_leaves(output_dir / f"{output_name}.eqx.PART", state.net)
    try:
        os.remove(output_dir / f"{output_name}.eqx")
    except FileNotFoundError:
        pass
    os.rename(output_dir / f"{output_name}.eqx.PART", output_dir / f"{output_name}.eqx")
    logger.info("Saved network parameters to %s in %s", output_name, output_dir)


def make_ou_solver(dt, t0=0.0, t1=1.0):
    num_steps = math.floor((t1 - t0) / dt)

    def drift(t, y, args):
        return -y

    def diffuse_control(t, y, args):
        return math.sqrt(2)

    def get_steps(snapshot, rng):
        brownian = diffrax.UnsafeBrownianPath(shape=snapshot.shape, key=rng)
        control_noise = diffrax.WeaklyDiagonalControlTerm(diffuse_control, brownian)
        terms = diffrax.MultiTerm(diffrax.ODETerm(drift), control_noise)
        solver = diffrax.Euler()
        saveat = diffrax.SaveAt(steps=True)
        solve = diffrax.diffeqsolve(terms, solver, t0, t1, dt0=dt, y0=snapshot, saveat=saveat, adjoint=diffrax.NoAdjoint())
        data = solve.ys[:num_steps]
        times = solve.ts[:num_steps]
        return jnp.concatenate([jnp.expand_dims(snapshot, 0), data]), jnp.concatenate([jnp.expand_dims(jnp.zeros_like(times[0]), 0), times])

    return get_steps


def make_epoch_computer(dt, batch_size, train_data, num_steps, num_hutch_samples):

    def sample_loss(snapshot, t, rng, net):
        net_rng, hutch_rng = jax.random.split(rng, 2)
        orig_shape = snapshot.shape
        est_trace_jac = jax_utils.trace_jac_hutch(
            lambda x: net(x.reshape(orig_shape), t, key=net_rng).ravel(),
            snapshot.ravel(),
            rng=hutch_rng,
            num_samples=num_hutch_samples,
        )
        est_norm = jnp.sum(net(snapshot, t, key=net_rng)**2) / 2
        return est_trace_jac + est_norm

    def batch_loss(net, snapshots, ts, rng):
        n_batch = snapshots.shape[0]
        rngs = jnp.stack(jax.random.split(rng, n_batch))
        losses = jax.vmap(functools.partial(sample_loss, net=net))(snapshots, ts, rngs)
        return jnp.mean(losses)

    def batch_sde(snapshots, rng):
        n_batch = snapshots.shape[0]
        rngs = jnp.stack(jax.random.split(rng, n_batch))
        ode_fwd = make_ou_solver(dt=dt, t0=0.0, t1=1.0)
        ys, ts = jax.vmap(ode_fwd)(snapshots, rngs)
        return ys, ts

    def do_batch(carry, _x, m_fixed):
        rng_ctr, m_vary = carry
        state = eqx.combine(m_vary, m_fixed)
        rng_batch, rng_sde, rng_times, rng_loss, rng_ctr = jax.random.split(rng_ctr, 5)
        batch_idx = jax.random.randint(rng_batch, (batch_size, ), 0, train_data.shape[0], dtype=jnp.uint32)
        batch = jnp.take(train_data, batch_idx, axis=0)
        # Run the ou process
        snaps, ts = batch_sde(batch, rng_sde)
        # Select times
        time_idx = jax.random.randint(rng_times, (batch_size, 1), 0, ts.shape[1], dtype=jnp.uint32)
        snaps = jax.vmap(functools.partial(jnp.take, axis=0))(snaps, time_idx).squeeze(axis=1)
        ts = jax.vmap(functools.partial(jnp.take, axis=0))(ts, time_idx)
        # Compute loss
        loss, grads = eqx.filter_value_and_grad(batch_loss)(state.net, snaps, ts, rng=rng_loss)
        # Update parameters
        out_state = state.apply_updates(grads)
        out_state_vary, _out_state_fixed = eqx.partition(state, eqx.is_array)
        return (rng_ctr, out_state_vary), loss

    def do_epoch(state, rng):
        m_vary, m_fixed = eqx.partition(state, eqx.is_array)
        (last_rng, new_vary), losses = jax.lax.scan(
            functools.partial(do_batch, m_fixed=m_fixed),
            (rng, m_vary),
            None,
            length=num_steps
        )
        new_state = eqx.combine(new_vary, m_fixed)
        return new_state, last_rng, losses

    return do_epoch


def init_network(lr, weight_decay, rng):
    net = UNet(key=rng)
    if weight_decay != 0:
        optim = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    else:
        optim = optax.adam(learning_rate=lr)
    state = jax_utils.EquinoxTrainState(
        net=net,
        optim=optim,
    )
    return state


def make_scalers(data):
    assert data.ndim == 4
    assert data.shape[1] == 2
    assert data.shape[-1] == data.shape[-2]
    mean = np.expand_dims(np.asarray(jnp.mean(data, axis=(0, -1, -2))), (-1, -2))
    std = np.expand_dims(np.asarray(jnp.std(data, axis=(0, -1, -2))), (-1, -2))

    def scaler(snapshot):
        return (snapshot - mean) / std

    def unscaler(snapshot):
        return (snapshot * std) + mean

    def scale_deriv(deriv):
        return deriv / std

    return scaler, unscaler, scale_deriv, {
        "mean": tuple(mean.squeeze()),
        "std": tuple(std.squeeze()),
    }


def main():
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    if out_dir.is_file():
        raise ValueError(f"Path must be a directory, not a file: {args.out_dir}")
    out_dir.mkdir(exist_ok=True)
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
    # Select seed
    if args.seed is None:
        seed = random.SystemRandom().randint(0, 2**32)
    else:
        seed = args.seed
    logger.info("Using seed %d", seed)

    # Configure required elements for training
    rng_ctr = jax.random.PRNGKey(seed=seed)
    train_path = pathlib.Path(args.train_set) / "data.hdf5"
    train_small_model = qg_model_from_hdf5(file_path=train_path, model="small")
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    # Construct neural net
    rng, rng_ctr = jax.random.split(rng_ctr, 2)
    logger.info("Training network: unet")
    state = init_network(
        lr=args.lr,
        weight_decay=args.weight_decay,
        rng=rng,
    )

    # Load dataset
    logger.info("Loading data")
    with h5py.File(train_path, "r") as train_file:
        train_data = train_file["final_q_steps"][:]
    # Mask out NaNs, put batches at the back, and move to GPU
    train_data = jax.device_put(train_data[np.all(np.isfinite(train_data), axis=(-1, -2, -3))])
    # Create data normalizer and its inverse
    scaler, unscaler, scale_deriv, data_stats = make_scalers(train_data)
    logger.info("Finished loading data, mean=(%g, %g), std=(%g, %g)", data_stats["mean"][0], data_stats["mean"][1], data_stats["std"][0], data_stats["std"][1])
    # Rescale data to be closer to standard normal per-channel
    train_data = jax.vmap(scaler)(train_data)

    # Training functions
    train_epoch_fn = eqx.filter_jit(
        make_epoch_computer(
            dt=args.dt,
            batch_size=args.batch_size,
            train_data=train_data,
            num_steps=args.batches_per_epoch,
            num_hutch_samples=args.num_hutch_samples,
        )
    )

    min_mean_loss = None
    train_rng, rng_ctr = jax.random.split(rng_ctr, 2)
    for epoch in range(args.num_epochs):
        # Training
        logger.info("Starting epoch %d of %d", epoch + 1, args.num_epochs)
        epoch_start = time.perf_counter()
        state, train_rng, losses = train_epoch_fn(state, train_rng)
        mean_loss = jax.device_get(jnp.mean(losses))
        final_loss = jax.device_get(losses[-1])
        epoch_end = time.perf_counter()
        logger.info("Finished epoch %d in %f sec", epoch + 1, epoch_end - epoch_start)
        logger.info("Epoch %d mean loss %f", epoch + 1, mean_loss)
        logger.info("Epoch %d final loss %f", epoch + 1, final_loss)

        # Validation

        # Save weights
        if min_mean_loss is None or (np.isfinite(mean_loss) and mean_loss <= min_mean_loss):
            min_mean_loss = mean_loss
            save_network("best_loss", output_dir=weights_dir, state=state, base_logger=logger)
        if epoch % args.save_interval == 0:
            save_network("interval", output_dir=weights_dir, state=state, base_logger=logger)


if __name__ == "__main__":
    main()
