import argparse
import pathlib
import math
import os
import random
import contextlib
import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
import optax
import numpy as np
import logging
import time
import functools
from systems.qg.loader import ThreadedQGLoader, SimpleQGLoader
from methods.gz_fcnn import GZFCNN
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
parser.add_argument("--dt", type=float, default=0.01, help="Time step size when running diffusion")
parser.add_argument("--num_epoch_samples", type=int, default=15, help="Number of samples to draw after each epoch")


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


def make_epoch_computer(batch_size, num_steps, loss_weight_func=None):
    # OU: dY = -0.5 * g(t)
    # beta(t) = 18 * t^2
    # int(beta)(t) = 6 * t^3
    t0 = 0.0
    t1 = 1.0

    def int_beta_func(t):
        return 6 * t**3

    min_variance = 1e-6

    if loss_weight_func is None:
        loss_weight_func = lambda t: 1 - jnp.exp(-int_beta_func(t))

    def sample_loss(snapshot, t, rng, net):
        mean = snapshot * jnp.exp(-0.5 * int_beta_func(t))
        var = jnp.maximum(min_variance, 1 - jnp.exp(-int_beta_func(t)))
        std = jnp.sqrt(var)
        noise = jax.random.normal(rng, shape=snapshot.shape)
        y = mean + std * noise
        pred = net(y, t)
        return loss_weight_func(t) * jnp.mean((pred + noise / std) ** 2)

    def batch_loss(net, snapshots, ts, rng):
        n_batch = snapshots.shape[0]
        rngs = jnp.stack(jax.random.split(rng, n_batch))
        losses = jax.vmap(functools.partial(sample_loss, net=net))(snapshots, ts, rngs)
        return jnp.mean(losses)

    def do_batch(carry, _x, m_fixed, train_data):
        rng_ctr, m_vary = carry
        state = eqx.combine(m_vary, m_fixed)
        # Produce RNGs
        rng_times, rng_batch, rng_loss, rng_ctr = jax.random.split(rng_ctr, 4)
        # Sample times (one in each time bucket)
        times = jax.random.uniform(rng_times, shape=(batch_size, ), minval=0, maxval=((t1 - t0) / batch_size), dtype=jnp.float32)
        times = times + jnp.arange(batch_size, dtype=jnp.float32) * ((t1 - t0) / batch_size)
        # Select batch
        batch_idx = jax.random.randint(rng_batch, (batch_size, ), 0, train_data.shape[0], dtype=jnp.uint32)
        batch = jnp.take(train_data, batch_idx, axis=0)
        # Compute losses
        loss, grads = eqx.filter_value_and_grad(batch_loss)(state.net, batch, times, rng=rng_loss)
        # Update parameters (if loss is finite)
        out_state = jax.lax.cond(
            jnp.isfinite(loss),
            lambda: state.apply_updates(grads),
            lambda: state,
        )
        out_state_vary, _out_state_fixed = eqx.partition(out_state, eqx.is_array)
        return (rng_ctr, out_state_vary), loss

    def do_epoch(state, rng, train_data):
        m_vary, m_fixed = eqx.partition(state, eqx.is_array)
        (last_rng, new_vary), losses = jax.lax.scan(
            functools.partial(do_batch, m_fixed=m_fixed, train_data=train_data),
            (rng, m_vary),
            None,
            length=num_steps
        )
        new_state = eqx.combine(new_vary, m_fixed)
        return new_state, last_rng, losses

    return do_epoch


def make_sampler(dt, num_samples, sample_shape):
    # OU: dY = -0.5 * g(t)
    # g(t) = t
    # int(g)(t) = t^2/2
    # int(g^2)(t) = t^3/3
    t0 = 0.0
    t1 = 1.0
    max_steps = math.ceil((t1 - t0) / dt) + 2

    def beta_func(t):
        return 18 * t**2

    def int_beta_func(t):
        return 6 * t**3

    def drift(t, y, args):
        net = args
        beta = beta_func(t)
        return -0.5 * beta * (y + net(y, t))

    def draw_single_sample(rng, net):
        snapshot = jax.random.normal(rng, sample_shape, dtype=jnp.float32)
        terms = diffrax.ODETerm(drift)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(t1=True)
        sol = diffrax.diffeqsolve(terms, solver, t1, t0, -dt, snapshot, saveat=saveat, adjoint=diffrax.NoAdjoint(), max_steps=max_steps, args=net)
        return sol.ys[0]

    def draw_samples(net, rng):
        sample_rngs = jnp.stack(jax.random.split(rng, num_samples))
        samples = jax.vmap(functools.partial(draw_single_sample, net=net))(sample_rngs)
        return samples

    return draw_samples

def init_network(lr, rng):
    net = GZFCNN(
        img_size=64,
        n_layers=2,
        key=rng,
    )
    optim = optax.adabelief(learning_rate=lr)
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
    logger.info("Training network: gzfcnn")
    state = init_network(
        lr=args.lr,
        rng=rng,
    )

    # Load dataset
    logger.info("Loading data")
    with h5py.File(train_path, "r") as train_file:
        train_data = jax.device_put(train_file["final_q_steps"][:])
    # Mask out NaNs, put batches at the back, and move to GPU
    train_data = train_data[jnp.all(jnp.isfinite(train_data), axis=(-1, -2, -3))]
    # Create data normalizer and its inverse
    scaler, unscaler, scale_deriv, data_stats = make_scalers(train_data)
    logger.info("Finished loading data, mean=(%g, %g), std=(%g, %g)", data_stats["mean"][0], data_stats["mean"][1], data_stats["std"][0], data_stats["std"][1])
    # Rescale data to be closer to standard normal per-channel
    train_data = jax.vmap(scaler)(train_data)

    # Training functions
    train_epoch_fn = eqx.filter_jit(
        make_epoch_computer(
            batch_size=args.batch_size,
            num_steps=args.batches_per_epoch,
        )
    )
    val_epoch_fn = eqx.filter_jit(
        make_sampler(
            dt=args.dt,
            num_samples=args.num_epoch_samples,
            sample_shape=train_data.shape[1:],
        )
    )

    min_mean_loss = None
    train_rng, rng_ctr = jax.random.split(rng_ctr, 2)
    for epoch in range(args.num_epochs):
        # Training
        logger.info("Starting epoch %d of %d", epoch + 1, args.num_epochs)
        epoch_start = time.perf_counter()
        state, train_rng, losses = train_epoch_fn(state, train_rng, train_data)
        mean_loss = jax.device_get(jnp.mean(losses))
        final_loss = jax.device_get(losses[-1])
        epoch_end = time.perf_counter()
        logger.info("Finished epoch %d in %f sec", epoch + 1, epoch_end - epoch_start)
        logger.info("Epoch %d mean loss %f", epoch + 1, mean_loss)
        logger.info("Epoch %d final loss %f", epoch + 1, final_loss)

        # Validation
        logger.info("Starting sample draw for epoch %d", epoch + 1)
        val_rng, train_rng = jax.random.split(train_rng, 2)
        sample_start = time.perf_counter()
        val_samples = np.asarray(jax.vmap(unscaler)(val_epoch_fn(state.net, val_rng)))
        sample_end = time.perf_counter()
        np.save(samples_dir / f"samples_{epoch + 1:05d}.npy", val_samples, allow_pickle=False)
        logger.info("Finished sample draw for epoch %d in %f sec", epoch + 1, sample_end - sample_start)
        val_mean = np.mean(val_samples, axis=(0, -1, -2))
        val_std = np.std(val_samples, axis=(0, -1, -2))
        logger.info("Val sample mean=(%g, %g) sample std=(%g, %g)", val_mean[0], val_mean[1], val_std[0], val_std[1])

        # Save weights
        if min_mean_loss is None or (np.isfinite(mean_loss) and mean_loss <= min_mean_loss):
            min_mean_loss = mean_loss
            save_network("best_loss", output_dir=weights_dir, state=state, base_logger=logger)
        if epoch % args.save_interval == 0:
            save_network("interval", output_dir=weights_dir, state=state, base_logger=logger)


if __name__ == "__main__":
    main()
