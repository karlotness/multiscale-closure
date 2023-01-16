import argparse
import pathlib
import math
import os
import random
import contextlib
import itertools
import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
import optax
import numpy as np
import logging
import time
import functools
from systems.qg.loader import ThreadedPreShuffledSnapshotLoader, SimpleQGLoader
from methods.gz_fcnn import GZFCNN
import jax_utils
import diffrax_utils
import utils


parser = argparse.ArgumentParser(description="Train neural networks for closure")
parser.add_argument("out_dir", type=str, help="Directory to store output (created if non-existing)")
parser.add_argument("train_set", type=str, help="Directory with training examples")
parser.add_argument("val_set", type=str, help="Directory with validation examples")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
parser.add_argument("--save_interval", type=int, default=1, help="Number of epochs between saves")
parser.add_argument("--seed", type=int, default=None, help="Seed to use with RNG (if None, select automatically)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batches_per_epoch", type=int, default=100, help="Training batches per epoch")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for optimizer")
parser.add_argument("--dt", type=float, default=0.01, help="Time step size when running diffusion")
parser.add_argument("--num_val_samples", type=int, default=5, help="Number of samples to draw in each validation period")
parser.add_argument("--val_interval", type=int, default=1, help="Number of epochs between validation periods")


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


def make_batch_computer(batch_size, q_scaler, forcing_scaler, loss_weight_func=None):
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

    def sample_loss(snap_q, snap_forcing, t, rng, net):
        mean = snap_forcing * jnp.exp(-0.5 * int_beta_func(t))
        var = jnp.maximum(min_variance, 1 - jnp.exp(-int_beta_func(t)))
        std = jnp.sqrt(var)
        noise = jax.random.normal(rng, shape=snap_forcing.shape)
        y = mean + std * noise
        net_input = jnp.concatenate([y, snap_q], axis=0)
        pred = net(net_input, t)
        return loss_weight_func(t) * jnp.mean((pred + noise / std) ** 2)

    def batch_loss(net, snap_q, snap_forcing, ts, rng):
        assert snap_q.shape == snap_forcing.shape
        n_batch = snap_q.shape[0]
        rngs = jnp.stack(jax.random.split(rng, n_batch))
        losses = jax.vmap(functools.partial(sample_loss, net=net))(snap_q, snap_forcing, ts, rngs)
        return jnp.mean(losses)

    def do_batch(batch, state, rng):
        # Extract batch components
        batch_q = jax.vmap(q_scaler.scale)(batch.q)
        batch_q_forcing = jax.vmap(forcing_scaler.scale)(batch.q_total_forcing)
        # Produce RNGs
        rng_times, rng_loss, rng_ctr = jax.random.split(rng, 3)
        # Sample times (one in each time bucket)
        times = jax.random.uniform(rng_times, shape=(batch_size, ), minval=0, maxval=((t1 - t0) / batch_size), dtype=jnp.float32)
        times = times + jnp.arange(batch_size, dtype=jnp.float32) * ((t1 - t0) / batch_size)
        # Compute losses
        loss, grads = eqx.filter_value_and_grad(batch_loss)(state.net, batch_q, batch_q_forcing, times, rng=rng_loss)
        # Update parameters (if loss is finite)
        out_state = state.apply_updates(grads)
        return out_state, rng_ctr, loss

    return do_batch


def do_epoch(train_state, train_rng, batch_iter, batch_fn, logger=None):
    if logger is None:
        logger = logging.getLogger("train_epoch")
    epoch_start = time.perf_counter()
    losses = []
    for batch in batch_iter:
        train_state, train_rng, batch_loss = batch_fn(batch, train_state, train_rng)
        losses.append(batch_loss)
    epoch_end = time.perf_counter()
    mean_loss = jax.device_get(jnp.mean(jnp.stack(losses)))
    final_loss = jax.device_get(losses[-1])
    logger.info("Finished epoch in %f sec", epoch_end - epoch_start)
    logger.info("Epoch mean loss %f", mean_loss)
    logger.info("Epoch final loss %f", final_loss)
    return train_state, train_rng, {"mean_loss": mean_loss, "final_loss": final_loss}


def make_sampler(dt, sample_shape, q_scaler, forcing_scaler):
    # OU: dY = -0.5 * g(t)
    # g(t) = t
    # int(g)(t) = t^2/2
    # int(g^2)(t) = t^3/3
    t0 = 0.0
    t1 = 1.0
    max_steps = math.ceil((t1 - t0) / dt) + 2
    t0 = jnp.float32(t0)
    t1 = jnp.float32(t1)
    dt = jnp.float32(dt)

    def beta_func(t):
        return jnp.float32(18 * t**2)

    def int_beta_func(t):
        return jnp.float32(6 * t**3)

    def drift(t, y, args):
        net, q = args
        beta = beta_func(t)
        net_input = jnp.concatenate([y, q], axis=0)
        return -0.5 * beta * (y + net(net_input, t))

    def draw_single_sample(batch_q, rng, net):
        snapshot = jax.random.normal(rng, sample_shape, dtype=jnp.float32)
        terms = diffrax.ODETerm(drift)
        solver = diffrax_utils.Tsit5Float32()
        saveat = diffrax.SaveAt(t1=True)
        sol = diffrax.diffeqsolve(terms, solver, t1, t0, -dt, snapshot, saveat=saveat, adjoint=diffrax_utils.NoAdjointFloat32(), max_steps=max_steps, args=(net, batch_q))
        return sol.ys[0]

    def draw_samples(state, batch_q, rng):
        net = state.net
        batch_q = jax.vmap(q_scaler.scale)(jnp.squeeze(batch_q, 1))
        num_samples = batch_q.shape[0]
        rng_ctr, *sample_rngs = jax.random.split(rng, num_samples + 1)
        sample_rngs = jnp.stack(sample_rngs)
        samples = jax.vmap(functools.partial(draw_single_sample, net=net))(batch_q, sample_rngs)
        samples = jax.vmap(forcing_scaler.unscale)(samples)
        return samples, rng_ctr

    return draw_samples


def do_validation(train_state, val_rng, np_rng, loader, sample_fn, num_samples, logger=None):
    if logger is None:
        logger = logging.getLogger("validation")
    # Sample indices
    traj = np_rng.integers(low=0, high=loader.num_trajs, size=num_samples)
    step = np_rng.integers(low=0, high=loader.num_steps, size=num_samples)
    # Load and stack q components
    logger.info("Starting validation, drawing %d samples", num_samples)
    val_start = time.perf_counter()
    batch_q = np.stack([loader.get_trajectory(traj=t, start=s, end=s+1).q for t, s in zip(traj, step)])
    samples, rng_ctr = sample_fn(state=train_state, batch_q=batch_q, rng=val_rng)
    samples = jax.device_get(samples)
    val_end = time.perf_counter()
    logger.info("Finished drawing %d samples in %f sec", num_samples, val_end - val_start)
    return samples, rng_ctr


def init_network(lr, rng):
    net = GZFCNN(
        img_size=64,
        n_layers_in=4,
        n_layers_out=2,
        key=rng,
    )
    optim = optax.adabelief(learning_rate=lr)
    state = jax_utils.EquinoxTrainState(
        net=net,
        optim=optim,
    )
    return state


@jax.tree_util.register_pytree_node_class
class Scaler:
    def __init__(self, mean, var):
        self.mean = jnp.expand_dims(jnp.asarray(mean, dtype=jnp.float32), (-1, -2))
        self.var = jnp.expand_dims(jnp.asarray(var, dtype=jnp.float32), (-1, -2))
        self.std = jnp.sqrt(self.var)

    def scale(self, a):
        return (a - self.mean) / self.std

    def unscale(self, a):
        return (a * self.std) + self.mean

    def tree_flatten(self):
        return (self.mean, self.var, self.std), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        mean, var, std = children
        obj = cls.__new__(cls)
        obj.mean = mean
        obj.var = var
        obj.std = std
        return obj


def make_scalers():
    stats_path = pathlib.Path(__file__).resolve().parent / "systems" / "qg" / "stats_op1.npz"
    with np.load(stats_path) as stats_file:
        q_scaler = Scaler(
            mean=stats_file["q_mean"],
            var=stats_file["q_var"],
        )
        forcing_scaler = Scaler(
            mean=stats_file["q_total_forcing_mean"],
            var=stats_file["q_total_forcing_var"],
        )
    return q_scaler, forcing_scaler


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
    np_rng = np.random.default_rng(seed=seed)

    # Configure required elements for training
    rng_ctr = jax.random.PRNGKey(seed=np_rng.integers(2**32).item())
    train_path = pathlib.Path(args.train_set) / "shuffled.hdf5"
    val_path = pathlib.Path(args.val_set) / "data.hdf5"
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
    # Create data normalizer and its inverse
    q_scaler, forcing_scaler = make_scalers()

    # Open data files
    with contextlib.ExitStack() as train_context:
        # Open data files
        train_loader = train_context.enter_context(
            ThreadedPreShuffledSnapshotLoader(
                file_path=train_path,
                batch_size=args.batch_size,
                buffer_size=10,
                chunk_size=32000,
                seed=np_rng.integers(2**32).item(),
                base_logger=logger.getChild("train_loader"),
            )
        )
        val_loader = train_context.enter_context(
            SimpleQGLoader(
                file_path=val_path,
                fields=["q"],
            )
        )

        # Training functions
        train_batch_fn = eqx.filter_jit(
            make_batch_computer(
                batch_size=args.batch_size,
                q_scaler=q_scaler,
                forcing_scaler=forcing_scaler,
            )
        )
        val_sample_fn = eqx.filter_jit(
            make_sampler(
                dt=args.dt,
                sample_shape=(2, 64, 64),
                q_scaler=q_scaler,
                forcing_scaler=forcing_scaler,
            )
        )

        # Running statistics
        min_mean_loss = None

        # Training loop
        for epoch in range(1, args.num_epochs + 1):
            logger.info("Starting epoch %d of %d", epoch, args.num_epochs)
            # Training step
            with contextlib.closing(train_loader.iter_batches()) as train_batch_iter:
                state, rng_ctr, epoch_stats = do_epoch(
                    train_state=state,
                    train_rng=rng_ctr,
                    batch_iter=itertools.islice(train_batch_iter, args.batches_per_epoch),
                    batch_fn=train_batch_fn,
                    logger=logger.getChild(f"{epoch:05d}_train"),
                )
            mean_loss = epoch_stats["mean_loss"]

            # Save snapshots
            if min_mean_loss is None or (math.isfinite(mean_loss) and mean_loss <= min_mean_loss):
                min_mean_loss = mean_loss
                save_network("best_loss", output_dir=weights_dir, state=state, base_logger=logger)
            if epoch % args.save_interval == 0:
                save_network("interval", output_dir=weights_dir, state=state, base_logger=logger)

            # Validation step
            if epoch % args.val_interval == 0:
                logger.info("Starting validation for epoch %d", epoch)
                val_samples, rng_ctr = do_validation(
                    train_state=state,
                    val_rng=rng_ctr,
                    np_rng=np_rng,
                    loader=val_loader,
                    sample_fn=val_sample_fn,
                    logger=logger.getChild(f"{epoch:05d}_val"),
                    num_samples=args.num_val_samples,
                )
                np.save(samples_dir / f"samples_{epoch:05d}.npy", val_samples, allow_pickle=False)
                logger.info("Saved samples to %s", f"samples_{epoch:05d}.npy")
                val_samples = None

            logger.info("Finished epoch %d", epoch)

    # End of training loop
    logger.info("Finished training")


if __name__ == "__main__":
    main()
