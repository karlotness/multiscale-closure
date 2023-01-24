import argparse
import dataclasses
import pathlib
import math
import os
import sys
import random
import contextlib
import itertools
import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
import optax
import h5py
import numpy as np
import logging
import time
import json
import functools
import operator
from systems.qg.loader import ThreadedPreShuffledSnapshotLoader, SimpleQGLoader
from systems.qg import coarsen, diagnostics as qg_spec_diag
from systems.qg.qg_model import QGModel
from methods import ARCHITECTURES
import jax_utils
from jax_utils import Scaler
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
parser.add_argument("--num_val_samples", type=int, default=10, help="Number of samples to draw in each validation period")
parser.add_argument("--val_mean_samples", type=int, default=25, help="Number of samples used to compute empirical means")
parser.add_argument("--val_interval", type=int, default=1, help="Number of epochs between validation periods")
parser.add_argument("--output_size", type=int, default=64, help="Scale of output forcing to generate")
parser.add_argument("--input_channels", type=str, nargs="+", default=["q_64"], help="Channels to show the network as input")
parser.add_argument("--processing_size", type=int, default=None, help="Size to user for internal network evaluation")
parser.add_argument("--architecture", type=str, default="gz-fcnn-v1", help="Network architecture to train")


def save_network(output_name, output_dir, state, base_logger=None):
    if base_logger is None:
        logger = logging.getLogger("save")
    else:
        logger = base_logger.getChild("save")
    output_dir = pathlib.Path(output_dir)
    with utils.rename_save_file_path(output_dir / f"{output_name}.eqx") as eqx_out_path:
        eqx.tree_serialise_leaves(eqx_out_path, state.net)
    logger.info("Saved network parameters to %s in %s", output_name, output_dir)


def determine_required_fields(input_channels, output_size):
    fields = {f"q_total_forcing_{output_size}"}
    for chan in input_channels:
        if chan.startswith("q_total_forcing_"):
            fields.add(chan)
        elif chan.startswith("q_"):
            fields.add("q")
        else:
            raise ValueError(f"Unsupported input field {chan}")
    return sorted(fields)


def determine_channel_sizes(input_channels):
    sizes = set()
    for chan in input_channels:
        if chan.startswith("q_total_forcing_"):
            sizes.add(int(chan[len("q_total_forcing_"):]))
        elif chan.startswith("q_"):
            sizes.add(int(chan[len("q_"):]))
        else:
            raise ValueError(f"Unsupported input field {chan}")
    return sizes


def determine_residual_size(input_channels, output_size):
    # The residual is the largest total_forcing channel with size <= output_size
    sizes = set()
    for chan in input_channels:
        if chan.startswith("q_total_forcing_"):
            sizes.add(int(chan[len("q_total_forcing_"):]))
    return max((s for s in sizes if s <= output_size), default=None)


def determine_processing_size(input_channels, target_size, user_processing_size=None):
    sizes = determine_channel_sizes(input_channels=input_channels)
    auto_processing_size = max(max(sizes), target_size)
    if user_processing_size is not None:
        user_processing_size = operator.index(user_processing_size)
        if user_processing_size < auto_processing_size:
            raise ValueError(f"invalid override processing size: must be at least {auto_processing_size}")
        return user_processing_size
    return auto_processing_size



def build_fixed_input_from_batch(batch, input_channels, scalers, coarseners):
    # For each input: 1. apply distribution scaling, then 2. (un)coarsen to processing_size
    input_stack = []
    for chan in sorted(set(input_channels)):
        if chan.startswith("q_total_forcing_"):
            # Stack a forcing channel
            size = int(chan[len("q_total_forcing_"):])
            input_stack.append(
                jax.vmap(coarseners[size])(
                    jax.vmap(scalers.q_total_forcing_scalers[size].scale_to_standard)(
                        batch.q_total_forcings[size]
                    )
                )
            )
        elif chan.startswith("q_"):
            # Stack the q channel
            # The q component is a special case since it gets downscaled
            # We rescale values *after* coarsening
            size = batch.q.shape[-1]
            q_val = jax.vmap(coarseners[size])(batch.q)
            processing_size = q_val.shape[-1]
            q_val = jax.vmap(scalers.q_scalers[processing_size].scale_to_standard)(q_val)
            input_stack.append(q_val)
        else:
            raise ValueError(f"Unsupported input field {chan}")
    return jnp.concatenate(input_stack, axis=-3)


def make_batch_computer(scalers, coarseners, input_channels, output_size):
    # OU: dY = -0.5 * g(t)
    # beta(t) = 18 * t^2
    # int(beta)(t) = 6 * t^3
    t0 = 0.0
    t1 = 1.0

    def int_beta_func(t):
        return 6 * t**3

    min_variance = 1e-6
    loss_weight_func = lambda t: 1 - jnp.exp(-int_beta_func(t))

    residual_size = determine_residual_size(input_channels, output_size)

    def sample_loss(fixed_input, targets, t, rng, net):
        mean = targets * jnp.exp(-0.5 * int_beta_func(t))
        var = jnp.maximum(min_variance, 1 - jnp.exp(-int_beta_func(t)))
        std = jnp.sqrt(var)
        noise = jax.random.normal(rng, shape=targets.shape, dtype=jnp.float32)
        y = mean + std * noise
        big_y = coarseners["output_rev"](y)
        net_input = jnp.concatenate([big_y, fixed_input], axis=0)
        pred = net(net_input, t)
        small_pred = coarseners["output"](pred)
        return loss_weight_func(t) * jnp.mean((small_pred + noise / std) ** 2)

    def batch_loss(net, fixed_input, targets, ts, rng):
        n_batch = targets.shape[0]
        rngs = jnp.stack(jax.random.split(rng, n_batch))
        losses = jax.vmap(functools.partial(sample_loss, net=net))(fixed_input, targets, ts, rngs)
        return jnp.mean(losses)

    def do_batch(batch, state, rng):
        # Extract batch components
        fixed_input = build_fixed_input_from_batch(
            batch=batch,
            input_channels=input_channels,
            scalers=scalers,
            coarseners=coarseners,
        )
        # If necessary, make sure we target the residual from the closest-sized input forcing channel
        targets = jax.vmap(scalers.q_total_forcing_scalers[output_size].scale_to_standard)(
            batch.q_total_forcings[output_size]
        )
        if residual_size is not None:
            existing_target = jax.vmap(coarseners["residual"])(
                jax.vmap(scalers.q_total_forcing_scalers[residual_size].scale_to_standard)(
                    batch.q_total_forcings[residual_size]
                )
            )
            targets = (targets - existing_target) / jnp.sqrt(2).astype(jnp.float32)
        batch_size = targets.shape[0]
        # Produce RNGs
        rng_times, rng_loss, rng_ctr = jax.random.split(rng, 3)
        # Sample times (one in each time bucket)
        times = jax.random.uniform(rng_times, shape=(batch_size, ), minval=0, maxval=((t1 - t0) / batch_size), dtype=jnp.float32)
        times = times + jnp.arange(batch_size, dtype=jnp.float32) * ((t1 - t0) / batch_size)
        # Compute losses
        loss, grads = eqx.filter_value_and_grad(batch_loss)(state.net, fixed_input, targets, times, rng=rng_loss)
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
    return train_state, train_rng, {"mean_loss": mean_loss.item(), "final_loss": final_loss.item(), "duration_sec": epoch_end - epoch_start}


def make_raw_sampler(coarseners, output_size, dt=0.01):
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
        net, fixed_input = args
        beta = beta_func(t)
        big_y = coarseners["output_rev"](y)
        net_input = jnp.concatenate([big_y, fixed_input], axis=0)
        pred = net(net_input, t)
        small_pred = coarseners["output"](pred)
        return -0.5 * beta * (y + small_pred)

    def draw_single_sample(fixed_input, rng, net, sample_shape):
        snapshot = jax.random.normal(rng, sample_shape, dtype=jnp.float32)
        terms = diffrax.ODETerm(drift)
        solver = diffrax_utils.Tsit5Float32()
        saveat = diffrax.SaveAt(t1=True)
        sol = diffrax.diffeqsolve(terms, solver, t1, t0, -dt, snapshot, saveat=saveat, adjoint=diffrax_utils.NoAdjointFloat32(), max_steps=max_steps, args=(net, fixed_input))
        return sol.ys[0]

    def draw_raw_samples(fixed_input, net, rng):
        n_batch = fixed_input.shape[0]
        rng_ctr, *rngs = jax.random.split(rng, n_batch + 1)
        samples = jax.vmap(functools.partial(draw_single_sample, net=net, sample_shape=(2, output_size, output_size)))(fixed_input, jnp.stack(rngs))
        return samples, rng_ctr

    return draw_raw_samples


def make_sampler(scalers, coarseners, output_size, input_channels, dt=0.01):

    residual_size = determine_residual_size(input_channels, output_size)

    def draw_samples(batch, net, rng):
        raw_sampler = make_raw_sampler(coarseners=coarseners, output_size=output_size, dt=dt)
        # Batch needs to have all required fields present (at least q and the relevant forcing dimensions)
        fixed_input = build_fixed_input_from_batch(
            batch=batch,
            input_channels=input_channels,
            scalers=scalers,
            coarseners=coarseners
        )
        raw_samples, rng_ctr = raw_sampler(fixed_input, net, rng)
        if residual_size is not None:
            # Handle adding the residual
            existing_target = jax.vmap(coarseners["residual"])(
                jax.vmap(scalers.q_total_forcing_scalers[residual_size].scale_to_standard)(
                    batch.q_total_forcings[residual_size]
                )
            )
            raw_samples = (jnp.sqrt(2).astype(jnp.float32) * raw_samples) + existing_target
        samples = jax.vmap(scalers.q_total_forcing_scalers[output_size].scale_from_standard)(raw_samples)
        return samples, rng_ctr

    return draw_samples


def make_validation_stats_function(num_mean_samples, scalers, coarseners, dt, input_channels, output_size):

    def multi_sampler(net, batch, rng):
        rng_ctr, *batch_rngs = jax.random.split(rng, num_mean_samples + 1)
        batch_rngs = jnp.stack(batch_rngs)
        sample_fn = jax.vmap(
            functools.partial(
                make_sampler(
                    scalers=scalers,
                    coarseners=coarseners,
                    output_size=output_size,
                    input_channels=input_channels,
                    dt=dt,
                ),
                net=net,
                batch=batch,
            )
        )
        samples, _unused_rng_ctr = sample_fn(rng=batch_rngs)
        return samples, rng_ctr

    def compute_stats(batch, net, rng):
        # Dimensions: [samples, batch, lev=2, ny, nx]
        targets = batch.q_total_forcings[output_size]
        samples, rng_ctr = multi_sampler(net, batch, rng)
        single_samples = samples[0]
        means = jnp.mean(samples, axis=0)
        # Compute stats
        # (expand dim to size [samples, time=1, lev=2, ny, nx]
        stats = qg_spec_diag.subgrid_scores(
            true=jnp.expand_dims(targets, 1),
            mean=jnp.expand_dims(means, 1),
            gen=jnp.expand_dims(single_samples, 1),
        )
        return stats, rng_ctr

    return compute_stats


def do_validation(train_state, val_rng, np_rng, loader, sample_stat_fn, num_samples, logger=None):
    if logger is None:
        logger = logging.getLogger("validation")
    # Sample indices
    traj = np_rng.integers(low=0, high=loader.num_trajs, size=num_samples)
    step = np_rng.integers(low=0, high=loader.num_steps, size=num_samples)
    # Load and stack q components
    logger.info("Loading %d samples of validation data", num_samples)
    batch = jax.tree_util.tree_map(
        lambda *args: jnp.concatenate(args, axis=0),
        *(loader.get_trajectory(traj=t, start=s, end=s+1) for t, s in zip(traj, step, strict=True))
    )
    logger.info("Starting validation")
    val_start = time.perf_counter()
    stats, rng_ctr = sample_stat_fn(batch, train_state.net, val_rng)
    stats = jax.device_get(stats)
    val_end = time.perf_counter()
    logger.info("Finished validation in %f sec", val_end - val_start)
    # Report statistics in JSON-serializable format
    stats_report = {
        "l2_mean": stats.l2_mean.item(),
        "l2_total": stats.l2_total.item(),
        "l2_residual": stats.l2_residual.item(),
        "var_ratio": stats.var_ratio.tolist(),
        "duration_sec": val_end - val_start,
    }
    logger.info("l2_mean: %f", stats.l2_mean.item())
    logger.info("l2_total: %f", stats.l2_total.item())
    logger.info("l2_residual: %f", stats.l2_residual.item())
    logger.info("var_ratio: [%f, %f]", stats.var_ratio[0].item(), stats.var_ratio[1].item())
    return stats_report, rng_ctr


def init_network(architecture, lr, rng, output_size, input_channels, processing_size):

    def leaf_map(leaf):
        if isinstance(leaf, jnp.ndarray):
            if leaf.dtype == jnp.dtype(jnp.float64):
                return leaf.astype(jnp.float32)
            if leaf.dtype == jnp.dtype(jnp.complex128):
                return leaf.astype(jnp.complex64)
        return leaf

    num_inputs = len(input_channels)
    args = {
        "img_size": processing_size,
        "n_layers_in": 2 + (num_inputs * 2),
        "n_layers_out": 2,
    }
    net_cls = ARCHITECTURES[architecture]
    net = net_cls(
        **args,
        key=rng,
    )
    optim = optax.adabelief(learning_rate=lr)
    net = jax.tree_util.tree_map(leaf_map, net)
    optim = jax.tree_util.tree_map(leaf_map, optim)
    state = jax_utils.EquinoxTrainState(
        net=net,
        optim=optim,
    )
    network_info = {
        "arch": architecture,
        "args": args,
        "input_channels": input_channels,
        "output_size": output_size,
        "processing_size": processing_size,
    }
    return state, network_info


@jax_utils.register_pytree_dataclass
@dataclasses.dataclass
class Scalers:
    q_scalers: dict[int, jax_utils.Scaler]
    q_total_forcing_scalers: dict[int, jax_utils.Scaler]


def make_scalers(source_data, target_size, input_channels):
    sizes = determine_channel_sizes(input_channels=input_channels)
    q_scalers = {}
    q_total_forcing_scalers = {}
    with h5py.File(source_data, "r") as data_file:
        for q_size_str in data_file["stats"]["q"].keys():
            q_scalers[int(q_size_str)] = Scaler(
                mean=data_file["stats"]["q"][q_size_str]["mean"][:],
                var=data_file["stats"]["q"][q_size_str]["var"][:],
            )
        for forcing_size_str in data_file["stats"]["q_total_forcing"].keys():
            q_total_forcing_scalers[int(forcing_size_str)] = Scaler(
                mean=data_file["stats"]["q_total_forcing"][forcing_size_str]["mean"][:],
                var=data_file["stats"]["q_total_forcing"][forcing_size_str]["var"][:],
            )
    return Scalers(
        q_scalers=q_scalers,
        q_total_forcing_scalers=q_total_forcing_scalers,
    )


def make_coarseners(source_data, target_size, input_channels, processing_size):
    sizes = determine_channel_sizes(input_channels=input_channels)
    models = {}
    coarseners = {}

    def _make_model(size, data_file):
        if size not in models:
            models[size] = QGModel.from_param_json(data_file["params"][f"small_model_{size}"].asstr()[()])
        return models[size]

    with h5py.File(source_data, "r") as data_file:
        q_size = json.loads(data_file["params"]["small_model"].asstr()[()])["nx"]
        coarse_cls = coarsen.COARSEN_OPERATORS[data_file["params"]["coarsen_op"].asstr()[()]]
        for size in set(itertools.chain(sizes, [q_size])):
            # All main input sizes must be scaled to processing_size
            big_model_size = max(processing_size, size)
            small_model_size = min(processing_size, size)
            big_model = _make_model(big_model_size, data_file)
            if size == processing_size:
                # Maintain size with a no-op
                coarsen_fn = coarsen.NoOpCoarsener(big_model=big_model, small_nx=small_model_size).coarsen
            elif size < processing_size:
                # Upscale with a basic coarsener
                coarsen_fn = coarsen.BasicSpectralCoarsener(big_model=big_model, small_nx=small_model_size).uncoarsen
            else:
                # Here size > processing_size. This can only be a q component
                # Use a real coarsener for this as a special case
                coarsen_fn = coarse_cls(big_model=big_model, small_nx=small_model_size).coarsen
            coarseners[size] = coarsen_fn
        # Next, set up coarsen functions for the output which must go from processing_size down to target_size
        big_model = _make_model(processing_size, data_file)
        if target_size == processing_size:
            coarsener = coarsen.NoOpCoarsener(big_model=big_model, small_nx=target_size)
        else:
            coarsener = coarsen.BasicSpectralCoarsener(big_model=big_model, small_nx=target_size)
        coarseners["output"] = coarsener.coarsen
        coarseners["output_rev"] = coarsener.uncoarsen
        # Finally, a coarsener to bring the residual up to target_size
        residual_size = determine_residual_size(input_channels, target_size)
        if residual_size is not None:
            big_model = _make_model(target_size, data_file)
            if residual_size == target_size:
                coarseners["residual"] = coarsen.NoOpCoarsener(big_model=big_model, small_nx=residual_size).uncoarsen
            else:
                coarseners["residual"] = coarsen.BasicSpectralCoarsener(big_model=big_model, small_nx=residual_size).uncoarsen
    return coarseners


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
    if not utils.check_environment_variables(base_logger=logger):
        sys.exit(1)
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
    # Determine what inputs we need
    input_channels = sorted(set(args.input_channels))
    output_size = args.output_size
    channel_sizes = determine_channel_sizes(input_channels=input_channels)
    processing_size = determine_processing_size(
        input_channels=input_channels,
        target_size=output_size,
        user_processing_size=args.processing_size,
    )
    logger.info("Internal processing size: %d", processing_size)
    required_fields = determine_required_fields(
        input_channels=input_channels,
        output_size=output_size,
    )
    logger.info("Required fields: %s", required_fields)
    # Construct neural net
    rng, rng_ctr = jax.random.split(rng_ctr, 2)
    logger.info("Training network: %s", args.architecture)
    state, network_info = init_network(
        architecture=args.architecture,
        lr=args.lr,
        rng=rng,
        output_size=output_size,
        input_channels=input_channels,
        processing_size=processing_size,
    )
    # Store network info
    with open(weights_dir / "network_info.json", "w", encoding="utf8") as net_info_file:
        json.dump(network_info, net_info_file)

    # Create data normalizer and its inverse
    scalers = make_scalers(
        source_data=train_path,
        target_size=output_size,
        input_channels=input_channels,
    )
    coarseners = make_coarseners(
        source_data=train_path,
        target_size=output_size,
        input_channels=input_channels,
        processing_size=processing_size,
    )

    # Open data files
    with contextlib.ExitStack() as train_context:
        # Open data files
        train_loader = train_context.enter_context(
            ThreadedPreShuffledSnapshotLoader(
                file_path=train_path,
                batch_size=args.batch_size,
                buffer_size=10,
                seed=np_rng.integers(2**32).item(),
                base_logger=logger.getChild("train_loader"),
                fields=required_fields,
            )
        )
        val_loader = train_context.enter_context(
            SimpleQGLoader(
                file_path=val_path,
                fields=required_fields,
            )
        )

        # Training functions
        train_batch_fn = eqx.filter_jit(
            make_batch_computer(
                scalers=scalers,
                coarseners=coarseners,
                input_channels=input_channels,
                output_size=output_size,
            )
        )
        val_sample_fn = eqx.filter_jit(
            make_validation_stats_function(
                num_mean_samples=args.val_mean_samples,
                scalers=scalers,
                coarseners=coarseners,
                dt=args.dt,
                input_channels=input_channels,
                output_size=output_size,
            )
        )

        # Running statistics
        min_mean_loss = None

        # Training loop
        epoch_reports = []
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
            saved_names = []
            if min_mean_loss is None or (math.isfinite(mean_loss) and mean_loss <= min_mean_loss):
                min_mean_loss = mean_loss
                save_network("best_loss", output_dir=weights_dir, state=state, base_logger=logger)
                saved_names.append("best_loss")
            if epoch % args.save_interval == 0:
                save_network("interval", output_dir=weights_dir, state=state, base_logger=logger)
                saved_names.append("interval")

            # Validation step
            val_stat_report = None
            if epoch % args.val_interval == 0:
                logger.info("Starting validation for epoch %d", epoch)
                val_stat_report, rng_ctr = do_validation(
                    train_state=state,
                    val_rng=rng_ctr,
                    np_rng=np_rng,
                    loader=val_loader,
                    sample_stat_fn=val_sample_fn,
                    logger=logger.getChild(f"{epoch:05d}_val"),
                    num_samples=args.num_val_samples,
                )
                logger.info("Finished validation for epoch %d", epoch)

            epoch_reports.append(
                {
                    "epoch": epoch,
                    "train_stats": epoch_stats,
                    "val_stats": val_stat_report,
                    "saved_names": saved_names,
                }
            )
            with utils.rename_save_file(out_dir / "train_report.json", "w", encoding="utf8") as train_report_file:
                json.dump(epoch_reports, train_report_file)

            logger.info("Finished epoch %d", epoch)

    # End of training loop
    logger.info("Finished training")


if __name__ == "__main__":
    main()
