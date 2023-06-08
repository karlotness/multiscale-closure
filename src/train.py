import argparse
import dataclasses
import pathlib
import math
import os
import sys
import re
import platform
import random
import contextlib
import itertools
import jax
import jax.numpy as jnp
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
from systems.qg import coarsen, diagnostics as qg_spec_diag, utils as qg_utils
from pyqg_jax.qg_model import QGModel
from methods import ARCHITECTURES
import jax_utils
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
parser.add_argument("--end_lr", type=float, default=None, help="Learning rate at end of schedule")
parser.add_argument("--num_val_samples", type=int, default=10, help="Number of samples to draw in each validation period")
parser.add_argument("--val_interval", type=int, default=1, help="Number of epochs between validation periods")
parser.add_argument("--output_channels", type=str, nargs="+", default=["q_total_forcing_64"], help="What output channels to produce")
parser.add_argument("--input_channels", type=str, nargs="+", default=["q_64"], help="Channels to show the network as input")
parser.add_argument("--noise_specs", type=str, nargs="+", default=[], help="Channels with noise variances (format 'channel=var0,var1')")
parser.add_argument("--processing_size", type=int, default=None, help="Size to user for internal network evaluation (default: select automatically)")
parser.add_argument("--architecture", type=str, default="gz-fcnn-v1", choices=sorted(ARCHITECTURES.keys()), help="Network architecture to train")
parser.add_argument("--optimizer", type=str, default="adabelief", choices=["adabelief", "adam", "adamw"], help="Which optimizer to use")
parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "warmup1-cosine"], help="What learning rate schedule")


def save_network(output_name, output_dir, state, base_logger=None):
    if base_logger is None:
        logger = logging.getLogger("save")
    else:
        logger = base_logger.getChild("save")
    output_dir = pathlib.Path(output_dir)
    with utils.rename_save_file(output_dir / f"{output_name}.eqx", "wb") as eqx_out_file:
        eqx.tree_serialise_leaves(eqx_out_file, state.net)
    logger.info("Saved network parameters to %s in %s", output_name, output_dir)


def determine_required_fields(channels):
    """Figure out what channels need to be loaded given a list of specifications"""
    loader_chans = set()
    for chan in channels:
        if re.match(r"^q_total_forcing_\d+$", chan):
            loader_chans.add(chan)
        elif re.match(r"^q_\d+$", chan):
            loader_chans.add("q")
        elif m := re.match(r"^q_scaled_forcing_(?P<orig_size>\d+)to\d+$", chan):
            orig_size = m.group("orig_size")
            loader_chans.update(determine_required_fields([f"q_total_forcing_{orig_size}"]))
        elif m := re.match(r"^q_scaled_(?P<orig_size>\d+)to\d+$", chan):
            orig_size = m.group("orig_size")
            loader_chans.update(determine_required_fields([f"q_{orig_size}"]))
        elif m := re.match(r"^residual:(?P<chan1>[^-]+)-(?P<chan2>[^-]+)$", chan):
            loader_chans.update(determine_required_fields([m.group("chan1"), m.group("chan2")]))
        else:
            raise ValueError(f"unsupported channel {chan}")
    return loader_chans


def determine_channel_size(chan):
    """Determine the final scaled size of the channel from its name"""
    if m := re.match(r"^q_total_forcing_(?P<size>\d+)$", chan):
        return int(m.group("size"))
    elif m := re.match(r"^q_(?P<size>\d+)$", chan):
        return int(m.group("size"))
    elif m := re.match(r"^q_scaled_forcing_\d+to(?P<size>\d+)$", chan):
        return int(m.group("size"))
    elif m := re.match(r"^q_scaled_\d+to(?P<size>\d+)$", chan):
        return int(m.group("size"))
    elif m := re.match(r"^residual:(?P<chan1>[^-]+)-(?P<chan2>[^-]+)$", chan):
        return max(
            determine_channel_size(m.group("chan1")),
            determine_channel_size(m.group("chan2")),
        )
    else:
        raise ValueError(f"unsupported channel {chan}")


def determine_channel_layers(chan):
    """Determine the number of layers based on the channel name"""
    if m := re.match(r"^q_total_forcing_\d+$", chan):
        return 2
    elif m := re.match(r"^q_\d+$", chan):
        return 2
    elif m := re.match(r"^q_scaled_forcing_(?P<orig_size>\d+)to\d+$", chan):
        orig_size = int(m.group("orig_size"))
        return determine_channel_layers(f"q_total_forcing_{orig_size}")
    elif m := re.match(r"^q_scaled_(?P<orig_size>\d+)to\d+$", chan):
        orig_size = int(m.group("orig_size"))
        return determine_channel_layers(f"q_{orig_size}")
    elif m := re.match(r"^residual:(?P<chan1>[^-]+)-(?P<chan2>[^-]+)$", chan):
        chan1_layers = determine_channel_layers(m.group("chan1"))
        chan2_layers = determine_channel_layers(m.group("chan2"))
        if chan1_layers != chan2_layers:
            raise ValueError(f"incompatible channel layer counts for {chan} ({chan1_layers} vs {chan2_layers})")
        return chan1_layers
    else:
        raise ValueError(f"unsupported channel {chan}")


def determine_processing_size(input_channels, output_channels, user_processing_size=None):
    """Determine what size the network should be run at"""
    auto_processing_size = max(determine_channel_size(chan) for chan in itertools.chain(input_channels, output_channels))
    if user_processing_size is not None:
        user_processing_size = operator.index(user_processing_size)
        if user_processing_size < auto_processing_size:
            raise ValueError(f"invalid override processing size: must be at least {auto_processing_size}")
        return user_processing_size
    return auto_processing_size


def determine_output_size(output_channels):
    sizes = {determine_channel_size(chan) for chan in output_channels}
    if len(sizes) != 1:
        raise ValueError("output channel sizes must be unique")
    return next(iter(sizes))


@jax_utils.register_pytree_dataclass
@dataclasses.dataclass
class Scalers:
    q_scalers: dict[int, jax_utils.Scaler]
    q_total_forcing_scalers: dict[int, jax_utils.Scaler]


def make_scalers(source_data):
    q_scalers = {}
    q_total_forcing_scalers = {}
    with h5py.File(source_data, "r") as data_file:
        for q_size_str in data_file["stats"]["q"].keys():
            q_scalers[int(q_size_str)] = jax_utils.Scaler(
                mean=data_file["stats"]["q"][q_size_str]["mean"][:],
                var=data_file["stats"]["q"][q_size_str]["var"][:],
            )
        for forcing_size_str in data_file["stats"]["q_total_forcing"].keys():
            q_total_forcing_scalers[int(forcing_size_str)] = jax_utils.Scaler(
                mean=data_file["stats"]["q_total_forcing"][forcing_size_str]["mean"][:],
                var=data_file["stats"]["q_total_forcing"][forcing_size_str]["var"][:],
            )
    return Scalers(
        q_scalers=q_scalers,
        q_total_forcing_scalers=q_total_forcing_scalers,
    )


@jax_utils.register_pytree_dataclass
@dataclasses.dataclass
class ModelParams:
    scalers: Scalers
    qg_models: dict[int, QGModel]
    scale_operator: str


def load_model_params(train_path):
    train_path = pathlib.Path(train_path)
    if not train_path.exists() and train_path.is_relative_to("/scratch"):
        # Fix train data paths when loading Greene paths on Flatiron systems
        train_path = pathlib.Path(os.environ["SCRATCH"]) / train_path.relative_to(train_path.parents[-3])
    # Continue with loading params
    qg_models = {}
    with h5py.File(train_path, "r") as data_file:
        coarse_op_name = data_file["params"]["coarsen_op"].asstr()[()]
        for k in data_file["params"]:
            if m := re.match(r"^small_model_(?P<size>\d+)$", k):
                qg_models[int(m.group("size"))] = qg_utils.qg_model_from_param_json(
                    data_file["params"][k].asstr()[()]
                )
    return ModelParams(
        scalers=make_scalers(train_path),
        qg_models=qg_models,
        scale_operator=coarse_op_name,
    )


def make_basic_coarsener(from_size, to_size, model_params, coarse_cls=coarsen.BasicSpectralCoarsener):
    model_size = max(from_size, to_size)
    small_size = min(from_size, to_size)
    big_model = model_params.qg_models[model_size]
    if from_size == to_size:
        return coarsen.NoOpCoarsener(big_model=big_model, small_nx=small_size).coarsen
    direct_op = coarse_cls(big_model=big_model, small_nx=small_size)
    if from_size < to_size:
        return direct_op.uncoarsen
    else:
        return direct_op.coarsen


def make_channel_from_batch(channel, batch, model_params, alt_source=None):
    if alt_source is not None and channel in alt_source:
        return alt_source[channel]
    end_size = determine_channel_size(channel)
    if re.match(r"^q_total_forcing_\d+$", channel):
        return jax.vmap(model_params.scalers.q_total_forcing_scalers[end_size].scale_to_standard)(
            batch.q_total_forcings[end_size]
        )
    elif re.match(r"^q_\d+$", channel):
        # Need to scale q down to proper size
        q_size = batch.q.shape[-1]
        if q_size != end_size:
            coarse_op = coarsen.COARSEN_OPERATORS[model_params.scale_operator](
                big_model=model_params.qg_models[q_size],
                small_nx=end_size,
            )
        else:
            coarse_op = coarsen.NoOpCoarsener(
                big_model=model_params.qg_models[q_size],
                small_nx=end_size,
            )
        return jax.vmap(model_params.scalers.q_scalers[end_size].scale_to_standard)(
            jax.vmap(coarse_op.coarsen)(batch.q)
        )
    elif m := re.match(r"^q_scaled_forcing_(?P<orig_size>\d+)to\d+$", channel):
        orig_size = int(m.group("orig_size"))
        return jax.vmap(make_basic_coarsener(orig_size, end_size, model_params))(
            make_channel_from_batch(f"q_total_forcing_{orig_size}", batch, model_params, alt_source=alt_source)
        )
    elif m := re.match(r"^q_scaled_(?P<orig_size>\d+)to\d+$", channel):
        orig_size = int(m.group("orig_size"))
        return jax.vmap(make_basic_coarsener(orig_size, end_size, model_params))(
            make_channel_from_batch(f"q_{orig_size}", batch, model_params, alt_source=alt_source)
        )
    elif m := re.match(r"^residual:(?P<chan1>[^-]+)-(?P<chan2>[^-]+)$", channel):
        chan1 = jax.vmap(
            make_basic_coarsener(
                determine_channel_size(m.group("chan1")),
                end_size,
                model_params,
            )
        )(make_channel_from_batch(m.group("chan1"), batch, model_params, alt_source=alt_source))
        chan2 = jax.vmap(
            make_basic_coarsener(
                determine_channel_size(m.group("chan2")),
                end_size,
                model_params,
            )
        )(make_channel_from_batch(m.group("chan2"), batch, model_params, alt_source=alt_source))
        return chan1 - chan2
    else:
        raise ValueError(f"unsupported channel {channel}")


def make_noisy_channel_from_batch(channel, batch, model_params, alt_source=None, noise_var=0, key=None):
    chan = make_channel_from_batch(
        channel=channel,
        batch=batch,
        model_params=model_params,
        alt_source=alt_source
    )
    if np.any(noise_var != 0):
        noise_var = jnp.sqrt(noise_var).astype(chan.dtype)
        if noise_var.ndim > 0:
            noise_var = jnp.expand_dims(noise_var, (-1, -2))
        noise_mask = noise_var * jax.random.normal(key=key, shape=chan.shape, dtype=chan.dtype)
        return chan + noise_mask
    else:
        return chan


def standardize_noise_specs(channels, noise_spec):
    noise_specs = {}
    if noise_spec is not None:
        unmatched_keys = noise_spec.keys() - set(channels)
        if unmatched_keys:
            raise ValueError(f"unmatched noise specs: {unmatched_keys}")
    for channel in channels:
        if noise_spec is not None and channel in noise_spec:
            noise_specs[channel] = noise_spec[channel]
        else:
            noise_specs[channel] = 0
    count_noise = sum(1 for var in noise_specs.values() if np.any(var != 0))
    return noise_specs, count_noise


def make_chunk_from_batch(channels, batch, model_params, processing_size, alt_source=None, noise_spec=None, key=None):
    standard_channels = sorted(set(channels))
    stacked_channels = []
    noise_spec, count_noise = standardize_noise_specs(channels, noise_spec)
    if count_noise > 0:
        assert key is not None
        keys = list(jax.random.split(key, count_noise))
    else:
        keys = []
    for channel in standard_channels:
        noise_var = noise_spec[channel]
        if np.any(noise_var != 0):
            key = keys.pop()
        else:
            key = None
        data = make_noisy_channel_from_batch(channel, batch, model_params, alt_source=alt_source, noise_var=noise_var, key=key)
        stacked_channels.append(
            jax.vmap(make_basic_coarsener(data.shape[-1], processing_size, model_params))(data)
        )
    return jnp.concatenate(stacked_channels, axis=-3)


def make_non_residual_chunk_from_batch(channels, batch, model_params, processing_size, alt_source=None):
    standard_channels = sorted(set(channels))
    stacked_channels = []
    for channel in standard_channels:
        if m := re.match(r"^residual:(?P<chan1>[^-]+)-(?P<chan2>[^-]+)$", channel):
            # Special processing for residual channel
            # Load base channel
            data = make_channel_from_batch(m.group("chan1"), batch, model_params, alt_source=alt_source)
            # Scale to residual size (and skip the subtraction)
            data = jax.vmap(make_basic_coarsener(data.shape[-1], determine_channel_size(channel), model_params))(data)
            # Scale to final size
            data = jax.vmap(make_basic_coarsener(data.shape[-1], processing_size, model_params))(data)
            stacked_channels.append(data)
        else:
            # Normal processing
            data = make_channel_from_batch(channel, batch, model_params, alt_source=alt_source)
            stacked_channels.append(
                jax.vmap(make_basic_coarsener(data.shape[-1], processing_size, model_params))(data)
            )
    return jnp.concatenate(stacked_channels, axis=-3)


def remove_residual_from_output_chunk(output_channels, output_chunk, batch, model_params, processing_size, alt_source=None):
    standard_channels = sorted(set(output_channels))
    stacked_channels = []
    for channel in standard_channels:
        if m := re.match(r"^residual:(?P<chan1>[^-]+)-(?P<chan2>[^-]+)$", channel):
            # Special processing for residual channel
            # Load base channel
            data = make_channel_from_batch(m.group("chan2"), batch, model_params, alt_source=alt_source)
            # Scale to residual size (and skip the subtraction)
            data = jax.vmap(make_basic_coarsener(data.shape[-1], determine_channel_size(channel), model_params))(data)
            # Scale to final size
            data = jax.vmap(make_basic_coarsener(data.shape[-1], processing_size, model_params))(data)
            stacked_channels.append(data)
        else:
            # Normal processing (no offset needed)
            channel_layers = determine_channel_layers(channel)
            output_shape = (channel_layers, processing_size, processing_size)
            stacked_channels.append(jnp.zeros(output_shape, dtype=output_chunk.dtype))

    return output_chunk + jnp.concatenate(stacked_channels, axis=-3)


def make_batch_computer(input_channels, output_channels, model_params, processing_size, noise_spec):
    output_size = determine_output_size(output_channels)

    def sample_loss(input_elem, target_elem, net):
        y = net(input_elem)
        y = make_basic_coarsener(processing_size, output_size, model_params)(y)
        mse = jnp.mean((y - target_elem)**2)
        return mse

    def batch_loss(net, input_chunk, target_chunk):
        losses = jax.vmap(
            functools.partial(
                sample_loss,
                net=net,
            )
        )(input_chunk, target_chunk)
        return jnp.mean(losses)

    def do_batch(batch, state, rng_ctr):
        rng, rng_ctr = jax.random.split(rng_ctr, 2)
        input_chunk = make_chunk_from_batch(
            channels=input_channels,
            batch=batch,
            model_params=model_params,
            processing_size=processing_size,
            noise_spec=noise_spec,
            key=rng,
        )
        target_chunk = make_chunk_from_batch(
            channels=output_channels,
            batch=batch,
            model_params=model_params,
            processing_size=output_size,
        )
        # Compute losses
        loss, grads = eqx.filter_value_and_grad(batch_loss)(state.net, input_chunk, target_chunk)
        # Update parameters
        out_state = state.apply_updates(grads)
        return out_state, loss, rng_ctr

    return do_batch


def do_epoch(train_state, batch_iter, batch_fn, rng_ctr, logger=None):
    if logger is None:
        logger = logging.getLogger("train_epoch")
    epoch_start = time.perf_counter()
    losses = []
    for batch in batch_iter:
        train_state, batch_loss, rng_ctr = batch_fn(batch, train_state, rng_ctr)
        losses.append(batch_loss)
    epoch_end = time.perf_counter()
    mean_loss = jax.device_get(jnp.mean(jnp.stack(losses)))
    final_loss = jax.device_get(losses[-1])
    logger.info("Finished epoch in %f sec", epoch_end - epoch_start)
    logger.info("Epoch mean loss %f", mean_loss)
    logger.info("Epoch final loss %f", final_loss)
    return train_state, {"mean_loss": mean_loss.item(), "final_loss": final_loss.item(), "duration_sec": epoch_end - epoch_start}, rng_ctr


def make_validation_stats_function(input_channels, output_channels, model_params, processing_size, include_raw_err=False):
    output_size = determine_output_size(output_channels)

    def make_samples(input_chunk, net):
        ys = jax.vmap(net)(input_chunk)
        return jax.vmap(make_basic_coarsener(processing_size, output_size, model_params))(ys)

    def compute_stats(batch, net):
        input_chunk = make_chunk_from_batch(
            channels=input_channels,
            batch=batch,
            model_params=model_params,
            processing_size=processing_size,
        )
        targets = make_non_residual_chunk_from_batch(
            channels=output_channels,
            batch=batch,
            model_params=model_params,
            processing_size=output_size,
        )
        samples = remove_residual_from_output_chunk(
            output_channels=output_channels,
            output_chunk=make_samples(input_chunk, net),
            batch=batch,
            model_params=model_params,
            processing_size=output_size,
        )
        err = targets - samples
        mse = jnp.mean(err**2)
        stats = qg_spec_diag.subgrid_scores(
            true=jnp.expand_dims(targets, 1),
            mean=jnp.expand_dims(samples, 1),
            gen=jnp.expand_dims(samples, 1),
        )
        stat_report = {
            "standard_mse": mse,
            "l2_mean": stats.l2_mean,
            "l2_total": stats.l2_total,
        }
        if include_raw_err:
            stat_report["raw_err"] = err
        return stat_report

    return compute_stats


def do_validation(train_state, np_rng, loader, sample_stat_fn, num_samples, logger=None):
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
    stats_report = sample_stat_fn(batch, train_state.net)
    val_end = time.perf_counter()
    logger.info("Finished validation in %f sec", val_end - val_start)
    # Report statistics in JSON-serializable format
    stats_report = jax_utils.make_json_serializable(stats_report)
    # Log stats
    for stat_name, stat_value in stats_report.items():
        logger.info("%s: %s", stat_name, stat_value)
    # Add validation time to stats
    stats_report["duration_sec"] = val_end - val_start
    return stats_report


def init_network(architecture, lr, rng, input_channels, output_channels, processing_size, train_path, optim_type, num_epochs, batches_per_epoch, end_lr, schedule_type, coarse_op_name):

    def leaf_map(leaf):
        if isinstance(leaf, jnp.ndarray):
            if leaf.dtype == jnp.dtype(jnp.float64):
                return leaf.astype(jnp.float32)
            if leaf.dtype == jnp.dtype(jnp.complex128):
                return leaf.astype(jnp.complex64)
        return leaf

    n_layers_in = sum(map(determine_channel_layers, input_channels))
    n_layers_out = sum(map(determine_channel_layers, output_channels))

    args = {
        "img_size": processing_size,
        "n_layers_in": n_layers_in,
        "n_layers_out": n_layers_out,
    }
    net_cls = ARCHITECTURES[architecture]
    net = net_cls(
        **args,
        key=rng,
    )

    # Configure learning rate schedule
    steps_per_epoch = batches_per_epoch
    total_steps = steps_per_epoch * num_epochs
    match schedule_type:
        case "constant":
            sched_args = {
                "type": "constant",
                "args": {
                    "value": lr,
                },
            }
            schedule = optax.constant_schedule(**sched_args["args"])
        case "warmup1-cosine":
            sched_args = {
                "type": "warmup1-cosine",
                "args": {
                    "init_value": 0.0,
                    "peak_value": lr,
                    "warmup_steps": steps_per_epoch,
                    "decay_steps": total_steps,
                    "end_value": (0.0 if end_lr is None else end_lr),
                },
            }
            schedule = optax.warmup_cosine_decay_schedule(**sched_args["args"])
        case _:
            raise ValueError(f"unsupported schedule {schedule_type}")

    match optim_type:
        case "adabelief":
            optim = optax.adabelief(learning_rate=schedule)
        case "adam":
            optim = optax.adam(learning_rate=schedule)
        case "adamw":
            optim = optax.adamw(learning_rate=schedule)
        case _:
            raise ValueError(f"unsupported optimizer {optim_type}")

    optim = optax.apply_if_finite(
        optax.chain(
            optax.clip(1.0),
            optim,
        ),
        100,
    )

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
        "output_channels": output_channels,
        "processing_size": processing_size,
        "train_path": str(pathlib.Path(train_path).resolve()),
        "coarse_op_name": coarse_op_name,
    }
    return state, network_info


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
    train_path = (pathlib.Path(args.train_set) / "shuffled.hdf5").resolve()
    val_path = (pathlib.Path(args.val_set) / "data.hdf5").resolve()
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    # Determine what inputs we need
    input_channels = sorted(set(args.input_channels))
    output_channels = sorted(set(args.output_channels))
    processing_size = determine_processing_size(
        input_channels=input_channels,
        output_channels=output_channels,
        user_processing_size=args.processing_size,
    )
    required_fields = sorted(
        determine_required_fields(
            itertools.chain(
                input_channels,
                output_channels,
            )
        )
    )
    logger.info("Required fields: %s", required_fields)
    logger.info("Input channels: %s", input_channels)
    logger.info("Processing size: %d", processing_size)
    logger.info("Output channels: %s", output_channels)
    logger.info("Output size: %d", determine_output_size(output_channels))


    # Create data normalizer and its inverse
    model_params = load_model_params(train_path)
    coarse_op_name = model_params.scale_operator
    # Construct neural net
    rng, rng_ctr = jax.random.split(rng_ctr, 2)
    logger.info("Training network: %s", args.architecture)
    state, network_info = init_network(
        architecture=args.architecture,
        lr=args.lr,
        rng=rng,
        input_channels=input_channels,
        output_channels=output_channels,
        processing_size=processing_size,
        train_path=train_path,
        optim_type=args.optimizer,
        num_epochs=args.num_epochs,
        batches_per_epoch=args.batches_per_epoch,
        end_lr=args.end_lr,
        schedule_type=args.lr_schedule,
        coarse_op_name=coarse_op_name,
    )
    # Store network info
    with utils.rename_save_file(weights_dir / "network_info.json", "w", encoding="utf8") as net_info_file:
        json.dump(network_info, net_info_file)
    # Store run details
    with utils.rename_save_file(out_dir / "cli_info.json", "w", encoding="utf8") as cli_info_file:
        cli_info = {
                "argv": sys.argv,
                "parsed_args": dict(vars(args)),
                "environ": dict(os.environ),
                "node": platform.node(),
        }
        if git_info is not None:
            cli_info["git_info"] = {
                "commit": git_info.hash,
                "clean_worktree": git_info.clean_worktree
            }
        else:
            cli_info["git_info"] = None
        json.dump(cli_info, cli_info_file)

    # Process noise_spec
    noise_spec = {}
    for spec in args.noise_specs:
        spec = spec.strip()
        if not spec:
            continue
        chan_name, var = spec.split("=")
        noise_spec[chan_name.strip()] = np.array([float(v.strip()) for v in var.strip().split(",")])

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
                input_channels=input_channels,
                output_channels=output_channels,
                model_params=model_params,
                processing_size=processing_size,
                noise_spec=noise_spec,
            )
        )
        val_stats_fn = eqx.filter_jit(
            make_validation_stats_function(
                input_channels=input_channels,
                output_channels=output_channels,
                model_params=model_params,
                processing_size=processing_size,
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
                state, epoch_stats, rng_ctr = do_epoch(
                    train_state=state,
                    batch_iter=itertools.islice(train_batch_iter, args.batches_per_epoch),
                    batch_fn=train_batch_fn,
                    logger=logger.getChild(f"{epoch:05d}_train"),
                    rng_ctr=rng_ctr,
                )
            mean_loss = epoch_stats["mean_loss"]

            # Save snapshots
            saved_names = []
            # Save the network after each epoch
            save_network(f"epoch{epoch:04d}", output_dir=weights_dir, state=state, base_logger=logger)
            saved_names.append(f"epoch{epoch:04d}")
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
                val_stat_report = do_validation(
                    train_state=state,
                    np_rng=np_rng,
                    loader=val_loader,
                    sample_stat_fn=val_stats_fn,
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
