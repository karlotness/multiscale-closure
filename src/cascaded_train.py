import argparse
import dataclasses
import itertools
import re
import pathlib
import functools
import time
import logging
import os
import sys
import platform
import contextlib
import math
import operator
import random
import json
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from methods import ARCHITECTURES
from train import determine_processing_size, load_model_params, make_basic_coarsener, determine_output_size, make_chunk_from_batch, determine_required_fields, save_network, make_non_residual_chunk_from_batch, remove_residual_from_output_chunk, determine_channel_layers
from systems.qg import diagnostics as qg_spec_diag
from systems.qg.loader import ThreadedPreShuffledSnapshotLoader, SimpleQGLoader
import jax_utils
import utils

parser = argparse.ArgumentParser(description="Cascaded network training")
parser.add_argument("out_dir", type=str, help="Directory to store output (created if non-existing)")
parser.add_argument("train_set", type=str, help="Directory with training examples")
parser.add_argument("val_set", type=str, help="Directory with validation examples")
parser.add_argument("processing_levels", type=int, nargs="+", help="What levels we use for processing (need at least two)")
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
parser.add_argument("--architecture", type=str, default="gz-fcnn-v1", choices=sorted(ARCHITECTURES.keys()), help="Network architecture to train")
parser.add_argument("--optimizer", type=str, default="adabelief", choices=["adabelief", "adam", "adamw"], help="Which optimizer to use")
parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "warmup1-cosine"], help="What learning rate schedule")
parser.add_argument("--normalization", type=str, default="none", choices=["none", "layer"], help="What type of normalization to apply in the network")


def name_remove_residual(channel):
    if m := re.match(r"^residual:(?P<chan1>[^-]+)-[^-]+$", channel):
        return m.group("chan1")
    return channel


def split_chunk_into_channels(channels, chunk):
    standard_channels = sorted(set(channels))
    channel_sizes = [determine_channel_layers(chan) for chan in standard_channels]
    if sum(channel_sizes) != chunk.shape[-3]:
        raise ValueError(
            f"input chunk has too few channels to split. found {chunk.shape[-3]}, but need {sum(channel_sizes)}"
        )
    ret = {
        chan: arr for chan, arr
        in zip(standard_channels, jnp.split(chunk, np.cumsum(channel_sizes)[:-1], axis=-3), strict=True)
    }
    assert all(ret[chan].shape[-3] == spec_size
               for chan, spec_size in zip(standard_channels, channel_sizes, strict=True))
    return ret


@jax_utils.register_pytree_dataclass
@dataclasses.dataclass
class NetData:
    input_channels: set[str]
    output_channels: set[str]
    processing_size: int


def init_networks(architecture, lr, rng, train_path, optim_type, num_epochs, batches_per_epoch, end_lr, schedule_type, coarse_op_name, processing_scales, normalization):

    def leaf_map(leaf):
        if isinstance(leaf, jnp.ndarray):
            if leaf.dtype == jnp.dtype(jnp.float64):
                return leaf.astype(jnp.float32)
            if leaf.dtype == jnp.dtype(jnp.complex128):
                return leaf.astype(jnp.complex64)
        return leaf

    processing_scales = set(processing_scales)
    args = []
    nets = []
    net_data = []
    rng, *rngs = jax.random.split(rng, len(processing_scales))

    # Build downscale network
    in_channels = [f"q_{max(processing_scales)}"]
    out_channels = [f"q_scaled_forcing_{max(processing_scales)}to{min(processing_scales)}"]
    processing_size = determine_processing_size(input_channels=in_channels, output_channels=out_channels)
    arg = {
        "img_size": processing_size,
        "n_layers_in": 2 * len(in_channels),
        "n_layers_out": 2 * len(out_channels),
        "normalization": normalization,
    }
    args.append(arg)
    nets.append(
        ARCHITECTURES[architecture](**arg, key=rng)
    )
    net_data.append(
        NetData(
            input_channels=set(in_channels),
            output_channels=set(out_channels),
            processing_size=processing_size,
        )
    )

    # Build upscale networks
    for (small, big), rng in zip(itertools.pairwise(sorted(processing_scales)), rngs, strict=True):
        in_channels = [f"q_{big}", f"q_scaled_forcing_{max(processing_scales)}to{small}"]
        if big == max(processing_scales):
            target_chan = f"q_total_forcing_{max(processing_scales)}"
        else:
            target_chan = f"q_scaled_forcing_{max(processing_scales)}to{big}"
        out_channels = [f"residual:{target_chan}-q_scaled_forcing_{max(processing_scales)}to{small}"]
        processing_size = determine_processing_size(input_channels=in_channels, output_channels=out_channels)
        arg = {
            "img_size": processing_size,
            "n_layers_in": 2 * len(in_channels),
            "n_layers_out": 2 * len(out_channels),
            "normalization": normalization,
        }
        args.append(arg)
        nets.append(
            ARCHITECTURES[architecture](**arg, key=rng)
        )
        net_data.append(
            NetData(
                input_channels=in_channels,
                output_channels=out_channels,
                processing_size=processing_size,
            )
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

    nets = jax.tree_util.tree_map(leaf_map, tuple(nets))
    optim = jax.tree_util.tree_map(leaf_map, optim)
    state = jax_utils.EquinoxTrainState(
        net=nets,
        optim=optim,
    )
    network_info = {
        "train_path": str(pathlib.Path(train_path).resolve()),
        "coarse_op_name": coarse_op_name,
        "processing_scales": sorted(processing_scales),
        "networks": [
            {
                "arch": architecture,
                "args": arg,
                "input_channels": list(data.input_channels),
                "output_channels": list(data.output_channels),
                "processing_size": data.processing_size,
            }
            for arg, data in zip(args, net_data)
        ],
    }
    return state, network_info, tuple(net_data)


def make_batch_computer(net_data, model_params):

    def sample_loss(input_elem, target_elem, net, output_size, processing_size):
        y = net(input_elem)
        y = make_basic_coarsener(processing_size, output_size, model_params)(y)
        mse = jnp.mean((y - target_elem)**2)
        return mse, y

    def batch_loss(net, input_chunk, target_chunk, output_size, processing_size):
        losses, predictions = jax.vmap(
            functools.partial(
                sample_loss,
                net=net,
                output_size=output_size,
                processing_size=processing_size,
            )
        )(input_chunk, target_chunk)
        return jnp.mean(losses), predictions

    def do_batch(batch, state):
        alt_sources = {}
        grads = []
        losses = []
        for net, data in zip(state.net, net_data, strict=True):
            output_size = determine_output_size(data.output_channels)
            input_chunk = make_chunk_from_batch(
                channels=data.input_channels,
                batch=batch,
                model_params=model_params,
                processing_size=data.processing_size,
                alt_source=alt_sources,
            )
            target_chunk = make_chunk_from_batch(
                channels=data.output_channels,
                batch=batch,
                model_params=model_params,
                processing_size=output_size,
            )
            # Compute losses
            (loss, predictions), grad = eqx.filter_value_and_grad(batch_loss, has_aux=True)(net, input_chunk, target_chunk, output_size, data.processing_size)
            losses.append(loss)
            grads.append(grad)
            # Process predictions and add to alt_sources
            predictions = split_chunk_into_channels(
                channels=data.output_channels,
                chunk=remove_residual_from_output_chunk(
                    output_channels=data.output_channels,
                    output_chunk=predictions,
                    batch=batch,
                    model_params=model_params,
                    processing_size=output_size,
                    alt_source=alt_sources,
                )
            )
            alt_sources.update({name_remove_residual(k): v for k, v in predictions.items()})
        # Apply updates
        out_state = state.apply_updates(tuple(grads))
        return out_state, jnp.mean(jnp.stack(losses))

    return do_batch


def do_epoch(train_state, batch_iter, batch_fn, logger=None):
    if logger is None:
        logger = logging.getLogger("train_epoch")
    epoch_start = time.perf_counter()
    losses = []
    for batch in batch_iter:
        train_state, batch_loss = batch_fn(batch, train_state)
        losses.append(batch_loss)
    epoch_end = time.perf_counter()
    mean_loss = jax.device_get(jnp.mean(jnp.stack(losses)))
    final_loss = jax.device_get(losses[-1])
    logger.info("Finished epoch in %f sec", epoch_end - epoch_start)
    logger.info("Epoch mean loss %f", mean_loss)
    logger.info("Epoch final loss %f", final_loss)
    return train_state, {"mean_loss": mean_loss.item(), "final_loss": final_loss.item(), "duration_sec": epoch_end - epoch_start}


def make_validation_stats_function(net_data, model_params, processing_scales):
    output_target = f"q_total_forcing_{max(processing_scales)}"

    def compute_stats(batch, nets):
        alt_sources = {}
        for net, data in zip(nets, net_data, strict=True):
            output_size = determine_output_size(data.output_channels)
            input_chunk = make_chunk_from_batch(
                channels=data.input_channels,
                batch=batch,
                model_params=model_params,
                processing_size=data.processing_size,
                alt_source=alt_sources,
            )
            predictions = jax.vmap(net)(input_chunk)
            predictions = jax.vmap(make_basic_coarsener(data.processing_size, output_size, model_params))(predictions)
            # Process predictions and add to alt_sources
            predictions = split_chunk_into_channels(
                channels=data.output_channels,
                chunk=remove_residual_from_output_chunk(
                    output_channels=data.output_channels,
                    output_chunk=predictions,
                    batch=batch,
                    model_params=model_params,
                    processing_size=output_size,
                    alt_source=alt_sources,
                )
            )
            alt_sources.update({name_remove_residual(k): v for k, v in predictions.items()})
        # Locate final prediction and compute stats
        samples = alt_sources[output_target]
        targets = make_non_residual_chunk_from_batch(
            channels=[output_target],
            batch=batch,
            model_params=model_params,
            processing_size=max(processing_scales),
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
        return stat_report

    return compute_stats


def do_validation(train_state, loader, sample_stat_fn, traj, step, logger=None):
    if logger is None:
        logger = logging.getLogger("validation")
    # Sample indices
    num_samples = traj.shape[0]
    if step.shape[0] != num_samples:
        logger.error("mismatched validation samples")
        raise ValueError("mismatched number of validation samples")
    if traj.ndim != 1 or step.ndim != 1:
        logger.error("validation sample arrays must be one-dimensional")
        raise ValueError("validation sample arrays must be one-dimensional")
    # Load and stack q components
    logger.info("Loading %d samples of validation data", num_samples)
    batch = jax.tree_util.tree_map(
        lambda *args: jnp.concatenate(args, axis=0),
        *(loader.get_trajectory(traj=operator.index(t), start=operator.index(s), end=operator.index(s)+1) for t, s in zip(traj, step, strict=True))
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

    processing_scales = set(args.processing_levels)
    if len(processing_scales) < 2:
        logger.error("Must provide at least two processing scale levels")
        sys.exit(2)

    # Create data normalizer and its inverse
    model_params = load_model_params(train_path)
    coarse_op_name = model_params.scale_operator
    # Construct neural net
    rng, rng_ctr = jax.random.split(rng_ctr, 2)
    logger.info("Training network: %s", args.architecture)
    state, network_info, net_data = init_networks(
        architecture=args.architecture,
        lr=args.lr,
        rng=rng,
        train_path=train_path,
        optim_type=args.optimizer,
        num_epochs=args.num_epochs,
        batches_per_epoch=args.batches_per_epoch,
        end_lr=args.end_lr,
        schedule_type=args.lr_schedule,
        coarse_op_name=coarse_op_name,
        processing_scales=processing_scales,
        normalization=args.normalization,
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

    # Determine what inputs we need
    required_fields = sorted(
        determine_required_fields(
            itertools.chain.from_iterable(
                itertools.chain(data.input_channels, data.output_channels)
                for data in net_data
            )
        )
    )
    logger.info("Required fields: %s", required_fields)
    logger.info("Output size: %d", max(processing_scales))

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
                net_data=net_data,
                model_params=model_params,
            )
        )
        val_stats_fn = eqx.filter_jit(
            make_validation_stats_function(
                net_data=net_data,
                model_params=model_params,
                processing_scales=processing_scales,
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
                state, epoch_stats = do_epoch(
                    train_state=state,
                    batch_iter=itertools.islice(train_batch_iter, args.batches_per_epoch),
                    batch_fn=train_batch_fn,
                    logger=logger.getChild(f"{epoch:05d}_train"),
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
