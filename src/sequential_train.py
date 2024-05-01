import argparse
import itertools
import pathlib
import functools
import logging
import os
import sys
import platform
import contextlib
import math
import random
import json
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from methods import get_net_constructor
from train import init_network as train_init_network, save_network, determine_processing_size, load_model_params, make_basic_coarsener, determine_output_size, make_chunk_from_batch, determine_required_fields, make_non_residual_chunk_from_batch, remove_residual_from_output_chunk, sniff_system_type, make_val_loader
import utils
from systems.qg import diagnostics as qg_spec_diag
from systems.qg.loader import ThreadedPreShuffledSnapshotLoader, SimpleQGLoader
from systems.ns import loader as ns_loader, utils as ns_utils
from eval import load_network
from cascaded_train import NetData, split_chunk_into_channels, name_remove_residual, do_epoch, do_validation

parser = argparse.ArgumentParser(description="Sequential network training")
parser.add_argument("out_dir", type=str, help="Directory to store output (created if non-existing)")
parser.add_argument("train_set", type=str, help="Directory with training examples")
parser.add_argument("val_set", type=str, help="Directory with validation examples")
parser.add_argument("train_step", type=int, help="Which network we are training in sequence (starts from 0)")
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
parser.add_argument("--val_sample_seed", type=int, default=1234, help="RNG seed to select validation samples")
parser.add_argument("--val_interval", type=int, default=1, help="Number of epochs between validation periods")
parser.add_argument("--architecture", type=str, nargs="+", default=["gz-fcnn-v1"], help="Network architecture to train")
parser.add_argument("--optimizer", type=str, default="adabelief", choices=["adabelief", "adam", "adamw"], help="Which optimizer to use")
parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "warmup1-cosine", "ross22"], help="What learning rate schedule")
parser.add_argument("--normalization", type=str, default="none", choices=["none", "layer"], help="What type of normalization to apply in the network")
parser.add_argument("--net_load_type", type=str, default="best_loss", help="Which saved weights to load for previous networks")
parser.add_argument("--no_residual", action="store_false", dest="output_residuals", help="Set sequential networks to output non-residual values (they learn to combine the fields)")
parser.add_argument("--loader_chunk_size", type=int, default=10850, help="Chunk size to read before batching")


def load_prev_networks(base_dir, train_step, net_load_type, base_logger=None):
    if base_logger is None:
        logger = logging.getLogger("load-prev")
    else:
        logger = base_logger.getChild("load-prev")
    base_dir = pathlib.Path(base_dir)
    loaded_nets = []
    loaded_net_data = []
    loaded_net_info = []
    logger.info("loading previous networks")
    for net_idx in range(train_step):
        logger.info("loading network %d %s", net_idx, net_load_type)
        net_path = base_dir / f"net{net_idx}" / "weights" / f"{net_load_type}.eqx"
        net, net_info = load_network(net_path)
        loaded_nets.append(net)
        loaded_net_info.append(net_info)
        loaded_net_data.append(
            NetData(
                input_channels=set(net_info["input_channels"]),
                output_channels=set(net_info["output_channels"]),
                processing_size=net_info["processing_size"],
            )
        )
    logger.info("finished loading networks")
    return loaded_nets, loaded_net_data, loaded_net_info


def init_network(architecture, lr, rng, train_path, optim_type, num_epochs, batches_per_epoch, end_lr, schedule_type, coarse_op_name, processing_scales, normalization, train_step, output_residuals=True, logger=None, system_type="qg"):
    if logger is None:
        logger = logging.getLogger("seq_net_init")
    processing_scales = set(processing_scales)
    if isinstance(architecture, str):
        architectures = [architecture]
    else:
        architectures = architecture
    if len(architectures) == 1:
        architectures = architecture * len(processing_scales)
    if len(architectures) != len(processing_scales):
        raise ValueError(f"must specify either 1 or {len(processing_scales)} architectures")

    # Determine input and output channels
    if train_step == 0:
        if system_type == "qg":
            in_channels = [f"q_{max(processing_scales)}"]
            out_channels = [f"q_scaled_forcing_{max(processing_scales)}to{min(processing_scales)}"]
        elif system_type == "ns":
            in_channels = [f"ns_uv_{max(processing_scales)}", f"ns_vort_{max(processing_scales)}"]
            out_channels = [f"ns_scaled_uv_corr_{max(processing_scales)}to{min(processing_scales)}"]
    elif 1 <= train_step < len(processing_scales):
        small, big = next(itertools.islice(itertools.pairwise(sorted(processing_scales)), train_step - 1, None))
        prev_scales = sorted(processing_scales)[:train_step]
        assert small in prev_scales
        if system_type == "qg":
            in_channels = [f"q_scaled_{max(processing_scales)}to{big}"] + [f"q_scaled_forcing_{max(processing_scales)}to{sz}" for sz in prev_scales]
        elif system_type == "ns":
            in_channels = [f"ns_scaled_uv_{max(processing_scales)}to{big}", f"ns_scaled_vort_{max(processing_scales)}to{big}"] + [f"ns_scaled_uv_corr_{max(processing_scales)}to{sz}" for sz in prev_scales]
        if big == max(processing_scales):
            if system_type == "qg":
                target_chan = f"q_total_forcing_{max(processing_scales)}"
            elif system_type == "ns":
                target_chan = f"ns_uv_corr_{max(processing_scales)}"
        else:
            if system_type == "qg":
                target_chan = f"q_scaled_forcing_{max(processing_scales)}to{big}"
            elif system_type == "ns":
                target_chan = f"ns_scaled_uv_corr_{max(processing_scales)}to{big}"
        if output_residuals:
            if system_type == "qg":
                out_channels = [f"residual:{target_chan}-q_scaled_forcing_{max(processing_scales)}to{small}"]
            elif system_type == "ns":
                out_channels = [f"residual:{target_chan}-ns_scaled_uv_corr_{max(processing_scales)}to{small}"]
        else:
            out_channels = [target_chan]
    else:
        raise ValueError(f"invalid train_step {train_step}")
    processing_size = determine_processing_size(input_channels=in_channels, output_channels=out_channels)
    net_data = NetData(
        input_channels=in_channels,
        output_channels=out_channels,
        processing_size=processing_size,
    )
    # Initialize network, etc.
    arch = architectures[train_step]
    logger.info("Training network: %s", arch)
    state, network_info = train_init_network(
        architecture=arch,
        lr=lr,
        rng=rng,
        input_channels=in_channels,
        output_channels=out_channels,
        processing_size=processing_size,
        train_path=train_path,
        optim_type=optim_type,
        num_epochs=num_epochs,
        batches_per_epoch=batches_per_epoch,
        end_lr=end_lr,
        schedule_type=schedule_type,
        coarse_op_name=coarse_op_name,
    )
    return state, network_info, net_data


def make_alt_source_computer(loaded_nets, loaded_net_data, model_params):

    def alt_source_computer(batch):
        alt_sources = {}
        for net, data in zip(loaded_nets, loaded_net_data, strict=True):
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
        return alt_sources

    return alt_source_computer


def make_batch_computer(net_data, alt_source_computer, model_params):
    output_size = determine_output_size(net_data.output_channels)

    def sample_loss(input_elem, target_elem, net):
        y = net(input_elem)
        y = make_basic_coarsener(net_data.processing_size, output_size, model_params)(y)
        mse = jnp.mean((y - target_elem)**2)
        return mse, y

    def batch_loss(net, input_chunk, target_chunk):
        losses, predictions = jax.vmap(
            functools.partial(
                sample_loss,
                net=net,
            )
        )(input_chunk, target_chunk)
        return jnp.mean(losses)

    def do_batch(batch, state):
        alt_sources = alt_source_computer(batch)
        input_chunk = make_chunk_from_batch(
            channels=net_data.input_channels,
            batch=batch,
            model_params=model_params,
            processing_size=net_data.processing_size,
            alt_source=alt_sources,
        )
        target_chunk = make_chunk_from_batch(
            channels=net_data.output_channels,
            batch=batch,
            model_params=model_params,
            processing_size=output_size,
            alt_source=alt_sources,
        )
        # Compute losses
        loss, grads = eqx.filter_value_and_grad(batch_loss)(state.net, input_chunk, target_chunk)
        # Update parameters
        out_state = state.apply_updates(grads)
        return out_state, loss

    return do_batch


def make_validation_stats_function(net_data, model_params, alt_source_computer):
    output_size = determine_output_size(net_data.output_channels)

    def make_samples(input_chunk, net):
        ys = jax.vmap(net)(input_chunk)
        return jax.vmap(make_basic_coarsener(net_data.processing_size, output_size, model_params))(ys)

    def compute_stats(batch, net):
        alt_sources = alt_source_computer(batch)
        input_chunk = make_chunk_from_batch(
            channels=net_data.input_channels,
            batch=batch,
            model_params=model_params,
            processing_size=net_data.processing_size,
            alt_source=alt_sources,
        )
        targets = make_non_residual_chunk_from_batch(
            channels=net_data.output_channels,
            batch=batch,
            model_params=model_params,
            processing_size=output_size,
        )
        samples = remove_residual_from_output_chunk(
            output_channels=net_data.output_channels,
            output_chunk=make_samples(input_chunk, net),
            batch=batch,
            model_params=model_params,
            processing_size=output_size,
            alt_source=alt_sources,
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


def make_train_loader(
    *,
    train_path,
    system_type,
    batch_size,
    loader_chunk_size,
    base_logger,
    np_rng,
    required_fields
):
    if system_type == "qg":
        return ThreadedPreShuffledSnapshotLoader(
            file_path=train_path,
            batch_size=batch_size,
            chunk_size=loader_chunk_size,
            buffer_size=10,
            seed=np_rng.integers(2**32).item(),
            base_logger=base_logger.getChild("train_loader"),
            fields=required_fields,
        )
    elif system_type == "ns":
        return ns_loader.NSThreadedPreShuffledSnapshotLoader(
            file_path=train_path,
            batch_size=batch_size,
            buffer_size=10,
            chunk_size=loader_chunk_size,
            seed=np_rng.integers(2**32).item(),
            base_logger=base_logger.getChild("train_loader"),
            fields=required_fields,
        )
    else:
        raise ValueError(f"unsupported system {system_type}")


def main():
    args = parser.parse_args()
    base_out_dir = pathlib.Path(args.out_dir)
    if base_out_dir.is_file():
        raise ValueError(f"Path must be a directory, not a file: {args.out_dir}")
    base_out_dir.mkdir(exist_ok=True)
    out_dir = base_out_dir / f"net{args.train_step}"
    out_dir.mkdir(exist_ok=True)
    utils.set_up_logging(level=args.log_level, out_file=out_dir/"run.log")
    logger = logging.getLogger("main")
    logger.info("Arguments: %s", vars(args))
    logger.info("Running train step %d of %d", args.train_step, len(set(args.processing_levels)))
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
    rng_ctr = jax.random.PRNGKey(seed=np_rng.integers(2**32).item())
    # Configure required elements for training
    train_path = (pathlib.Path(args.train_set) / "shuffled.hdf5").resolve()
    val_path = (pathlib.Path(args.val_set) / "data.hdf5").resolve()
    system_type = sniff_system_type(train_path)
    logger.info("System type %s", system_type)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    processing_scales = set(args.processing_levels)
    if len(processing_scales) < 2:
        logger.error("Must provide at least two processing scale levels")
        sys.exit(2)

    # Create data normalizer and its inverse
    model_params = load_model_params(train_path)
    coarse_op_name = model_params.scale_operator

    # Load previous networks
    loaded_nets, loaded_net_data, loaded_net_info = load_prev_networks(out_dir.parent, args.train_step, args.net_load_type, base_logger=logger)
    # Check that training sets match
    for net_i, net_info in enumerate(loaded_net_info):
        if pathlib.Path(net_info["train_path"]).resolve() != train_path:
            logger.error(f"mismatched train path for net %d", net_i)
            sys.exit(1)

    # Construct neural net
    rng, rng_ctr = jax.random.split(rng_ctr, 2)
    state, network_info, net_data = init_network(
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
        train_step=args.train_step,
        output_residuals=args.output_residuals,
        logger=logger.getChild("init_net"),
        system_type=system_type,
    )
    logger.info("Input channels: %s", net_data.input_channels)
    logger.info("Output channels: %s", net_data.output_channels)
    logger.info("Processing size: %d", net_data.processing_size)
    # Check overlaps
    for (a_i, a_ni), (b_i, b_ni) in itertools.pairwise(enumerate(itertools.chain(loaded_net_data, [net_data]))):
        if not set(map(name_remove_residual, a_ni.output_channels)).intersection(b_ni.input_channels):
            logger.error("no intersection between networks %d and %d", a_i, b_i)
            sys.exit(1)

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
                for data in itertools.chain([net_data], loaded_net_data)
            )
        )
    )
    logger.info("Required fields: %s", required_fields)

    with contextlib.ExitStack() as train_context:
        # Open data files
        train_loader = train_context.enter_context(
            make_train_loader(
                train_path=train_path,
                system_type=system_type,
                batch_size=args.batch_size,
                loader_chunk_size=args.loader_chunk_size,
                base_logger=logger.getChild("train_loader"),
                np_rng=np_rng,
                required_fields=required_fields,
            )
        )
        val_loader = train_context.enter_context(
            make_val_loader(
                file_path=val_path,
                required_fields=required_fields,
                system_type=system_type,
            )
        )

        # Training functions
        alt_source_computer = make_alt_source_computer(
            loaded_nets=loaded_nets,
            loaded_net_data=loaded_net_data,
            model_params=model_params,
        )
        train_batch_fn = eqx.filter_jit(
            make_batch_computer(
                net_data=net_data,
                alt_source_computer=alt_source_computer,
                model_params=model_params,
            ),
            donate="all",
        )
        # Determine fixed validation samples
        val_samp_rng = np.random.default_rng(seed=args.val_sample_seed)
        val_traj_idxs = val_samp_rng.integers(low=0, high=val_loader.num_trajs, size=args.num_val_samples, dtype=np.uint64)
        val_step_idxs = val_samp_rng.integers(low=0, high=val_loader.num_steps, size=args.num_val_samples, dtype=np.uint64)
        val_stats_fn = eqx.filter_jit(
            make_validation_stats_function(
                net_data=net_data,
                model_params=model_params,
                alt_source_computer=alt_source_computer,
            )
        )

        # Running statistics
        min_mean_loss = None
        min_val_loss = None

        # Training loop
        epoch_reports = []
        save_names_written = set()
        save_names_permanent = set()
        save_names_mapping = {}
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

            # Validation step
            val_stat_report = None
            val_loss = None
            if epoch % args.val_interval == 0:
                logger.info("Starting validation for epoch %d", epoch)
                val_stat_report = do_validation(
                    train_state=state,
                    loader=val_loader,
                    sample_stat_fn=val_stats_fn,
                    traj=val_traj_idxs,
                    step=val_step_idxs,
                    logger=logger.getChild(f"{epoch:05d}_val"),
                )
                val_loss = val_stat_report["standard_mse"]
                logger.info("Finished validation for epoch %d", epoch)

            # Save snapshots
            saved_names = []
            # Save the network after each epoch
            epoch_name = f"epoch{epoch:04d}"
            epoch_file = weights_dir / f"{epoch_name}.eqx"
            save_network(epoch_name, output_dir=weights_dir, state=state, base_logger=logger)
            save_names_written.add(epoch_name)
            # Link checkpoint
            utils.atomic_symlink(epoch_file, weights_dir / "checkpoint.eqx")
            save_names_mapping["checkpoint"] = epoch_name
            saved_names.append("checkpoint")
            # Link best loss (maybe)
            if min_mean_loss is None or (math.isfinite(mean_loss) and mean_loss <= min_mean_loss):
                min_mean_loss = mean_loss
                utils.atomic_symlink(epoch_file, weights_dir / "best_loss.eqx")
                save_names_mapping["best_loss"] = epoch_name
                saved_names.append("best_loss")
            if val_loss is not None and (min_val_loss is None or (math.isfinite(val_loss) and val_loss <= min_val_loss)):
                min_val_loss = val_loss
                utils.atomic_symlink(epoch_file, weights_dir / "best_val_loss.eqx")
                save_names_mapping["best_val_loss"] = epoch_name
                saved_names.append("best_val_loss")
            # Save interval
            if epoch % args.save_interval == 0:
                utils.atomic_symlink(epoch_file, weights_dir / "interval.eqx")
                save_names_mapping["interval"] = epoch_name
                saved_names.append("interval")
            # Permanently fix epoch (if requested)
            if (epoch % args.save_interval == 0) or (epoch == args.num_epochs):
                save_names_permanent.add(epoch_name)
                saved_names.append(epoch_name)
            # Save the final epoch with a special name
            if epoch == args.num_epochs:
                utils.atomic_symlink(epoch_file, weights_dir / "final_snapshot.eqx")
                save_names_mapping["final_snapshot"] = epoch_name
                saved_names.append("final_snapshot")
            logger.info("Wrote file and link names: %s", saved_names)
            # Clean up any now unlinked files
            save_names_to_remove = (save_names_written - save_names_permanent) - {v for v in save_names_mapping.values() if v is not None}
            for name_to_remove in save_names_to_remove:
                try:
                    logger.debug("Removing weights file %s", name_to_remove)
                    os.remove(weights_dir / f"{name_to_remove}.eqx")
                    save_names_written.discard(name_to_remove)
                except FileNotFoundError:
                    logger.warning("Tried to remove missing weights file %s", name_to_remove)

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
