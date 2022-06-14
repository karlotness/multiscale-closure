import argparse
import pathlib
import math
import re
import random
import itertools
import contextlib
import json
import jax
import jax.numpy as jnp
import optax
import flax.serialization
import flax.training.train_state
import flax.struct
import numpy as np
import logging
import time
import functools
from typing import Callable
from systems.qg.qg_model import QGModel
from systems.qg import utils as qg_utils
from systems.qg.loader import ThreadedQGLoader, SimpleQGLoader, qg_model_from_hdf5
import utils
from methods import ARCHITECTURES

MEAN_QG_VAL_VALUE = 2.716908e-07

parser = argparse.ArgumentParser(description="Train neural networks for closure")
parser.add_argument("out_dir", type=str, help="Directory to store output (created if non-existing)")
parser.add_argument("train_set", type=str, help="Directory with training examples")
parser.add_argument("val_set", type=str, help="Directory with validation examples")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
parser.add_argument("--batch_size", type=int, default=25, help="Batch size")
parser.add_argument("--train_epochs", type=int, default=700, help="Number of training epochs")
parser.add_argument("--batches_per_epoch", type=int, default=100, help="Number of batches per epoch (with replacement)")
parser.add_argument("--rollout_length", type=str, default="5 300@10 500@20 600@40", help="Schedule for rollout length (a space-separated list of <start_epoch>@<length>, if start_epoch omitted, it is implicitly zero)")
parser.add_argument("--val_steps", type=int, default=500, help="Number of steps to run in validation")
parser.add_argument("--val_samples", type=int, default=10, help="Number of batches to run in a validation pass")
parser.add_argument("--seed", type=int, default=None, help="Seed to use with RNG (if None, select automatically)")
parser.add_argument("--architecture", type=str, default="closure-cnn-v1", help="Choose architecture to train", choices=sorted(ARCHITECTURES.keys()))


class RecurrentTrainState(flax.training.train_state.TrainState):
    memory_init_fn: Callable = flax.struct.field(pytree_node=False)


def mean_l1err_loss(real_step, est_step):
    err = jnp.abs(est_step - real_step)
    return jnp.mean(err)

def qg_mean_l1err_loss(real_step, est_step):
    err = jnp.abs(est_step - real_step)
    return jnp.mean(err) * (1 / MEAN_QG_VAL_VALUE)

def relerr_loss(real_step, est_step):
    err = jnp.abs(est_step - real_step)
    return jnp.mean(err / jnp.abs(real_step))


def compare_val_loss(a, b, rollout_len):
    l_a = np.mean(a[:rollout_len])
    l_b = np.mean(b[:rollout_len])
    return l_a - l_b


def init_network(architecture, lr, weight_decay, rng, small_model):
    dummy_u = jnp.zeros((small_model.nz, small_model.ny, small_model.nx))
    dummy_v = jnp.zeros((small_model.nz, small_model.ny, small_model.nx))
    net = ARCHITECTURES[architecture]()
    rng_1, rng_2 = jax.random.split(rng, 2)
    params = net.init(rng_1, dummy_u, dummy_v)
    # Change initialization
    defs, tree = jax.tree_util.tree_flatten(params)
    new_params = []
    for p in defs:
        rng_use, rng_2 = jax.random.split(rng_2)
        new_params.append(jax.random.normal(rng_use, shape=p.shape) * 1e-5)
    params = jax.tree_util.tree_unflatten(tree, new_params)
    optim = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    return net, RecurrentTrainState.create(
        apply_fn=functools.partial(net.apply, method=net.parameterization),
        memory_init_fn=functools.partial(net.apply, method=net.init_memory),
        params=params,
        tx=optim,
    )


def epoch_batch_iterators(train_file, batch_size, rollout_length_str, seed=None, base_logger=None, max_buffer_steps=100000):
    if base_logger is None:
        logger = logging.getLogger("epoch_iter")
    else:
        logger = base_logger.getChild("epoch_iter")
    rollout_re = re.compile("^(?:(?P<ep_start>\d+)@)?(?P<rollout>\d+)$")
    # Parse schedule for rollout lengths
    rollout_schedule = []
    for spec in rollout_length_str.split():
        if rgx_match := rollout_re.match(spec):
            if rgx_match.group("ep_start") is not None:
                epoch_start = int(rgx_match.group("ep_start"))
            else:
                epoch_start = 0
            rollout = int(rgx_match.group("rollout"))
            rollout_schedule.append((epoch_start, rollout))
        else:
            raise ValueError(f"invalid rollout specification {spec}")
    if rollout_schedule[0][0] != 0:
        raise ValueError(f"rollout schedule starts at epoch {rollout_schedule[0][0]}, but must start at 0")
    rollout_steps = []
    for i in range(len(rollout_schedule) - 1):
        epoch_start, rollout = rollout_schedule[i]
        next_start, _ = rollout_schedule[i + 1]
        if epoch_start >= next_start:
            raise ValueError(f"epoch starts must be in increasing order {epoch_start} >= {next_start}")
        rollout_steps.append((rollout, range(next_start - epoch_start)))
    rollout_steps.append((rollout_schedule[-1][1], itertools.count()))
    # Iterate over schedule steps, create loader (in context manager),yield the loader's iterator for the correct number of epochs
    rng = random.Random(seed)
    for rollout_steps, step_iter in rollout_steps:
        loader_seed = rng.randint(0, 2**32)
        buffer_size = max(1, math.floor(max_buffer_steps / (batch_size * rollout_steps)))
        logger.info("New loader for %d rollout steps, buffering %d batches", rollout_steps, buffer_size)
        logger.debug("Using %d as seed for new loader", loader_seed)
        with ThreadedQGLoader(
                file_path=train_file,
                batch_size=batch_size,
                rollout_steps=rollout_steps,
                split_name="train",
                base_logger=logger,
                buffer_size=buffer_size,
                seed=loader_seed,
        ) as loader:
            for _step in step_iter:
                yield rollout_steps, loader.iter_batches()


def make_train_batch_computer(small_model, loss_fn):
    def do_batch(batch, train_state):
        batch_loss_func = jax.vmap(qg_utils.get_online_batch_loss, in_axes=(0, None, None, None, None, None))
        def _get_losses(params):
            step_losses = batch_loss_func(batch, train_state.apply_fn, params, small_model, loss_fn, train_state.memory_init_fn)
            return jnp.mean(step_losses)

        loss, grads = jax.value_and_grad(_get_losses)(train_state.params)
        new_train_state = jax.lax.cond(
            jnp.isfinite(jnp.abs(loss)),
            lambda: train_state.apply_gradients(grads=grads),
            lambda: train_state,
        )
        return new_train_state, loss

    return do_batch


def make_val_computer(small_model, loss_fn):
    def do_traj(traj, train_state):
        batch_loss_func = jax.vmap(
            qg_utils.get_online_batch_loss,
            in_axes=(0, None, None, None, None, None)
        )
        step_losses = batch_loss_func(
            traj,
            train_state.apply_fn,
            train_state.params,
            small_model,
            loss_fn,
            train_state.memory_init_fn,
        )
        step_losses = jnp.nan_to_num(step_losses[0], nan=jnp.inf, posinf=jnp.inf, neginf=jnp.inf)
        uncorrected_step_losses = batch_loss_func(
            traj,
            lambda params, u, v, memory: (jnp.zeros_like(u), jnp.zeros_like(v), None),
            None,
            small_model,
            loss_fn,
            lambda params, u, v: None,
        )
        uncorrected_step_losses = jnp.nan_to_num(uncorrected_step_losses[0], nan=jnp.inf, posinf=jnp.inf, neginf=jnp.inf)
        return step_losses, uncorrected_step_losses

    return do_traj


def do_epoch(train_state, train_func, epoch_num, num_batches, batch_iter, logger):
    losses = []
    for batch in itertools.islice(batch_iter, num_batches):
        train_state, loss = train_func(batch, train_state)
        losses.append(loss)
    losses = np.array(losses)
    num_skipped = np.count_nonzero(np.logical_not(np.isfinite(losses)))
    if num_skipped > 0:
        logger.warning("Skipped %d batches due to nan/inf loss", num_skipped)
    return train_state, float(np.nanmean(losses))


def do_validation(val_batch_iter, train_state, val_func, val_file, logger):
    traj_losses = []
    uncorrected_losses = []
    for batch in val_batch_iter:
        logger.debug("Doing validation")
        traj_loss, uncorrected_loss = val_func(batch, train_state)
        traj_losses.append(traj_loss)
        uncorrected_losses.append(uncorrected_loss)
    return np.mean(traj_losses, axis=0), np.mean(uncorrected_losses, axis=0)


def save_network(output_name, output_dir, net, params, base_logger=None):
    if base_logger is None:
        logger = logging.getLogger("save")
    else:
        logger = base_logger.getChild("save")
    output_dir = pathlib.Path(output_dir)
    with open(output_dir / f"{output_name}.flaxnn", "wb") as out_file:
        out_file.write(flax.serialization.to_bytes(params))
    with open(output_dir / f"{output_name}.json", "w", encoding="utf8") as out_file:
        json.dump(net.net_description(), out_file)
    logger.info("Saved network parameters to %s in %s", output_name, output_dir)


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
    loader_rng_init = random.Random(seed)
    train_file = pathlib.Path(args.train_set) / "data.hdf5"
    val_file = pathlib.Path(args.val_set) / "data.hdf5"
    train_small_model = qg_model_from_hdf5(file_path=train_file, model="small")
    val_small_model = qg_model_from_hdf5(file_path=val_file, model="small")
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    # Construct neural net
    rng, rng_ctr = jax.random.split(rng_ctr, 2)
    logger.info("Training network %s", args.architecture)
    net, train_state = init_network(
        architecture=args.architecture,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rng=rng,
        small_model=train_small_model,
    )
    train_func = jax.jit(
        make_train_batch_computer(
            small_model=train_small_model,
            loss_fn=qg_mean_l1err_loss,
        )
    )
    val_func = jax.jit(
        make_val_computer(
            small_model=val_small_model,
            loss_fn=relerr_loss,
        )
    )
    # Begin training
    start = time.perf_counter()
    epoch_stats = []
    best_val_loss = None
    with contextlib.ExitStack() as train_process_context:
        epoch_batch_iter = train_process_context.enter_context(
            contextlib.closing(
                epoch_batch_iterators(
                    train_file=train_file,
                    batch_size=args.batch_size,
                    rollout_length_str=args.rollout_length,
                    seed=loader_rng_init.randint(0, 2**32),
                    base_logger=logger,
                )
            )
        )
        val_loader = train_process_context.enter_context(
            ThreadedQGLoader(
                file_path=val_file,
                batch_size=1,
                rollout_steps=args.val_steps,
                split_name="val",
                base_logger=logger,
                buffer_size=args.val_samples,
                seed=loader_rng_init.randint(0, 2**32),
                num_workers=1,
            )
        )
        for epoch, (train_rollout_len, raw_batch_iter) in zip(range(args.train_epochs), epoch_batch_iter):
            with contextlib.closing(raw_batch_iter) as batch_iter:
                # Do training phase
                logger.info("Starting epoch %d of %d (%d batches)", epoch + 1, args.train_epochs, args.batches_per_epoch)
                train_start = time.perf_counter()
                train_state, mean_train_loss = do_epoch(
                    train_state=train_state,
                    train_func=train_func,
                    epoch_num=epoch,
                    num_batches=args.batches_per_epoch,
                    batch_iter=batch_iter,
                    logger=logger.getChild("train_epoch"),
                )
                train_elapsed = time.perf_counter() - train_start
                logger.info("Finished epoch %d in %f sec. train_loss=%f", epoch + 1, train_elapsed, mean_train_loss)
            # Run validation phase
            with contextlib.closing(val_loader.iter_batches()) as val_batch_iter:
                logger.info("Starting validation after epoch %d", epoch + 1)
                val_start = time.perf_counter()
                val_loss_horizons, uncorr_loss_horizons = do_validation(
                    val_batch_iter=itertools.islice(val_batch_iter, args.val_samples),
                    train_state=train_state,
                    val_func=val_func,
                    val_file=val_file,
                    logger=logger.getChild("validate"),
                )
                val_elapsed = time.perf_counter() - val_start
                logger.info("Finished validation for epoch %d in %f sec.", epoch + 1, val_elapsed)
            val_report_horizons = {
                "end": {
                    "val": float(np.mean(val_loss_horizons)),
                    "uncorr": float(np.mean(uncorr_loss_horizons)),
                }
            }
            for horizon in sorted({5, 10, 25, 50, 100, 250, 500, 750, 1000}):
                if horizon - 1 < val_loss_horizons.shape[0]:
                    horiz_val = float(np.mean(val_loss_horizons[:horizon]))
                    uncorr_val = float(np.mean(uncorr_loss_horizons[:horizon]))
                    val_report_horizons[f"{horizon:d}"] = {
                        "val": horiz_val,
                        "uncorr": uncorr_val,
                    }
                    logger.info(
                        "Validation horizon %d steps: %f%% (vs. %f%%)",
                        horizon,
                        horiz_val * 100,
                        uncorr_val * 100,
                    )
            # If validation improved, store snapshot
            if best_val_loss is None or compare_val_loss(val_loss_horizons, best_val_loss, rollout_len=train_rollout_len) < 0:
                logger.info("Validation performance improved, saving weights")
                save_network("best_val", weights_dir, net=net, params=train_state.params, base_logger=logger)
                best_val_loss = val_loss_horizons
                new_best_val = True
            else:
                new_best_val = False
            # Record some training statistics
            epoch_stats.append(
                {
                    "epoch": epoch,
                    "train_elapsed_secs": train_elapsed,
                    "mean_train_loss": mean_train_loss,
                    "mean_val_losses": val_report_horizons,
                    "new_best_val": new_best_val,
                }
            )
    # Store final weights
    save_network("final_net", weights_dir, net=net, params=train_state.params, base_logger=logger)
    end = time.perf_counter()
    # Finished training
    logger.info("Finished training in %f sec", end - start)
    # Save train statistics
    with open(out_dir / "epoch_stats.json", "w", encoding="utf8") as stats_file:
        json.dump(epoch_stats, stats_file)
    logger.info("Recorded epoch statistics to %s", out_dir / "epoch_stats.json")


if __name__ == "__main__":
    main()
