import argparse
import pathlib
import math
import re
import random
import itertools
import contextlib
import jax
import jax.numpy as jnp
import optax
import flax.serialization
from flax.training.train_state import TrainState
import numpy as np
import logging
import time
from systems.qg.qg_model import QGModel
from systems.qg.loader import ThreadedQGLoader, qg_model_from_hdf5
import utils
from methods import ARCHITECTURES

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
parser.add_argument("--rollout_length", type=str, default="40", help="Schedule for rollout length (a space-separated list of <start_epoch>@<length>, if start_epoch omitted, it is implicitly zero)")
parser.add_argument("--seed", type=int, default=None, help="Seed to use with RNG (if None, select automatically)")
parser.add_argument("--architecture", type=str, default="closure-cnn-v1", help="Choose architecture to train", choices=sorted(ARCHITECTURES.keys()))


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
    #optim = optax.sgd(learning_rate=lr)
    return net, TrainState.create(
        apply_fn=net.apply,
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
                yield loader.iter_batches()


def do_epoch(train_state, epoch_num, num_batches, batch_iter, logger):
    logger.info("Starting epoch %d for %d batches", epoch_num, num_batches)
    start = time.perf_counter()
    for batch in itertools.islice(batch_iter, num_batches):
        loss = 0
    end = time.perf_counter()
    logger.info("Finished epoch %d in %f sec", epoch_num, end - start)
    return train_state, loss


def save_network(output_path, params, base_logger=None):
    if base_logger is None:
        logger = logging.getLogger("save")
    else:
        logger = base_logger.getChild("save")
    with open(output_path, "wb") as out_file:
        out_file.write(flax.serialization.to_bytes(params))
        logger.info("Saved network parameters to %s", output_path)


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
    train_file = pathlib.Path(args.train_set) / "data.hdf5"
    val_file = pathlib.Path(args.val_set) / "data.hdf5"
    train_small_model = qg_model_from_hdf5(file_path=train_file, model="small")
    val_small_model = qg_model_from_hdf5(file_path=val_file, model="small")
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
    # Begin training
    start = time.perf_counter()
    with contextlib.closing(
            epoch_batch_iterators(
                train_file=train_file,
                batch_size=args.batch_size,
                rollout_length_str=args.rollout_length,
                seed=seed,
                base_logger=logger,
            )
    ) as epoch_batch_iter:
        for epoch, batch_iter in zip(range(args.train_epochs), epoch_batch_iter):
            train_state, mean_train_loss = do_epoch(
                train_state=train_state,
                epoch_num=epoch,
                num_batches=args.batches_per_epoch,
                batch_iter=batch_iter,
                logger=logger.getChild("train_epoch"),
            )
            # Run validation

            # If validation improved, store snapshot
    # Store final weights
    save_network(out_dir / "final_net.flaxnn", params=train_state.params, base_logger=logger)
    end = time.perf_counter()
    # Finished training
    logger.info("Finished training in %f sec", end - start)


if __name__ == "__main__":
    main()
