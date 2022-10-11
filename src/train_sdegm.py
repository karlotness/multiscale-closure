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
import optax
import flax.serialization
import flax.training.train_state
import flax.struct
from flax.core import frozen_dict
import numpy as np
import logging
import time
import functools
from typing import Callable
from systems.qg.qg_model import QGModel
from systems.qg import utils as qg_utils
from systems.qg.loader import ThreadedQGLoader, SimpleQGLoader, qg_model_from_hdf5
import jax_utils
import utils
from methods import ARCHITECTURES
from methods.score_sde.models import utils as mutils
from methods.score_sde import losses as losses_module
from methods.score_sde import sampling
from methods.score_sde.configs.dummy_qg_snap_config import build_dummy_qg_snap_config
from methods.score_sde import datasets, sde_lib

MEAN_QG_VAL_VALUE = 2.716908e-07

parser = argparse.ArgumentParser(description="Train neural networks for closure")
parser.add_argument("out_dir", type=str, help="Directory to store output (created if non-existing)")
parser.add_argument("train_set", type=str, help="Directory with training examples")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
parser.add_argument("--save_interval", type=int, default=1, help="Number of epochs between saves")
parser.add_argument("--seed", type=int, default=None, help="Seed to use with RNG (if None, select automatically)")
parser.add_argument("--architecture", type=str, default="sdegm-ncsnpp", help="Choose architecture to train", choices=sorted(ARCHITECTURES.keys()))
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batches_per_epoch", type=int, default=100, help="Training batches per epoch")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for optimizer")
parser.add_argument("--num_epoch_samples", type=int, default=15, help="Number of samples to draw after each epoch")


def save_network(output_name, output_dir, state, base_logger=None):
    if base_logger is None:
        logger = logging.getLogger("save")
    else:
        logger = base_logger.getChild("save")
    output_dir = pathlib.Path(output_dir)
    with open(output_dir / f"{output_name}.flaxnn.PART", "wb") as out_file:
        out_file.write(
            flax.serialization.to_bytes(
                frozen_dict.freeze(
                    {
                        "params": state.params,
                        "ema_rate": state.ema_rate,
                        "params_ema": state.params_ema,
                        "model_state": state.model_state,
                    }
                )
            )
        )
    try:
        os.remove(output_dir / f"{output_name}.flaxnn")
    except FileNotFoundError:
        pass
    os.rename(output_dir / f"{output_name}.flaxnn.PART", output_dir / f"{output_name}.flaxnn")
    logger.info("Saved network parameters to %s in %s", output_name, output_dir)



def make_epoch_computer(sde, score_model, optimize_fn, reduce_mean, continuous, likelihood_weighting, scaler, inverse_scaler, num_steps, batch_size, train_data):
    train_step_fn = losses_module.get_step_fn(sde, score_model, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)

    def do_batch(carry, _x):
        rng_ctr, state = carry
        rng, rng_ctr = jax.random.split(rng_ctr)
        batch_idx = jax.random.randint(rng, (batch_size, ), 0, train_data.shape[0], dtype=jnp.uint32)
        batch = jnp.take(train_data, batch_idx, axis=0)
        batch = {"image": scaler(batch)}
        (_, new_state), loss = train_step_fn((rng, state), batch)
        # Guard against NaN losses, keep old state
        out_state = jax.lax.cond(
            jnp.isfinite(loss),
            lambda: new_state,
            lambda: state,
        )
        return (rng_ctr, out_state), loss

    def do_epoch(state, rng):
        (last_rng, new_state), losses = jax.lax.scan(do_batch, (rng, state), None, length=num_steps)
        return new_state, last_rng, losses

    return do_epoch


def make_sample_computer(config, sde, score_model, inverse_scaler, sampling_eps, num_epoch_samples, data_min, data_max):

    sampling_shape = (num_epoch_samples, config.data.image_size,
                      config.data.image_size, config.data.num_channels)
    sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler, sampling_eps)

    def do_sample(rng, state):
        sample, nfe = sampling_fn(rng, state)
        sample = (sample * (data_max - data_min)) + data_min
        return sample, nfe

    return do_sample



def init_network(model_name, config, rng):
    model_def = functools.partial(ARCHITECTURES[model_name], config=config)
    input_shape = (config.training.batch_size, config.data.image_size, config.data.image_size, config.data.num_channels)
    label_shape = (1, )
    fake_input = jnp.zeros(input_shape)
    fake_label = jnp.zeros(label_shape, dtype=jnp.int32)
    params_rng, dropout_rng, rng = jax.random.split(rng, 3)
    model = model_def()
    variables = model.init({'params': params_rng, 'dropout': dropout_rng}, fake_input, fake_label)
    # Variables is a `flax.FrozenDict`. It is immutable and respects functional programming
    init_model_state, initial_params = variables.pop('params')
    # So far we have model, init_model_state, initial_params
    optimizer = losses_module.get_optimizer(config)
    optim_state = optimizer.init(initial_params)
    state = mutils.State(
        lr=config.optim.lr,
        ema_rate=config.model.ema_rate,
        model_state=init_model_state,
        params_ema=initial_params,
        rng=rng,
        step=0,
        apply_fn=model.apply,
        params=initial_params,
        tx=optimizer,
        opt_state=optim_state,
    )
    return model, state
    # mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
    #                      model_state=init_model_state,
    #                      ema_rate=config.model.ema_rate,
    #                      params_ema=initial_params,
    #                      rng=rng)  # pytype: disable=wrong-keyword-args
    # return model, state

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
    if jax.device_count() != 1:
        logger.error("Cannot handle multiple JAX devices, found %d", jax.device_count())
        raise ValueError(f"Too many JAX devices {jax.device_count()}")
    # Build dummy config
    config = build_dummy_qg_snap_config(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        batches_per_epoch=args.batches_per_epoch,
        lr=args.lr,
    )
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
    logger.info("Training network %s", args.architecture)
    score_model, state = init_network(
        model_name=args.architecture,
        config=config,
        rng=rng,
    )
    initial_step = 0
    rng = state.rng

    # Load dataset
    logger.info("Loading data")
    with h5py.File(train_path, "r") as train_file:
        train_data = train_file["final_q_steps"][:]
    # Mask out NaNs, put batches at the back, and move to GPU
    train_data = jax.device_put(np.moveaxis(train_data[np.all(np.isfinite(train_data), axis=(-1, -2, -3))], 1, -1))
    # Rescale data to [0, 1]
    train_min = jnp.min(train_data)
    train_max = jnp.max(train_data)
    train_data = (train_data - train_min) / (train_max - train_min)
    logger.info("Finished loading data")
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDE
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses_module.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    # TRAINING FUNCTIONS
    train_epoch_fn = jax.jit(
        make_epoch_computer(
            sde=sde,
            score_model=score_model,
            optimize_fn=optimize_fn,
            reduce_mean=reduce_mean,
            continuous=continuous,
            likelihood_weighting=likelihood_weighting,
            scaler=scaler,
            inverse_scaler=inverse_scaler,
            num_steps=args.batches_per_epoch,
            batch_size=args.batch_size,
            train_data=train_data
        )
    )
    sample_fn = jax.jit(
        make_sample_computer(
            config=config,
            sde=sde,
            score_model=score_model,
            inverse_scaler=inverse_scaler,
            sampling_eps=sampling_eps,
            num_epoch_samples=args.num_epoch_samples,
            data_min=train_min,
            data_max=train_max,
        )
    )

    min_mean_loss = None
    rng, rng_ctr = jax.random.split(rng_ctr, 2)
    for epoch in range(args.num_epochs):
        logger.info("Starting epoch %d of %d", epoch + 1, args.num_epochs)
        epoch_start = time.perf_counter()
        state, rng, losses = train_epoch_fn(state, rng)
        mean_loss = jax.device_get(jnp.mean(losses))
        final_loss = jax.device_get(losses[-1])
        epoch_end = time.perf_counter()
        logger.info("Finished epoch %d in %f sec", epoch + 1, epoch_end - epoch_start)
        logger.info("Epoch %d mean loss %f", epoch + 1, mean_loss)
        logger.info("Epoch %d final loss %f", epoch + 1, final_loss)

        # Save weights
        if min_mean_loss is None or (np.isfinite(mean_loss) and mean_loss <= min_mean_loss):
            min_mean_loss = mean_loss
            save_network("best_loss", output_dir=weights_dir, state=state, base_logger=logger)
        if epoch % args.save_interval == 0:
            save_network("epoch", output_dir=weights_dir, state=state, base_logger=logger)

        # Do epoch sampling
        logger.info("Starting sampling")
        rng, sample_rng = jax.random.split(rng)
        sample_start = time.perf_counter()
        sample, nfe = jax.device_get(sample_fn(sample_rng, state))
        sample_end = time.perf_counter()
        sample = np.moveaxis(sample, -1, 1)
        logger.info("Finished sampling in %f sec", sample_end - sample_start)
        sample_min = np.min(sample)
        sample_max = np.max(sample)
        sample_mean = np.mean(sample)
        logger.info("Sample stats: min=%g, max=%g, mean=%g", sample_min, sample_max, sample_mean)
        epoch_sample_path = samples_dir / f"samples_ep{epoch + 1:05d}.npz"
        np.savez(epoch_sample_path, sample=sample, nfe=nfe)
        logger.info("Saved samples to %s", epoch_sample_path)


if __name__ == "__main__":
    main()
