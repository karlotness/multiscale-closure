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
import importlib
from systems.qg.loader import ThreadedPreShuffledSnapshotLoader, SimpleQGLoader, AggregateLoader, FillableDataLoader, SnapshotStates
from systems.qg import coarsen, diagnostics as qg_spec_diag, utils as qg_utils
import pyqg_jax
from methods import ARCHITECTURES
from generate_data import make_parameter_sampler
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
parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "warmup1-cosine", "ross22"], help="What learning rate schedule")
parser.add_argument("--network_zero_mean", action="store_true", help="Constrain the network to zero mean outputs")
parser.add_argument("--loader_chunk_size", type=int, default=10850, help="Chunk size to read before batching")
parser.add_argument("--net_weight_continue", type=str, default=None, help="Network weights to load and continue training")
parser.add_argument("--live_gen_start_epoch", type=int, default=1, help="")
parser.add_argument("--live_gen_interval", type=int, default=1, help="")
parser.add_argument("--live_gen_sampler", type=str, default="rand-eddy-to-jet-1.0", help="")
parser.add_argument("--live_gen_candidates", type=int, default=0, help="")
parser.add_argument("--live_gen_winners", type=int, default=0, help="")
parser.add_argument("--live_gen_mode", type=str, default="add-hardest", choices=["add-hardest", "network-noise", "network-noise-onestep", "schedule-only"], help="")
parser.add_argument("--live_gen_net_steps", type=int, default=5, help="")
parser.add_argument("--live_gen_base_data", type=str, default=None, help="")


def live_sample_get_args(train_path, required_fields):
    q_forcing_sizes = set()
    for field in required_fields:
        if m := re.match(r"^q_total_forcing_(?P<size>\d+)$", field):
            q_forcing_sizes.add(int(m.group("size")))
    with h5py.File(train_path, "r") as data_file:
        dt = data_file["params"]["dt"][()].item()
        subsample = data_file["params"]["subsample"][()].item()
        num_warmup_steps = math.ceil(155520000.0 / dt)
        _traj_store_steps = data_file["source"]["step"][:].max()
        q_size = data_file["shuffled"][0]["q"].shape[-1]
        num_steps = (_traj_store_steps * subsample) + num_warmup_steps
        return {
            "dt": dt,
            "subsample": subsample,
            "num_warmup_steps": num_warmup_steps,
            "num_steps": num_steps,
            "q_size": q_size,
            "q_forcing_sizes": list(q_forcing_sizes),
        }


def make_live_gen_func(start_epoch, interval, sample_conf, num_candidates, num_winners, dt, num_steps, num_warmup_steps, subsample, np_rng, logger, model_params, net_info, q_size, q_forcing_sizes, live_gen_mode, train_path, rollout_net_steps):

    def no_gen_traj_func(epoch, rng_ctr, net):
        return (
            [],
            {
                "num_winners": 0,
                "num_candidates": 0,
            },
            rng_ctr,
        )

    # User requested that no trajectories be generated
    if num_candidates == 0:
        logger.info("Will generate *NO* live trajectories")
        return no_gen_traj_func

    assert num_winners <= num_candidates
    assert start_epoch > 0
    assert num_steps > num_warmup_steps

    # late import to avoid issues with circular imports before initialization
    make_parameterized_stepped_model = importlib.import_module("cascaded_online_eval").make_parameterized_stepped_model
    make_ke_time_computer = importlib.import_module("online_ensemble_compare").make_ke_time_computer
    NetData = importlib.import_module("cascaded_eval").NetData
    param_sampler = make_parameter_sampler(args_config=sample_conf, np_rng=np_rng)
    coarse_cls = coarsen.COARSEN_OPERATORS[model_params.scale_operator]
    big_model = model_params.qg_models["big_model"]
    processing_model = model_params.qg_models[net_info["processing_size"]]
    net_data = NetData(
        input_channels=net_info["input_channels"],
        output_channels=net_info["output_channels"],
        processing_size=net_info["processing_size"],
    )
    q_coarsener = coarse_cls(big_model=big_model, small_nx=q_size)
    forcing_coarseners = {s: coarse_cls(big_model=big_model, small_nx=s) for s in q_forcing_sizes}

    def make_reference_traj(rng, sys_params):
        nonlocal q_coarsener
        nonlocal forcing_coarseners
        # Prepare for stepping
        stepped_model = pyqg_jax.steppers.SteppedModel(
            big_model,
            pyqg_jax.steppers.AB3Stepper(dt=dt),
        )
        # initial state
        init_state = stepped_model.create_initial_state(rng)

        # do warmup phase
        def step_until_warmup(carry, _x):
            prev_big_state = carry
            next_big_state = stepped_model.step_model(prev_big_state)
            return next_big_state, None

        big_warmed_up_step, _ys = jax.lax.scan(
            step_until_warmup,
            init_state,
            None,
            length=num_warmup_steps,
        )

        # collect reference steps
        def ref_steps(carry, _x):
            prev_big_state = carry
            next_big_state = stepped_model.step_model(prev_big_state)
            # Produce new "main size" state for output
            prev_small_q = q_coarsener.coarsen(prev_big_state.state.q)
            prev_forcings = {
                s: op.compute_q_total_forcing(prev_big_state.state.q)
                for s, op in forcing_coarseners.items()
            }
            return next_big_state, (prev_small_q, prev_forcings)

        _carry, (ref_q, ref_forcings) = jax_utils.strided_scan(
            ref_steps,
            big_warmed_up_step,
            None,
            length=num_steps - num_warmup_steps,
            stride=subsample,
        )
        return ref_q, ref_forcings

    def get_final_net_step(net, init_q, sys_params):
        qg_model_params = qg_utils.qg_model_to_args(q_coarsener.small_model)
        param_step_func = make_parameterized_stepped_model(
            nets=[net],
            net_data=[net_data],
            model_params=model_params,
            qg_model_args=qg_model_params,
            dt=dt,
        )
        states = param_step_func(
            initial_q=init_q,
            num_steps=num_steps - num_warmup_steps,
            subsampling=1,
            sys_params=sys_params,
            skip_steps=(num_steps - num_warmup_steps) - 1,
        )
        assert states.q.shape[0] == 1
        return states.q[0]

    def get_ke_relerr(ref_q, net_q):
        ke_time = make_ke_time_computer(q_coarsener.small_model)
        ref_ke = ke_time(jnp.expand_dims(ref_q, 0))
        net_ke = ke_time(jnp.expand_dims(net_q, 0))
        err = ref_ke - net_ke
        relerr = jnp.abs(err / ref_ke)
        return relerr

    def run_sequence(rng, sys_params, net):
        ref_q, ref_forcings = make_reference_traj(rng, sys_params)
        init_q = ref_q[0]
        final_net_q = get_final_net_step(net, init_q, sys_params)
        ke_relerr = get_ke_relerr(ref_q[-1], final_net_q)
        return ref_q, ref_forcings, ke_relerr

    @eqx.filter_jit
    def run_multi_sequence(rng, sys_params, net):
        rngs = jax.random.split(rng, num_candidates)
        ref_q, ref_forcing, ke_relerr = jax.vmap(functools.partial(run_sequence, net=net))(rngs, sys_params)
        return ref_q, ref_forcing, ke_relerr

    def gen_traj_func(epoch, rng_ctr, net):
        # Inactive epoch, generate no trajectories
        if epoch < start_epoch or ((epoch - start_epoch) % interval) != 0:
            return no_gen_traj_func(epoch=epoch, rng_ctr=rng_ctr, net=net)
        # Active epoch, continue with generation
        rng, rng_ctr = jax.random.split(rng_ctr, 2)
        plain_sys_params = list(itertools.islice(param_sampler, num_candidates))
        sys_params = jax.tree_util.tree_map(
            lambda *args: jnp.stack(args),
            *plain_sys_params,
        )
        ref_q, ref_forcing, ke_relerr = run_multi_sequence(rng, sys_params, net)
        # Pick top trajectories by ke_relerr
        ke_relerr = np.nan_to_num(np.asarray(ke_relerr), nan=np.inf, posinf=np.inf, neginf=0)
        idxs = np.arange(ke_relerr.shape[0])
        relerr_idx = list(zip(ke_relerr, idxs, strict=True))
        relerr_idx.sort(reverse=True)

        def valid_idx(args):
            _ke, idx = args
            return jnp.all(jnp.isfinite(ref_q[idx])) and all(jnp.all(jnp.isfinite(v[idx])) for v in ref_forcing.values())

        top_candidates = [
            idx for _err, idx in
            itertools.islice(
                filter(
                    valid_idx,
                    relerr_idx,
                ),
                num_winners
            )
        ]
        logger.info("Selected %d candidates from live generation", len(top_candidates))

        # Package selected candidates into data suitable for storage
        winners = []
        for idx in top_candidates:
            winners.append(
                SnapshotStates(
                    q=ref_q[idx],
                    q_total_forcings={
                        k: v[idx] for k, v in ref_forcing.items()
                    },
                    sys_params={k: np.asarray(v) for k, v in plain_sys_params[idx].items()},
                )
            )
        result_stats = {
            "num_winners": len(top_candidates),
            "num_candidates": num_candidates,
            "winning_ke_relerrs": np.asarray(ke_relerr).tolist(),
        }
        return winners, result_stats, rng_ctr

    def generate_network_path(net, init_q, sys_params, num_steps):
        qg_model_params = qg_utils.qg_model_to_args(processing_model)
        param_step_func = make_parameterized_stepped_model(
            nets=[net],
            net_data=[net_data],
            model_params=model_params,
            qg_model_args=qg_model_params,
            dt=dt,
        )
        states = param_step_func(
            initial_q=init_q,
            num_steps=num_steps + 1,
            subsampling=subsample,
            sys_params=sys_params,
            skip_steps=1,
        )
        return states.q

    net_num_steps = rollout_net_steps * subsample
    generate_net_path_fn = eqx.filter_jit(functools.partial(generate_network_path, num_steps=net_num_steps))

    def net_noise_gen_func(epoch, rng_ctr, net):
        # Inactive epoch, generate no trajectories
        if epoch < start_epoch or ((epoch - start_epoch) % interval) != 0:
            return no_gen_traj_func(epoch=epoch, rng_ctr=rng_ctr, net=net)
        # Active epoch, continue with generation
        # Open a basic loader on the training set
        fields = determine_required_fields(
            itertools.chain(
                net_info["input_channels"],
                net_info["output_channels"],
            )
        )
        fields.update(["rek", "beta", "delta"])
        fixed_train_path = pathlib.Path(train_path)
        train_name = fixed_train_path.parts[-3]
        assert train_name.startswith("train")
        if m := re.match(r"^train(?P<size>\d+)$", train_name):
            traj_limit = int(m.group("size"))
        else:
            traj_limit = None
        fixed_train_path = fixed_train_path.parent.parent.parent / "train" / fixed_train_path.parent.name / "data.hdf5"

        def generate_trajs():
            with SimpleQGLoader(file_path=fixed_train_path, fields=fields) as ref_loader:
                if traj_limit is not None:
                    select_trajs = min(ref_loader.num_trajs, traj_limit)
                else:
                    select_trajs = ref_loader.num_trajs
                while True:
                    traj = np_rng.integers(select_trajs).item()
                    step = np_rng.integers(low=0, high=ref_loader.num_steps - rollout_net_steps - 1).item()
                    traj_data = ref_loader.get_trajectory(traj, step, step + rollout_net_steps + 1)
                    traj_sys_params = jax.tree_map(lambda d: jnp.asarray(d[0, 0, 0, 0]), traj_data.sys_params)
                    init_q = traj_data.q[0]
                    net_q = np.asarray(generate_net_path_fn(net, init_q, traj_sys_params))
                    assert net_q.shape[0] == rollout_net_steps
                    assert net_q.shape[0] == traj_data.q.shape[0] - 1
                    if not np.all(np.isfinite(net_q)):
                        logger.warning("Dropping one trajectory for NaNs")
                        continue
                    yield SnapshotStates(
                        q=net_q,
                        q_total_forcings={
                            k: np.asarray(v[1:]) for k, v in traj_data.q_total_forcings.items()
                        },
                        sys_params={
                            k: np.asarray(v)
                            for k, v in traj_sys_params.items()
                        },
                    )
        # Return results
        result_stats = {
            "num_winners": num_candidates,
            "num_candidates": num_candidates,
        }
        return itertools.islice(generate_trajs(), num_candidates), result_stats, rng_ctr

    def generate_network_finalstep(net, init_q_batch, sys_params_batch, num_steps):
        qg_model_params = qg_utils.qg_model_to_args(processing_model)
        param_step_func = make_parameterized_stepped_model(
            nets=[net],
            net_data=[net_data],
            model_params=model_params,
            qg_model_args=qg_model_params,
            dt=dt,
        )
        ps_func = lambda iq, sp: param_step_func(
            initial_q=iq,
            num_steps=num_steps,
            subsampling=1,
            sys_params=sp,
            skip_steps=num_steps - 1,
        )
        states = jax.vmap(ps_func)(init_q_batch, sys_params_batch)
        return states.q

    net_num_steps = rollout_net_steps * subsample
    generate_net_finalstep_fn = eqx.filter_jit(functools.partial(generate_network_finalstep, num_steps=net_num_steps))

    def net_noise_gen_onestep_func(epoch, rng_ctr, net):
        # Inactive epoch, generate no trajectories
        if epoch < start_epoch or ((epoch - start_epoch) % interval) != 0:
            return no_gen_traj_func(epoch=epoch, rng_ctr=rng_ctr, net=net)
        # Active epoch, continue with generation
        # Open a basic loader on the training set
        fields = determine_required_fields(
            itertools.chain(
                net_info["input_channels"],
                net_info["output_channels"],
            )
        )
        fields.update(["rek", "beta", "delta"])
        fixed_train_path = pathlib.Path(train_path)
        train_name = fixed_train_path.parts[-3]
        assert train_name.startswith("train")
        if m := re.match(r"^train(?P<size>\d+)$", train_name):
            traj_limit = int(m.group("size"))
        else:
            traj_limit = None
        fixed_train_path = fixed_train_path.parent.parent.parent / "train" / fixed_train_path.parent.name / "data.hdf5"

        def generate_onestep_trajs():
            with SimpleQGLoader(file_path=fixed_train_path, fields=fields) as ref_loader:
                if traj_limit is not None:
                    select_trajs = min(ref_loader.num_trajs, traj_limit)
                else:
                    select_trajs = ref_loader.num_trajs
                while True:
                    BATCH_SIZE = 5
                    # Assemble batch
                    batch = []
                    for _ in range(BATCH_SIZE):
                        traj = np_rng.integers(select_trajs).item()
                        step = np_rng.integers(low=0, high=ref_loader.num_steps - rollout_net_steps).item()
                        traj_data = ref_loader.get_trajectory(traj, step, step + rollout_net_steps)
                        traj_sys_params = jax.tree_map(lambda d: jnp.asarray(d[0, 0, 0, 0]), traj_data.sys_params)
                        batch.append((traj_data, traj_sys_params))
                    traj_data, traj_sys_params = jax.tree_util.tree_map(lambda *args: jnp.stack(args, axis=0), *batch)
                    init_q = traj_data.q[:, 0]
                    net_qs = np.asarray(generate_net_finalstep_fn(net, init_q, traj_sys_params))
                    assert net_qs.shape[0] == BATCH_SIZE
                    assert net_qs.shape[1] == 1
                    # Split into batches
                    for batch_idx in range(net_qs.shape[0]):
                        net_q = net_qs[batch_idx]
                        if not np.all(np.isfinite(net_q)):
                            logger.warning("Dropping one trajectory for NaNs")
                            continue
                        yield SnapshotStates(
                            q=net_q,
                            q_total_forcings={
                                k: np.expand_dims(np.asarray(v[batch_idx, -1]), 0) for k, v in traj_data.q_total_forcings.items()
                            },
                            sys_params={
                                k: np.asarray(v)[batch_idx]
                                for k, v in traj_sys_params.items()
                            },
                        )

        # Return results
        result_stats = {
            "num_winners": num_candidates,
            "num_candidates": num_candidates,
        }
        return itertools.islice(generate_onestep_trajs(), num_candidates), result_stats, rng_ctr

    def schedule_only(epoch, rng_ctr, net):
        # Inactive epoch, generate no trajectories
        if epoch < start_epoch or ((epoch - start_epoch) % interval) != 0:
            return no_gen_traj_func(epoch=epoch, rng_ctr=rng_ctr, net=net)
        result_stats = {
            "num_winners": num_candidates,
            "num_candidates": num_candidates,
        }
        return [], result_stats, rng_ctr

    match live_gen_mode:
        case "add-hardest":
            logger.info("Live generation %d candidates, %d winners, %d steps", num_candidates, num_winners, (num_steps - num_warmup_steps) // subsample)
            logger.info("Generating samples with %s", sample_conf)
            return gen_traj_func
        case "network-noise":
            logger.info("Live generation with net noise of %d trajectories %d steps", num_candidates, rollout_net_steps)
            return net_noise_gen_func
        case "network-noise-onestep":
            logger.info("Live generation with net noise of %d using only single step %d", num_candidates, rollout_net_steps)
            return net_noise_gen_onestep_func
        case "schedule-only":
            logger.info("Live generation for schedule purposes only")
            return schedule_only
        case _:
            raise ValueError(f"invalid live generation mode {live_gen_mode}")


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
        elif m := re.match(r"^(?P<chan>q|rek|delta|beta)_\d+$", chan):
            loader_chans.add(m.group("chan"))
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
    if m := re.match(r"^(q_total_forcing|q|rek|delta|beta)_(?P<size>\d+)$", chan):
        return int(m.group("size"))
    elif m := re.match(r"^(q_scaled_forcing|q_scaled)_\d+to(?P<size>\d+)$", chan):
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
    if re.match(r"^(q|q_total_forcing)_\d+$", chan):
        return 2
    elif re.match(r"^(rek|delta|beta)_\d+$", chan):
        return 1
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
    qg_models: dict[int, pyqg_jax.qg_model.QGModel]
    scale_operator: str


def load_model_params(train_path, eval_path=None):
    if eval_path is None:
        eval_path = train_path
    train_path = pathlib.Path(train_path)
    eval_path = pathlib.Path(eval_path)
    if not train_path.exists() and train_path.is_relative_to("/scratch"):
        # Fix train data paths when loading Greene paths on Flatiron systems
        train_path = pathlib.Path(os.environ["SCRATCH"]) / train_path.relative_to(train_path.parents[-3])
    if not eval_path.exists() and eval_path.is_relative_to("/scratch"):
        # Fix train data paths when loading Greene paths on Flatiron systems
        eval_path = pathlib.Path(os.environ["SCRATCH"]) / eval_path.relative_to(eval_path.parents[-3])
    # Continue with loading params
    with h5py.File(train_path, "r") as data_file:
        coarse_op_name = data_file["params"]["coarsen_op"].asstr()[()]
    qg_models = {}
    with h5py.File(eval_path, "r") as data_file:
        # Load the big model
        qg_models["big_model"] = qg_utils.qg_model_from_param_json(data_file["params"]["big_model"].asstr()[()])
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
        ).astype(jnp.float32)
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
        ).astype(jnp.float32)
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
    # System parameters
    elif m := re.match(r"^(?P<chan>rek|delta|beta)_\d+$", channel):
        data = batch.sys_params[m.group("chan")]
        size = determine_channel_size(channel)
        assert data.shape[-3:] == (determine_channel_layers(channel), 1, 1)
        tile_shape = ((1, ) * (data.ndim - 2)) + (size, size)
        return jnp.tile(data, tile_shape)
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
        noise_var = jnp.asarray(noise_var).astype(chan.dtype)
        if noise_var.ndim > 0:
            noise_var = jnp.expand_dims(noise_var, (-1, -2))
        noise_mask = jnp.sqrt(noise_var) * jax.random.normal(key=key, shape=chan.shape, dtype=chan.dtype)
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
        assert channel not in {"rek", "beta", "delta"} or data.shape[-1] == processing_size
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
            assert channel not in {"rek", "beta", "delta"} or data.shape[-1] == processing_size
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

    def do_batch(batches, samples_per_batch, state, rng_ctr, clean_vs_noise_spec_counts):
        input_chunks = []
        target_chunks = []
        batch_sizes = {leaf.shape[0] for leaf in jax.tree_util.tree_leaves(batches)}
        if len(batch_sizes) != 1:
            raise ValueError(f"inconsistent batch sizes {batch_sizes}")
        batch_size = batch_sizes.pop()

        # Special processing for the first chunk (gaussian noise, if needed)
        batch = batches[0]
        target_chunk1 = make_chunk_from_batch(
            channels=output_channels,
            batch=batch,
            model_params=model_params,
            processing_size=output_size,
        )
        if noise_spec:
            # Need to do noise processing and selection
            rng1, rng2, rng_ctr = jax.random.split(rng_ctr, 3)
            n_clean, n_noise = clean_vs_noise_spec_counts
            prob_clean = n_clean / (n_clean + n_noise)
            num_clean = jnp.count_nonzero(jax.random.uniform(rng1, shape=(batch_size,), dtype=jnp.float32) <= prob_clean)
            input_chunk_noisy1 = make_chunk_from_batch(
                channels=input_channels,
                batch=batch,
                model_params=model_params,
                processing_size=processing_size,
                noise_spec=noise_spec,
                key=rng2,
            )
            input_chunk_clean1 = make_chunk_from_batch(
                channels=input_channels,
                batch=batch,
                model_params=model_params,
                processing_size=processing_size,
            )
            # Pick how many to apply noise to
            indices = jnp.arange(batch_size, dtype=jnp.uint32)
            select_indices = (indices >= num_clean).astype(jnp.uint8)
            input_chunk1 = jnp.stack([input_chunk_clean1, input_chunk_noisy1], axis=0)[select_indices, indices]
        else:
            input_chunk1 = make_chunk_from_batch(
                channels=input_channels,
                batch=batch,
                model_params=model_params,
                processing_size=processing_size,
            )
        # Store chunks
        input_chunks.append(input_chunk1)
        target_chunks.append(target_chunk1)

        # Continue with remaining batches
        for batch in itertools.islice(batches, 1, None):
            if batch is None:
                continue
            input_chunks.append(
                make_chunk_from_batch(
                    channels=input_channels,
                    batch=batch,
                    model_params=model_params,
                    processing_size=processing_size,
                )
            )
            target_chunks.append(
                make_chunk_from_batch(
                    channels=output_channels,
                    batch=batch,
                    model_params=model_params,
                    processing_size=output_size,
                )
            )
        # Do selection slicing
        assert len(input_chunks) == len(target_chunks)
        indices = jnp.arange(batch_size, dtype=jnp.uint32)
        select_indices = jnp.sum((jnp.expand_dims(indices, 0) >= jnp.expand_dims(jnp.cumsum(samples_per_batch, axis=0), -1)).astype(jnp.uint8), axis=0)
        select_indices = jnp.minimum(select_indices, jnp.uint8(len(input_chunks) - 1))
        input_chunk = jnp.stack(input_chunks, axis=0)[select_indices, indices]
        target_chunk = jnp.stack(target_chunks, axis=0)[select_indices, indices]
        # Compute losses
        loss, grads = eqx.filter_value_and_grad(batch_loss)(state.net, input_chunk, target_chunk)
        # Update parameters
        out_state = state.apply_updates(grads)
        return out_state, loss, rng_ctr

    return do_batch


def do_epoch(train_state, batch_iter, batch_fn, rng_ctr, clean_vs_noise_spec_counts, logger=None):
    n_clean, n_noisy = clean_vs_noise_spec_counts
    logger.info("Epoch with virtual noise samples clean=%d, noisy=%d", n_clean, n_noisy)
    if logger is None:
        logger = logging.getLogger("train_epoch")
    epoch_start = time.perf_counter()
    losses = []
    for batch in batch_iter:
        batches, samples_per_batch = batch
        train_state, batch_loss, rng_ctr = batch_fn(batches, samples_per_batch, train_state, rng_ctr, (jnp.uint32(n_clean), jnp.uint32(n_noisy)))
        batches = None
        samples_per_batch = None
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


def init_network(architecture, lr, rng, input_channels, output_channels, processing_size, train_path, optim_type, num_epochs, batches_per_epoch, end_lr, schedule_type, coarse_op_name, arch_args={}):

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
        **arch_args,
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
        case "ross22":
            sched_args = {
                "type": "ross22",
                "args": {
                    "init_value": lr,
                    "boundaries_and_scales": {
                        step: 0.1
                        for step in [math.floor(s * steps_per_epoch * num_epochs) for s in (1/2, 3/4, 7/8)]
                    },
                },
            }
            schedule = optax.piecewise_constant_schedule(**sched_args["args"])
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
            optax.identity() if schedule_type in {"ross22"} else optax.clip(1.0),
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


def load_network_continue(weight_file, old_state, old_net_info):
    weight_file = pathlib.Path(weight_file)
    # Load network info
    with open(weight_file.parent / "network_info.json", "r", encoding="utf8") as net_info_file:
        net_info = json.load(net_info_file)
    net = eqx.tree_deserialise_leaves(weight_file, old_state.net)
    # Replace old net with new net
    state = old_state
    state.net = net
    # Compare net_info contents
    if net_info != old_net_info:
        raise ValueError("network info does not match (check command line arguments)")
    return state, net_info


def main(args):
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
    if args.live_gen_base_data is None:
        live_gen_path = train_path
    else:
        live_gen_path = (pathlib.Path(args.live_gen_base_data) / "shuffled.hdf5").resolve()
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
        arch_args={
            "zero_mean": args.network_zero_mean,
        },
    )
    if args.net_weight_continue is not None:
        logger.info("CONTINUING NETWORK: %s", args.net_weight_continue)
        # Load network from file, wrap in train state
        state, network_info = load_network_continue(
            args.net_weight_continue,
            state,
            network_info,
        )
        logger.info("Loaded trained network weights")
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

    # Live generation function
    live_gen_func = make_live_gen_func(
        start_epoch=args.live_gen_start_epoch,
        interval=args.live_gen_interval,
        sample_conf=args.live_gen_sampler,
        num_candidates=args.live_gen_candidates,
        num_winners=args.live_gen_winners,
        np_rng=np_rng,
        logger=logger.getChild("live-gen"),
        model_params=model_params,
        net_info=network_info,
        live_gen_mode=args.live_gen_mode,
        train_path=live_gen_path,
        rollout_net_steps=args.live_gen_net_steps,
        **live_sample_get_args(
            train_path=live_gen_path,
            required_fields=required_fields,
        ),
    )

    # Open data files
    with contextlib.ExitStack() as train_context:
        # Open data files
        train_loader = train_context.enter_context(
            AggregateLoader(
                loaders=[
                    ThreadedPreShuffledSnapshotLoader(
                        file_path=train_path,
                        batch_size=args.batch_size,
                        chunk_size=args.loader_chunk_size,
                        buffer_size=10,
                        seed=np_rng.integers(2**32).item(),
                        base_logger=logger.getChild("train_loader"),
                        fields=required_fields,
                    ),
                    FillableDataLoader(
                        batch_size=args.batch_size,
                        fields=required_fields,
                        seed=np_rng.integers(2**32).item(),
                    ),
                    FillableDataLoader(
                        batch_size=args.batch_size,
                        fields=required_fields,
                        seed=np_rng.integers(2**32).item(),
                    ),
                ],
                batch_size=args.batch_size,
                seed=np_rng.integers(2**32).item(),
            )
        )
        val_loader = train_context.enter_context(
            SimpleQGLoader(
                file_path=val_path,
                fields=required_fields,
            )
        )

        num_clean_samples = train_loader.num_samples()
        num_dirty_samples = 0

        # Training functions
        train_batch_fn = eqx.filter_jit(
            make_batch_computer(
                input_channels=input_channels,
                output_channels=output_channels,
                model_params=model_params,
                processing_size=processing_size,
                noise_spec=noise_spec,
            ),
            donate="all",
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
                    clean_vs_noise_spec_counts=(num_clean_samples, num_dirty_samples),
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

            # Add live-generated trajectories, if required
            logger.info("Starting live trajectory generation")
            fillable_loader = train_loader.loaders[-1]
            assert isinstance(fillable_loader, FillableDataLoader)
            new_live_trajs, new_traj_info, rng_ctr = live_gen_func(
                epoch=epoch,
                rng_ctr=rng_ctr,
                net=state.net,
            )
            num_dirty_samples += new_traj_info["num_winners"]
            added_trajs = 0
            for live_traj in new_live_trajs:
                fillable_loader.add_data(live_traj)
                added_trajs += 1
            logger.info("Added %d live-generated trajectories", added_trajs)
            new_live_trajs = None

            epoch_reports.append(
                {
                    "epoch": epoch,
                    "train_stats": epoch_stats,
                    "val_stats": val_stat_report,
                    "saved_names": saved_names,
                    "live_traj": new_traj_info,
                }
            )
            with utils.rename_save_file(out_dir / "train_report.json", "w", encoding="utf8") as train_report_file:
                json.dump(epoch_reports, train_report_file)

            logger.info("Finished epoch %d", epoch)

    # End of training loop
    logger.info("Finished training")


if __name__ == "__main__":
    main(parser.parse_args())
