import argparse
import pathlib
import dataclasses
import utils
import logging
import contextlib
import math
import operator
import sys
import jax
import jax.numpy as jnp
import numpy as np
import h5py
from systems.qg import coarsen, compute_stats
from systems.qg import utils as qg_utils
from systems.ns import config as ns_config, compute_stats as ns_stats
import xarray
import haiku as hk
import gin
import jax_cfd
import jax_cfd.ml, jax_cfd.base, jax_cfd.spectral, jax_cfd.base.grids, jax_cfd.data
import jax_utils
import powerpax as ppx
import pyqg_jax
import itertools
import collections
import re
import functools
import random
import asyncio


CONFIG_VARS = {
    "eddy": {
        "rek": 5.787e-7,
        "delta": 0.25,
        "beta": 1.5e-11,

    },
    "jet": {
        "rek": 7e-08,
        "delta": 0.1,
        "beta": 1e-11,
    },
}


parser = argparse.ArgumentParser(description="Generate data for a variety of systems")
parser.add_argument("out_dir", type=str, help="Directory to store output (created if non-existing)")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
subparsers = parser.add_subparsers(help="Choice of system to generate", dest="system", required=True)

# QG options
parser_qg = subparsers.add_parser("qg", help="Generate training data like PyQG")
parser_qg.add_argument("seed", type=int, help="RNG seed, must be unique for unique trajectory")
parser_qg.add_argument("--dt", type=float, default=3600.0, help="Time step size")
parser_qg.add_argument("--twarmup", type=float, default=0.0, help="Time at which we should start saving steps")
parser_qg.add_argument("--tmax", type=float, default=311040000.0, help="End time for the model")
parser_qg.add_argument("--big_size", type=int, default=256, help="Scale of large model")
parser_qg.add_argument("--small_size", type=int, nargs="+", default=[128, 96, 64], help="Scale of small model")
parser_qg.add_argument("--subsample", type=int, default=1, help="Stride used to select how many generated steps to keep")
parser_qg.add_argument("--num_trajs", type=int, default=1, help="Number of trajectories to generate")
parser_qg.add_argument("--config", type=str, default="eddy", help="Eddy or jet configuration")
parser_qg.add_argument("--coarse_op", type=str, default="op1", help="Which coarsening operators to apply", choices=sorted(coarsen.COARSEN_OPERATORS.keys()))
parser_qg.add_argument("--max_gen_tries", type=int, default=25, help="Number of retries to generate a trajectory")
parser_qg.add_argument("--traj_slice", type=str, default=None, help="Which subset of trajectories to generate")
parser_qg.add_argument("--precision", type=str, default="single", choices=["single", "double"], help="Precision to use when generating trajectories")
parser_qg.add_argument("--store_as_single", action="store_true", help="Cast trajectory data to single precision before storing")

# Combine QG slice options
parser_combine_qg_slice = subparsers.add_parser("combine_qg_slice", help="Combine slices of QG data")

# NS options
parser_ns = subparsers.add_parser("ns", help="Generate training snapshots for NS learned correction")
parser_ns.add_argument("seed", type=int, help="RNG seed, must be unique for unique trajectories")
parser_ns.add_argument("--twarmup", type=float, default=40.0, help="Time at which we should start saving steps")
parser_ns.add_argument("--tmax", type=float, default=70.0, help="End time for the model")
parser_ns.add_argument("--tstep", type=float, default=0.006574203376652748, help="Stride at which snapshots should be stored")
parser_ns.add_argument("--big_size", type=int, default=2048, help="Scale of large model")
parser_ns.add_argument("--small_size", type=int, nargs="+", default=[128, 64], help="Scale of small models")
parser_ns.add_argument("--num_trajs", type=int, default=1, help="Number of trajectories to generate")
parser_ns.add_argument("--traj_slice", type=str, default=None, help="Which subset of trajectories to generate")
parser_ns.add_argument("--store_as_single", action="store_true", help="Cast trajectory data to single precision before storing")
parser_ns.add_argument("--viscosity", type=float, default=0.001, help="Setting for model viscosity")
parser_ns.add_argument("--max_velocity", type=float, default=7.0, help="Setting for maximum velocity")
parser_ns.add_argument("--cfl_factor", type=float, default=0.5, help="Setting for CFL factor when picking time step")
parser_ns.add_argument("--simultaneous_trajs", type=int, default=2, help="Number of trajectories to generate at once")
parser_ns.add_argument("--peak_wavenumber", type=int, default=4, help="Peak wavenumber for initial condition")

# Combine NS slice options
parser_combine_ns_slice = subparsers.add_parser("combine_ns_slice", help="Combine slices of NS data")

# Shuffle NS data options
parser_shuffle_ns_data = subparsers.add_parser("shuffle_ns_data", help="Pre-shuffle a combined NS dataset")
parser_shuffle_ns_data.add_argument("--seed", type=int, default=None, help="Seed used for shuffling data (default: auto)")
parser_shuffle_ns_data.add_argument("--num_workers", type=int, default=32, help="Number of async workers")
parser_shuffle_ns_data.add_argument("--chunk_size", type=int, default=1000, help="Number of samples per load batch")


def make_coarsen_to_size(coarse_op, big_model, small_nx):

    def do_coarsening(var):
        assert var.shape[-2:] == (big_model.ny, big_model.nx)
        if big_model.nx == small_nx:
            # No coarsening needed
            res = coarsen.NoOpCoarsener(big_model=big_model, small_nx=small_nx).coarsen(var)
        else:
            res = coarse_op(big_model=big_model, small_nx=small_nx).coarsen(var)

        assert res.shape[-2:] == (small_nx, small_nx)
        return res

    return do_coarsening


def as_chunk_host_iter(source_arr, dtype, chunk_size=1000):
    arr_size = operator.index(source_arr.shape[0])
    num_batches, remainder = divmod(arr_size, chunk_size)
    # Yield chunks
    for i in range(num_batches):
        start = i * chunk_size
        end = start + chunk_size
        yield np.asarray(source_arr[start:end]).astype(dtype)
    # Yield remainder
    if remainder != 0:
        yield np.asarray(source_arr[-remainder:]).astype(dtype)


def rng_yielder(rng_ctr):

    while True:
        rng, rng_ctr = jax.random.split(rng_ctr, 2)
        yield rng


def xarray_concatenate_samples(path, new_dataset):
    try:
        base_dataset = xarray.load_dataset(path)
    except FileNotFoundError:
        base_dataset = None
    if base_dataset is not None:
        new_dataset = xarray.concat((base_dataset, new_dataset), "sample")
    with utils.safe_replace(path):
        new_dataset.to_netcdf(path)


@jax_utils.register_pytree_dataclass
@dataclasses.dataclass
class CoarseTrajResult:
    q: jax.Array
    t: jax.Array
    tc: jax.Array
    ablevel: jax.Array
    dqhdt: jax.Array
    q_total_forcings: dict[int, jax.Array]


def array_cast_single(leaf):
    if leaf.dtype == jnp.dtype(jnp.float64):
        return leaf.astype(jnp.float32)
    elif leaf.dtype == jnp.dtype(jnp.complex128):
        return leaf.astype(jnp.complex64)
    else:
        return leaf

def array_cast_identity(leaf):
    return leaf


def make_generate_coarse_traj(big_model, small_sizes, coarse_op_cls, num_warmup_steps, num_steps, subsample):
    if num_warmup_steps >= num_steps:
        raise ValueError(f"warmup steps {num_warmup_steps} larger than total steps {num_steps}")
    coarse_ops = {}
    base_big_model = big_model
    for size in set(small_sizes):
        size = operator.index(size)
        if size > big_model.model.nx:
            raise ValueError(
                f"invalid coarse size {size}, larger than base size {big_model.nx}"
            )
        elif size < big_model.model.nx:
            op = coarse_op_cls(
                big_model=big_model.model,
                small_nx=size,
            )
        else:
            op = coarsen.NoOpCoarsener(
                big_model=big_model.model,
                small_nx=size,
            )
        coarse_ops[size] = op
    size_main_states = max(map(operator.index, small_sizes))

    def make_traj(big_initial_step, sys_params):

        model_args = qg_utils.qg_model_to_args(base_big_model.model)
        model_args.update(sys_params)
        step_big_model = pyqg_jax.steppers.SteppedModel(
            model=pyqg_jax.qg_model.QGModel(**model_args),
            stepper=pyqg_jax.steppers.AB3Stepper(dt=base_big_model.stepper.dt),
        )

        def _step_until_warmup(carry, _x):
            prev_big_state = carry
            next_big_state = step_big_model.step_model(prev_big_state)
            return next_big_state, None

        def _step_forward(carry, x):
            prev_big_state = carry
            # Produce new "main size" state for output
            prev_small_q = coarse_ops[size_main_states].coarsen(prev_big_state.state.q)
            prev_big_dqhdt = step_big_model.get_full_state(prev_big_state).dqhdt
            prev_small_dqhdt = coarse_ops[size_main_states].coarsen(prev_big_dqhdt)
            # Step the large model forward
            next_big_state = step_big_model.step_model(prev_big_state)
            return next_big_state, CoarseTrajResult(
                q=prev_small_q,
                t=prev_big_state.t,
                tc=prev_big_state.tc,
                ablevel=prev_big_state._ablevel,
                dqhdt=prev_small_dqhdt,
                q_total_forcings={
                    k: op.compute_q_total_forcing(prev_big_state.state.q)
                    for k, op in coarse_ops.items()
                },
            )

        if num_warmup_steps > 0:
            big_warmed_up_step, _ys = jax.lax.scan(
                _step_until_warmup,
                big_initial_step,
                None,
                length=num_warmup_steps,
            )
        else:
            big_warmed_up_step = big_initial_step

        _carry, results = jax_utils.strided_scan(
            _step_forward,
            big_warmed_up_step,
            None,
            length=num_steps - num_warmup_steps,
            stride=subsample,
        )
        return results

    return make_traj


def make_parameter_sampler(args_config, np_rng):
    if args_config == "eddy":
        return itertools.repeat(CONFIG_VARS["eddy"])
    elif args_config == "jet":
        return itertools.repeat(CONFIG_VARS["jet"])
    elif args_config.startswith("rand-eddy-to-jet-"):
        assert CONFIG_VARS["eddy"].keys() == CONFIG_VARS["jet"].keys()
        frac = float(args_config[len("rand-eddy-to-jet-"):])
        assert 0 <= frac <= 1

        def sample_rand_eddy_jet(np_rng, frac):
            while True:
                # We interpolate three variables:
                # 1. rek
                # 2. beta
                # 3. delta
                # At high rek factors, beta and delta need to be closer together
                # So we decay how far apart they are permitted to be as the factor rises
                rek_factor = np_rng.random() * frac
                delta_beta_max_radius = np.clip(1 / (1 + np.exp(12 * rek_factor**2 - 6)), 0, 1)
                delta_factor = np_rng.random() * frac
                min_beta = np.clip(delta_factor - delta_beta_max_radius, 0, frac)
                max_beta = np.clip(delta_factor + delta_beta_max_radius, 0, frac)
                beta_factor = min_beta + (max_beta - min_beta) * np_rng.random()
                yield {
                    var: float((1 - factor) * CONFIG_VARS["eddy"][var] + factor * CONFIG_VARS["jet"][var])
                    for var, factor in [("rek", rek_factor), ("delta", delta_factor), ("beta", beta_factor)]
                }

        return sample_rand_eddy_jet(np_rng, frac)
    elif args_config.startswith("rand-1d-eddy-to-jet-"):
        assert CONFIG_VARS["eddy"].keys() == CONFIG_VARS["jet"].keys()
        frac = float(args_config[len("rand-1d-eddy-to-jet-"):])
        assert 0 <= frac <= 1

        def sample_rand_1d_eddy_jet(np_rng, frac):
            while True:
                res = {}
                rand = np_rng.random() * frac
                for k, eddy_endpoint in CONFIG_VARS["eddy"].items():
                    jet_endpoint = CONFIG_VARS["jet"][k]
                    res[k] = float((1 - rand) * eddy_endpoint + rand * jet_endpoint)
                yield res

        return sample_rand_1d_eddy_jet(np_rng, frac)
    else:
        raise ValueError(f"invalid configuration type {args_config}")


def combine_qg_slice(out_dir, args, base_logger):
    out_dir = pathlib.Path(out_dir)
    logger = base_logger.getChild("combine_qg_slice")
    # Locate data slice files
    slice_files = list(out_dir.glob("data-slice*.hdf5"))
    if not slice_files:
        logger.error("No slice files found")
        raise ValueError("No slice files found")
    logger.info("Located %d dataset slice files", len(slice_files))
    with h5py.File(out_dir / "data.hdf5", "w", libver="latest") as out_file:
        for slice_file in slice_files:
            logger.info("Processing slice file %s", slice_file)
            with h5py.File(slice_file, "r") as in_file:
                # Copy parameters if not present
                if "params" not in out_file:
                    in_file.copy(source=in_file["/params"], dest=out_file["/"], shallow=False, expand_soft=True, expand_external=True, expand_refs=True, without_attrs=False)
                # Copy trajectories
                if "trajs" not in out_file:
                    out_file.create_group("trajs")
                for k, v in in_file["trajs"].items():
                    logger.info("Copying trajectory data %s", k)
                    in_file.copy(source=v, dest=out_file["/trajs"], shallow=False, expand_soft=True, expand_external=True, expand_refs=True, without_attrs=False)
        # Need to compute statistics over the new dataset
        # Determine small_sizes
        logger.info("Recomputing statistics")
        small_sizes = set()
        op_name = out_file["params"]["coarsen_op"].asstr()[()]
        logger.info("Coarsening operator %s", op_name)
        for k in out_file["params"].keys():
            if m := re.match(r"^small_model_(?P<size>\d+)$", k):
                small_sizes.add(int(m.group("size")))
        main_small_size = max(small_sizes)
        logger.info("Main small size %d", main_small_size)
        coarse_cls = coarsen.COARSEN_OPERATORS[op_name]
        # Prep stats computers
        forcing_stats = {sz: compute_stats.QGStatAccumulator() for sz in small_sizes}
        q_stats = {sz: compute_stats.QGStatAccumulator() for sz in small_sizes}
        big_model = qg_utils.qg_model_from_param_json(out_file["params"]["big_model"].asstr()[()])
        # Prep coarsen fns
        coarsen_fns = {
            size: jax.jit(
                jax.vmap(
                    make_coarsen_to_size(
                        coarse_op=coarse_cls,
                        big_model=coarse_cls(big_model=big_model, small_nx=main_small_size).small_model if main_small_size != big_model.nx else big_model,
                        small_nx=size,
                    ),
                ),
            )
            for size in small_sizes
        }
        # Loop over all q and q_total_forcing values and compute stats
        logger.info("Computing statistics for variables")
        for k in out_file["trajs"].keys():
            if m := re.match(r"^traj\d+_q_total_forcing_(?P<size>\d+)$", k):
                logger.info("Processing forcing %s", k)
                forcing_size = int(m.group("size"))
                with contextlib.closing(as_chunk_host_iter(out_file["trajs"][k], dtype=np.float64, chunk_size=1000)) as batch_iter:
                    for batch in batch_iter:
                        forcing_stats[forcing_size].add_batch(batch)
            elif re.match(r"^traj\d+_q$", k):
                logger.info("Processing q %s", k)
                # The q values require their own coarsening steps
                with contextlib.closing(as_chunk_host_iter(out_file["trajs"][k], dtype=np.float64, chunk_size=1000)) as batch_iter:
                    for batch in batch_iter:
                        for size in small_sizes:
                            q_stats[size].add_batch(coarsen_fns[size](batch))
        # Store statistics
        logger.info("Finalizing statistics")
        stats_group = out_file.create_group("stats")
        forcing_stats_group = stats_group.create_group("q_total_forcing")
        for size in small_sizes:
            stats = forcing_stats[size].finalize()
            group = forcing_stats_group.create_group(f"{size}")
            group.create_dataset("mean", data=stats.mean)
            group.create_dataset("var", data=stats.var)
            group.create_dataset("min", data=stats.min)
            group.create_dataset("max", data=stats.max)
        q_stats_group = stats_group.create_group("q")
        for size in small_sizes:
            stats = q_stats[size].finalize()
            group = q_stats_group.create_group(f"{size}")
            group.create_dataset("mean", data=stats.mean)
            group.create_dataset("var", data=stats.var)
            group.create_dataset("min", data=stats.min)
            group.create_dataset("max", data=stats.max)
        logger.info("Finished storing statistics")
    logger.info("Finished merging data")


def gen_qg(out_dir, args, base_logger):
    out_dir = pathlib.Path(out_dir)
    logger = base_logger.getChild("qg")
    logger.info("Generating trajectory for QG with seed %d", args.seed)
    # Initialize model and locate coarsening class
    array_caster = jax.jit(array_cast_single if args.store_as_single else array_cast_identity)
    small_sizes = sorted(set(map(operator.index, args.small_size)))
    main_small_size = max(map(operator.index, small_sizes))
    big_model = pyqg_jax.steppers.SteppedModel(
        model=pyqg_jax.qg_model.QGModel(
            nx=args.big_size,
            ny=args.big_size,
            precision=pyqg_jax.state.Precision[args.precision.upper()],
        ),
        stepper=pyqg_jax.steppers.AB3Stepper(dt=args.dt),
    )
    num_warmup_steps = math.ceil(args.twarmup / args.dt)
    num_steps = math.ceil(args.tmax / args.dt)
    if num_warmup_steps >= num_steps:
        logger.error("warmup steps %d larger than total steps %d", num_warmup_steps, num_steps)
    initial_state_fn = jax.jit(big_model.create_initial_state)
    op_name = args.coarse_op
    coarse_cls = coarsen.COARSEN_OPERATORS[op_name]
    # Set up data generator
    rng_ctr = jax.random.PRNGKey(seed=args.seed)
    np_rng = np.random.default_rng(args.seed)
    # Create trajectory generation function
    traj_gen_func = jax.jit(
        make_generate_coarse_traj(
            big_model=big_model,
            small_sizes=small_sizes,
            coarse_op_cls=coarse_cls,
            num_warmup_steps=num_warmup_steps,
            num_steps=num_steps,
            subsample=args.subsample,
        )
    )
    coarsen_fns = {
        size: jax.jit(
            jax.vmap(
                make_coarsen_to_size(
                    coarse_op=coarse_cls,
                    big_model=coarse_cls(big_model=big_model.model, small_nx=main_small_size).small_model if main_small_size != big_model.model.nx else big_model.model,
                    small_nx=size,
                ),
            ),
        )
        for size in small_sizes
    }
    # Do computations
    logger.info("Generating %d trajectories with %d steps (subsampled by a factor of %d to %d steps)", args.num_trajs, num_steps - num_warmup_steps, args.subsample, (num_steps - num_warmup_steps) // args.subsample)
    # Create directory for this operator
    op_directory = out_dir / op_name
    op_directory.mkdir(exist_ok=True)
    # Prep stats computers
    forcing_stats = {sz: compute_stats.QGStatAccumulator() for sz in small_sizes}
    q_stats = {sz: compute_stats.QGStatAccumulator() for sz in small_sizes}
    if args.traj_slice is None:
        out_file_name = "data.hdf5"
        traj_slice_start = None
        traj_slice_end = None
    else:
        slice_underscore = args.traj_slice.replace(":", "_")
        out_file_name = f"data-slice{slice_underscore}.hdf5"
        slice_parts = [int(p.strip()) if p.strip() else None for p in args.traj_slice.split(":")]
        if len(slice_parts) != 2:
            raise ValueError(f"invalid slice spec!")
        traj_slice_start = slice_parts[0]
        traj_slice_end = slice_parts[1]
    with h5py.File(op_directory / out_file_name, "w", libver="latest") as out_file:
        # Store model parameters in each file
        param_group = out_file.create_group("params")
        param_group.create_dataset("big_model", data=qg_utils.qg_model_param_json(big_model.model))
        for size in set(small_sizes):
            size = operator.index(size)
            size_coarse_cls = coarse_cls if size != args.big_size else coarsen.NoOpCoarsener
            param_group.create_dataset(f"small_model_{size}", data=qg_utils.qg_model_param_json(size_coarse_cls(big_model=big_model.model, small_nx=size).small_model))
        param_group["small_model"] = h5py.SoftLink(f"/params/small_model_{main_small_size}")
        param_group.create_dataset("dt", data=args.dt)
        param_group.create_dataset("coarsen_op", data=op_name)
        param_group.create_dataset("subsample", data=args.subsample)
        out_file.flush()
        # Generate trajectories
        compound_dtype = None
        out_file.create_group("trajs")
        for traj_num, sys_params, rng_ctr in itertools.islice(
                zip(range(args.num_trajs), make_parameter_sampler(args.config, np_rng), rng_yielder(rng_ctr)),
                traj_slice_start,
                traj_slice_end,
        ):
            logger.info("Starting trajectory %d", traj_num)
            for generation_trial in range(args.max_gen_tries):
                logger.info("Generation attempt %d of %d", generation_trial + 1, args.max_gen_tries)
                rng, rng_ctr = jax.random.split(rng_ctr, 2)
                result = traj_gen_func(initial_state_fn(rng), sys_params)
                if jnp.all(jnp.isfinite(result.q)) and jnp.all(jnp.isfinite(result.dqhdt)):
                    # Successfully generated trajectory
                    break
                logger.warning("Generation attempt %d produced NaN", generation_trial + 1)
            else:
                logger.error("Exhausted all %d attempts for trajectory %d, failing", args.max_gen_tries, traj_num)
                raise RuntimeError(f"Exhausted all attempts when generating trajectory {traj_num}")
            logger.info("Finished generating trajectory %d", traj_num)
            if traj_num == 0:
                # First trajectory, store t, tc, ablevel
                out_file["trajs"].create_dataset("t", data=np.asarray(result.t))
                out_file["trajs"].create_dataset("tc", data=np.asarray(result.tc))
                out_file["trajs"].create_dataset("ablevel", data=np.asarray(result.ablevel))
                out_file.flush()
            # Store q and dqhdt
            out_file["trajs"].create_dataset(f"traj{traj_num:05d}_q", data=np.asarray(array_caster(result.q)))
            out_file.flush()
            out_file["trajs"].create_dataset(f"traj{traj_num:05d}_dqhdt", data=np.asarray(array_caster(result.dqhdt)))
            out_file.flush()
            # Store forcings
            for size in small_sizes:
                forcing_dataset = out_file["trajs"].create_dataset(f"traj{traj_num:05d}_q_total_forcing_{size}", data=np.asarray(array_caster(result.q_total_forcings[size])))
                out_file.flush()
            out_file["trajs"][f"traj{traj_num:05d}_q_total_forcing"] = h5py.SoftLink(f"/trajs/traj{traj_num:05d}_q_total_forcing_{main_small_size}")
            out_file.flush()
            # Store trajectory system parameters
            traj_sys_param_group = out_file["trajs"].create_group(f"traj{traj_num:05d}_sysparams")
            for param_k, param_v in sys_params.items():
                traj_sys_param_group.create_dataset(param_k, data=param_v)
            out_file.flush()
            logger.info("Finished storing trajectory %d", traj_num)
            # Keep only q and forcings and discard the rest
            logger.info("Updating statistics for trajectory %d", traj_num)
            traj_q = result.q
            traj_forcings = result.q_total_forcings
            result = None
            # Update statistics
            #   First, the forcings
            for size in small_sizes:
                with contextlib.closing(as_chunk_host_iter(traj_forcings[size], dtype=np.float64, chunk_size=1000)) as batch_iter:
                    for batch in batch_iter:
                        forcing_stats[size].add_batch(batch)
                        batch = None
                batch_iter = None
            traj_forcings = None
            #   Next, the q values which require their own coarsening steps
            for size in small_sizes:
                with contextlib.closing(as_chunk_host_iter(coarsen_fns[size](traj_q), dtype=np.float64, chunk_size=1000)) as batch_iter:
                    for batch in batch_iter:
                        q_stats[size].add_batch(batch)
                        batch = None
                batch_iter = None
            traj_q = None
            logger.info("Finished statistics for trajectory %d", traj_num)
            # Finished processing this trajectory
        # Store out statistics
        logger.info("Finalizing statistics")
        stats_group = out_file.create_group("stats")
        forcing_stats_group = stats_group.create_group("q_total_forcing")
        for size in small_sizes:
            stats = forcing_stats[size].finalize()
            group = forcing_stats_group.create_group(f"{size}")
            group.create_dataset("mean", data=stats.mean)
            group.create_dataset("var", data=stats.var)
            group.create_dataset("min", data=stats.min)
            group.create_dataset("max", data=stats.max)
        q_stats_group = stats_group.create_group("q")
        for size in small_sizes:
            stats = q_stats[size].finalize()
            group = q_stats_group.create_group(f"{size}")
            group.create_dataset("mean", data=stats.mean)
            group.create_dataset("var", data=stats.var)
            group.create_dataset("min", data=stats.min)
            group.create_dataset("max", data=stats.max)
        logger.info("Finished storing statistics")


def gen_ns(out_dir, args, base_logger):
    out_dir = pathlib.Path(out_dir)
    logger = base_logger.getChild("ns")
    logger.info("Generating trajectory for Navier-Stokes with seed %d", args.seed)
    # Initialize model and locate coarsening class
    array_caster = jax.jit(array_cast_single if args.store_as_single else array_cast_identity)
    full_config_str = ns_config.make_generation_config(viscosity=args.viscosity)
    gin.parse_config(full_config_str)
    # Set up RNG
    rng_ctr = jax.random.key(args.seed)
    np_rng = np.random.default_rng(args.seed)
    # Construct grids and model classes
    big_grid = ns_config.make_grid(size=args.big_size)
    small_grids = {s: ns_config.make_grid(size=s) for s in args.small_size}
    physics_specs = jax_cfd.ml.physics_specifications.get_physics_specs()
    dt = jax_cfd.base.equations.stable_time_step(args.max_velocity, args.cfl_factor, args.viscosity, big_grid)
    stable_steps = {s: jax_cfd.base.equations.stable_time_step(args.max_velocity, args.cfl_factor, args.viscosity, sg) for s, sg in small_grids.items()}
    logger.info("Generating using dt=%f", dt)
    big_model_cls = jax_cfd.ml.model_builder.get_model_cls(big_grid, dt, physics_specs)
    small_model_cls = {s: jax_cfd.ml.model_builder.get_model_cls(g, dt, physics_specs) for s, g in small_grids.items()}
    # Compute time step counts
    total_steps = math.ceil(args.tmax / dt)
    warmup_steps = math.ceil(args.twarmup / dt)
    stride = max(round(args.tstep / dt), 1)
    logger.info("Computing %d total steps, skipping %d for warmup and subsampling by factor %d", total_steps, warmup_steps, stride)
    # Determine output paths
    if args.traj_slice is None:
        out_file_paths = {s: (out_dir / f"data-sz{s}.nc") for s in args.small_size}
        traj_slice_start = 0
        traj_slice_end = args.num_trajs
    else:
        slice_underscore = args.traj_slice.replace(":", "_")
        out_file_paths = {s: (out_dir / f"data-slice{slice_underscore}-sz{s}.nc") for s in args.small_size}
        slice_parts = [int(p.strip()) if p.strip() else None for p in args.traj_slice.split(":")]
        if len(slice_parts) != 2:
            raise ValueError(f"invalid slice spec!")
        traj_slice_start = slice_parts[0]
        traj_slice_end = slice_parts[1]

    # Set up generation functions
    def downsample(x, source_grid, destination_grid):
        return jax_cfd.base.resize.downsample_staggered_velocity(
            source_grid=source_grid,
            destination_grid=destination_grid,
            velocity=x,
        )

    def step_fn(carry, xs, big_model, big_grid, small_models, small_grids, dt):
        assert xs is None

        t, curr_big_step = carry
        next_big_step = big_model.advance(curr_big_step)
        # Compute correction values
        outputs = {}
        for size in small_models.keys():
            small_model = small_models[size]
            small_grid = small_grids[size]
            downsampler = functools.partial(
                downsample,
                source_grid=big_grid,
                destination_grid=small_grid,
            )
            small_curr_step = downsampler(curr_big_step)
            small_next_step = small_model.advance(small_curr_step)
            target_next_step = downsampler(next_big_step)
            target_val = jax.tree.map(lambda a, b: (a - b) / dt, target_next_step, small_next_step)
            outputs[f"{size}_x"] = small_model.decode(small_curr_step)
            outputs[f"{size}_y"] = small_model.decode(target_val)

        return (t + dt, next_big_step), (t, outputs)

    def gen_traj(v0, num_steps, init_skip, subsample):
        big_model = big_model_cls()
        small_models = {s: sm() for s, sm in small_model_cls.items()}
        enc_v0 = big_model.encode(v0)
        _, (time, res) = ppx.sliced_scan(
            functools.partial(
                step_fn,
                big_model=big_model,
                big_grid=big_grid,
                small_models=small_models,
                small_grids=small_grids,
                dt=dt,
            ),
            (0.0, enc_v0),
            None,
            length=num_steps,
            start=init_skip,
            step=subsample,
        )
        return time, res

    traj_fn = jax.jit(
        jax.vmap(
            functools.partial(
                hk.without_apply_rng(
                    hk.transform(
                        functools.partial(
                            gen_traj,
                            num_steps=total_steps,
                            init_skip=warmup_steps,
                            subsample=stride,
                        )
                    )
                ).apply,
                {},
            ),
            in_axes=0,
            out_axes=(None, 0),
        )
    )

    for param_batches in itertools.batched(
        itertools.islice(
            enumerate(rng_yielder(rng_ctr)),
            traj_slice_start,
            traj_slice_end,
        ),
        args.simultaneous_trajs,
    ):
        traj_nums = []
        rng_ctrs = []
        for b in param_batches:
            traj_nums.append(b[0])
            rng_ctrs.append(b[1])
        logger.info("Generating trajectories %s", ", ".join(map(str, traj_nums)))
        sample_ids = np.array(traj_nums)
        v0 = jax.vmap(
            lambda r: jax_cfd.base.initial_conditions.filtered_velocity_field(
                r, big_grid, args.max_velocity, args.peak_wavenumber
            )
        )(jnp.stack(rng_ctrs))
        v0 = tuple(jnp.expand_dims(e.data, 1) for e in v0)
        logger.info("Starting generation")
        times, trajs = traj_fn(v0)
        logger.info("Finished generating %s samples", times.shape[0])
        # For each small size, produce new dataset and join to datasets on disk
        for small_size in set(args.small_size):
            attrs = {
                "seed": args.seed,
                "ndim": 2,
                "domain_size_multiple": 1,
                "warmup_grid_size": args.big_size,
                "simulation_grid_size": args.big_size,
                "save_grid_size": small_size,
                "warmup_time": args.twarmup,
                "simulation_time": args.tmax - args.twarmup,
                "time_subsample_factor": stride,
                "maximum_velocity": args.max_velocity,
                "init_peak_wavenumber": args.peak_wavenumber,
                "init_cfl_safety_factor": args.cfl_factor,
                "full_config_str": full_config_str,
                "dt": dt,
                "stable_time_step": stable_steps[small_size],
                "viscosity": args.viscosity,
            }
            dataset = xarray.merge(
                [
                    jax_cfd.data.xarray_utils.velocity_trajectory_to_xarray(
                        tuple(array_caster(d) for d in trajs[f"{small_size}_x"]),
                        small_grids[small_size],
                        time=array_caster(times),
                        samples=True,
                        attrs=attrs,
                    ),
                    jax_cfd.data.xarray_utils.velocity_trajectory_to_xarray(
                        tuple(array_caster(d) for d in trajs[f"{small_size}_y"]),
                        small_grids[small_size],
                        time=array_caster(times),
                        samples=True,
                        attrs=attrs,
                    ).rename({"u": "u_corr", "v": "v_corr"}),
                ]
            ).assign_coords(sample=sample_ids)
            # Save to file then drop dataset
            xarray_concatenate_samples(out_file_paths[small_size], dataset)
            dataset = None


def combine_ns_slice(out_dir, args, base_logger):
    out_dir = pathlib.Path(out_dir)
    logger = base_logger.getChild("combine_ns_slice")
    # Locate data slice files
    slice_files = list(out_dir.glob("data-slice*-sz*.nc"))
    if not slice_files:
        logger.error("No slice files found")
        raise ValueError("No slice files found")
    logger.info("Located %d dataset slice files", len(slice_files))
    # Pre-scan: determine trajectories with NaNs (discard) and how many different sizes we're merging
    valid_size_trajs = {}
    for path in slice_files:
        with xarray.open_dataset(path) as dataset:
            size = dataset.sizes["x"]
            if size not in valid_size_trajs:
                valid_size_trajs[size] = {}
            for traj_num, sample_id in enumerate(dataset.sample):
                sample_id = sample_id.item()
                for name in ["u", "v", "u_corr", "v_corr"]:
                    if not np.all(np.isfinite(dataset[name].isel(sample=traj_num))):
                        logger.warning("Trajectory %d size %d contains NaN/inf, will skip. File %s", sample_id, size, path)
                        break
                else:
                    # Trajectory was ok, include it
                    if sample_id in valid_size_trajs[size]:
                        raise ValueError(f"Duplicate entries for %d size %d", sample_id, size)
                    valid_size_trajs[size][sample_id] = path
    logger.info("Merging %d sizes: %s", len(valid_size_trajs), ", ".join(map(str, valid_size_trajs)))
    for k, v in valid_size_trajs.items():
        logger.info("Size %d has %d valid trajectories", k, len(v))
    # Do the actual merge operation (computing stats as we go)
    with h5py.File(out_dir / "data.hdf5", "w", libver="latest") as out_file:
        # Tag this data file as containing NS data
        out_file.create_dataset("model_type", data="ns")
        for size in valid_size_trajs:
            logger.info("Merging size %d", size)
            size_group = out_file.create_group(f"sz{size}")
            # Process parameters
            param_ref_set = valid_size_trajs[size][min(valid_size_trajs[size])]
            logger.info("Referencing parameters from %s", param_ref_set)
            param_group = size_group.create_group("params")
            trajs_group = size_group.create_group("trajs")
            ns_names = ("u", "v", "u_corr", "v_corr")
            stat_accumulators = {
                name: ns_stats.NSStatAccumulator() for name in ns_names
            }
            with xarray.open_dataset(param_ref_set) as dataset:
                for k, v in dataset.attrs.items():
                    param_group.create_dataset(k, data=v)
                # Handle dimension info
                sample_record = trajs_group.create_dataset("sample", shape=(len(valid_size_trajs[size]),), dtype=np.int32)
                trajs_group.create_dataset("time", data=dataset.time.data)
                trajs_group.create_dataset("x", data=dataset.x.data)
                trajs_group.create_dataset("y", data=dataset.y.data)
            # Handle trajectories
            for traj_num, sample_id in enumerate(sorted(valid_size_trajs[size].keys())):
                data_path = valid_size_trajs[size][sample_id]
                sample_record[traj_num] = sample_id
                with xarray.open_dataset(data_path) as dataset:
                    for name in ns_names:
                        batch_data = dataset[name].sel(sample=sample_id).data
                        out_data = trajs_group.create_dataset(f"traj{traj_num:05d}_{name}", data=batch_data)
                        stat_accumulators[name].add_batch(batch_data)
                        batch_data = None
                        for k, v in dataset[name].attrs.items():
                            out_data.attrs[k] = v
            # Add statistics
            stats_group = size_group.create_group("stats")
            for name, stat_comp in stat_accumulators.items():
                name_group = stats_group.create_group(name)
                stats = stat_comp.finalize()
                name_group.create_dataset("mean", data=stats.mean)
                name_group.create_dataset("var", data=stats.var)
                name_group.create_dataset("min", data=stats.min)
                name_group.create_dataset("max", data=stats.max)
    logger.info("Finished merging data")


def shuffle_ns_data(out_dir, args, base_logger):
    out_dir = pathlib.Path(out_dir)
    logger = base_logger.getChild("shuffle_ns_data")
    in_path = out_dir / "data.hdf5"
    out_path = out_dir / "shuffled.hdf5"
    if args.seed is None:
        seed = random.SystemRandom().randint(0, 2**32)
    else:
        seed = args.seed
    logger.info("Using seed %d", seed)
    rng = np.random.default_rng(seed=seed)

    async def proc_worker(in_queue, out_queue, size, in_path, field_types, logger):
        try:
            total_bytes = sum(ft["bytes"] for ft in field_types)
            logger.debug("Spawning process")
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(pathlib.Path(__file__).resolve().parent / "systems" / "ns" / "_background_loader.py"),
                str(in_path),
                "1",
                str(size),
                "--fields",
                *(ft["name"] for ft in field_types),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
            while (job := await in_queue.get()) is not None:
                job_id, traj, step = job
                proc.stdin.write(f"{traj:d} {step:d}\n".encode("utf8"))
                await proc.stdin.drain()
                data_bytes = await proc.stdout.readexactly(total_bytes)
                ret = {}
                byte_cursor = 0
                for field in field_types:
                    ret[field["name"]] = np.frombuffer(data_bytes[byte_cursor:byte_cursor + field["bytes"]], dtype=field["dtype"]).reshape(field["shape"])
                    byte_cursor += field["bytes"]
                data_bytes = None
                await out_queue.put((job_id, ret))
                ret = None
            logger.debug("Cleaning up")
            # Clean up the process
            proc.stdin.write_eof()
            return_code = await proc.wait()
            if return_code != 0:
                logger.error(f"worker process exited abnormally {return_code}")
                raise RuntimeError(f"worker process exited abnormally {return_code}")
        finally:
            logger.debug("Sending termination signal")
            await out_queue.put(None)

    async def process_size(size, in_path, out_dataset, trajs, steps, chunk_size, num_workers, base_logger, field_types):
        logger = base_logger.getChild("process_size")
        # Set up queues
        ready_batch_queue = asyncio.Queue()
        batch_idx_queue = asyncio.Queue()
        batch_size = chunk_size
        total_steps = trajs.shape[0]
        # Spawn workers
        logger.info("Spawning %d workers", num_workers)
        workers = {
            asyncio.create_task(
                proc_worker(
                    in_queue=batch_idx_queue,
                    out_queue=ready_batch_queue,
                    size=size,
                    in_path=in_path,
                    field_types=field_types,
                    logger=logger.getChild(f"worker{worker_id}")
                )
            )
            for worker_id in range(num_workers)
        }
        try:
            iter_range = range(0, total_steps, batch_size)
            for chunk, (start, end) in enumerate(itertools.pairwise(itertools.chain(iter_range, [None]))):
                logger.info("Starting chunk %d of %d", chunk, len(iter_range))
                batch_size = trajs[start:end].shape[0]
                # Submit jobs to workers
                for i, (traj, step) in enumerate(zip(trajs[start:end], steps[start:end], strict=True)):
                    await batch_idx_queue.put((i, traj, step))
                batch_results = []
                for _batch in range(batch_size):
                    result = await ready_batch_queue.get()
                    if result is None:
                        raise RuntimeError("worker exited unexpectedly")
                    batch_results.append(result)
                batch_results.sort(key=operator.itemgetter(0))
                # Stack results and sort
                result_sources = collections.ChainMap(
                    {
                        k: np.stack([br[1][k] for br in batch_results])
                        for k in batch_results[0][1].keys()
                    },
                    {
                        "traj": trajs[start:end],
                        "step": steps[start:end],
                    }
                )
                out_dataset[start:end] = np.core.records.fromarrays(
                    [result_sources[name] for name in out_dataset.dtype.fields],
                    dtype=out_dataset.dtype
                )
        finally:
            # Stop our remaining workers
            logger.info("Stopping background workers")
            for _ in workers:
                await batch_idx_queue.put(None)
            await asyncio.gather(*workers, return_exceptions=True)


    with h5py.File(in_path, "r") as in_file, h5py.File(out_path, "w", libver="latest") as out_file:
        if (model_tag := in_file["model_type"].asstr()[()]) != "ns":
            raise ValueError(f"Incorrect model {model_tag} should be 'ns'")
        out_file.create_dataset("model_type", data="ns")
        # Determine sizes
        ns_names = ("u", "v", "u_corr", "v_corr")
        sizes = set()
        for name in in_file.keys():
            if match := re.match(r"^sz(?P<size>\d+)$", name):
                sizes.add(int(match.group("size")))
        logger.info("Shuffling sizes %s", ", ".join(map(str, sizes)))
        for size in sorted(sizes):
            in_size_group = in_file[f"sz{size}"]
            out_size_group = out_file.create_group(f"sz{size}")
            in_trajs_group = in_size_group["trajs"]
            # Copy params and stats
            for name in {"params", "stats"}:
                in_size_group.copy(
                    source=in_size_group[name],
                    dest=out_size_group,
                    shallow=False,
                    expand_soft=True,
                    expand_external=True,
                    expand_refs=True,
                    without_attrs=False,
                )
            # Copy dimension info
            out_dim_group = out_size_group.create_group("dims")
            for name in {"sample", "time", "x", "y"}:
                in_trajs_group.copy(
                    source=in_trajs_group[name],
                    dest=out_dim_group,
                    shallow=False,
                    expand_soft=True,
                    expand_external=True,
                    expand_refs=True,
                    without_attrs=False,
                )
            # Product compound dtype for shuffled step
            dtype_parts = [
                ("traj", np.uint32),
                ("step", np.uint32),
            ]
            field_types = []
            for name in ns_names:
                sample = in_trajs_group[f"traj00000_{name}"]
                dtype_parts.append(
                    (name, in_trajs_group[f"traj00000_{name}"].dtype, in_trajs_group[f"traj00000_{name}"].shape[1:])
                )
                field_types.append(
                    {
                        "name": name,
                        "dtype": sample.dtype,
                        "shape": sample.shape[1:],
                        "bytes": sample[0:1].nbytes,
                    },
                )
            out_dtype = np.dtype(dtype_parts)
            # Compute shuffle trajs and steps
            num_steps = in_trajs_group[f"traj00000_u"].shape[0]
            num_trajs = sum(1 for name in in_trajs_group.keys() if re.match(r"^traj\d+_u$", name))
            num_samples = num_steps * num_trajs
            indices = np.arange(num_samples, dtype=np.uint64)
            rng.shuffle(indices)
            trajs, steps = np.divmod(indices, num_steps)
            # Handle attributes
            logger.info("Copy attributes")
            out_attrs_group = out_size_group.create_group("attrs")
            for name in ns_names:
                out_attr_name_group = out_attrs_group.create_group(name)
                for k, v in in_trajs_group[f"traj00000_{name}"].attrs.items():
                    out_attr_name_group.create_dataset(k, data=v)
            # Do shuffling
            logger.info("Creating output dataset")
            out_dataset = out_size_group.create_dataset("shuffled", shape=(num_samples,), dtype=out_dtype)
            logger.info("Starting to shuffle size %d", size)
            asyncio.run(
                process_size(
                    size=size,
                    in_path=in_path,
                    out_dataset=out_dataset,
                    trajs=trajs,
                    steps=steps,
                    chunk_size=args.chunk_size,
                    num_workers=args.num_workers,
                    base_logger=logger,
                    field_types=field_types,
                )
            )
            logger.info("Finished shuffling size %d", size)



if __name__ == "__main__":
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    if out_dir.is_file():
        raise ValueError(f"Path must be a directory, not a file: {args.out_dir}")
    out_dir.mkdir(exist_ok=True)
    match args.system:
        case "qg" | "ns":
            if args.traj_slice is not None:
                slice_underscore = args.traj_slice.replace(":", "_")
                log_name = f"run-slice{slice_underscore}.log"
            else:
                log_name = "run.log"
        case "combine_qg_slice" | "combine_ns_slice":
            log_name = "run-combine.log"
        case "shuffle_ns_data":
            log_name = "shuffle_run.log"
        case _:
            log_name = "run.log"
    utils.set_up_logging(level=args.log_level, out_file=out_dir / log_name)
    logger = logging.getLogger("generate_data")
    logger.info("Arguments: %s", vars(args))
    git_info = utils.get_git_info(base_logger=logger)
    if not utils.check_environment_variables(base_logger=logger):
        sys.exit(1)
    if git_info is not None:
        logger.info(
            "Running on commit %s (%s worktree)",
            git_info.hash,
            "clean" if git_info.clean_worktree else "dirty"
        )
    match args.system:
        case "qg":
            gen_qg(out_dir, args, logger)
        case "combine_qg_slice":
            combine_qg_slice(out_dir, args, logger)
        case "ns":
            gen_ns(out_dir, args, logger)
        case "combine_ns_slice":
            combine_ns_slice(out_dir, args, logger)
        case "shuffle_ns_data":
            shuffle_ns_data(out_dir, args, logger)
        case _:
            raise ValueError(f"invalid system: {args.system}")
    logger.info("Finished generating data")
