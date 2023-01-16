import argparse
import pathlib
import dataclasses
import utils
import logging
import contextlib
import math
import os
import operator
import ast
import sys
import dataclasses
import jax
from jaxtyping import Array
import jax.numpy as jnp
import numpy as np
import h5py
from systems.qg import utils as qg_utils
from systems.qg.qg_model import QGModel
from systems.qg import coarsen, compute_stats
import jax_utils

parser = argparse.ArgumentParser(description="Generate data for a variety of systems")
parser.add_argument("out_dir", type=str, help="Directory to store output (created if non-existing)")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
subparsers = parser.add_subparsers(help="Choice of system to generate", dest="system", required=True)

# QG options
parser_qg = subparsers.add_parser("qg", help="Generate training data like PyQG")
parser_qg.add_argument("seed", type=int, help="RNG seed, must be unique for unique trajectory")
parser_qg.add_argument("--dt", type=float, default=3600.0, help="Time step size")
parser_qg.add_argument("--tmax", type=float, default=311040000.0, help="End time for the model")
parser_qg.add_argument("--big_size", type=int, default=256, help="Scale of large model")
parser_qg.add_argument("--small_size", type=int, nargs="+", default=[128, 96, 64], help="Scale of small model")
parser_qg.add_argument("--subsample", type=int, default=1, help="Stride used to select how many generated steps to keep")
parser_qg.add_argument("--num_trajs", type=int, default=1, help="Number of trajectories to generate")
parser_qg.add_argument("--config", type=str, default="eddy", help="Eddy or jet configuration", choices=["eddy", "jet"])
parser_qg.add_argument("--coarse_op", type=str, default="op1", help="Which coarsening operators to apply", choices=sorted(coarsen.COARSEN_OPERATORS.keys()))

# QG snapshot options
parser_qg_snap = subparsers.add_parser("qg_snap", help="Generate training data like PyQG")
parser_qg_snap.add_argument("seed", type=int, help="RNG seed, must be unique for unique trajectory")
parser_qg_snap.add_argument("--dt", type=float, default=3600.0, help="Time step size")
parser_qg_snap.add_argument("--tmax", type=float, default=155520000.0, help="End time for the model")
parser_qg_snap.add_argument("--big_size", type=int, default=256, help="Scale of large model")
parser_qg_snap.add_argument("--small_size", type=int, default=64, help="Scale of small model")
parser_qg_snap.add_argument("--num_trajs", type=int, default=1, help="Number of trajectories to generate")
parser_qg_snap.add_argument("--batch_size", type=int, default=100, help="Number of generation steps to batch together")
parser_qg_snap.add_argument("--config", type=str, default="eddy", help="Eddy or jet configuration", choices=["eddy", "jet"])

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


def make_coarsen_to_size(coarse_op, big_model):

    def do_coarsening(var, small_nx):
        assert var.shape[-2:] == (big_model.ny, big_model.nx)
        if big_model.nx == small_nx:
            # No coarsening needed
            res = coarsen.NoOpCoarsener(big_model=big_model, small_nx=small_nx).coarsen(var)
        else:
            res = coarse_op(big_model=big_model, small_nx=small_nx).coarsen(var)

        assert res.shape[-2:] == (small_nx, small_nx)
        return res

    return do_coarsening



@jax_utils.register_pytree_dataclass
@dataclasses.dataclass
class CoarseTrajResult:
    q: Array
    t: Array
    tc: Array
    ablevel: Array
    dqhdt: Array
    q_total_forcings: dict[int, Array]


def make_generate_coarse_traj(big_model, small_sizes, coarse_op_cls, num_steps, subsample):
    coarse_ops = {}
    for size in set(small_sizes):
        size = operator.index(size)
        if size != big_model.nx:
            op = coarse_op_cls(
                big_model=big_model,
                small_nx=size,
            )
        else:
            op = coarsen.NoOpCoarsener(
                big_model=big_model,
                small_nx=size,
            )
        coarse_ops[size] = op
    size_main_states = max(map(operator.index, small_sizes))

    # Compute dummy initialization shapes
    _jaxpr, dummy_init_treedef = jax.make_jaxpr(coarse_ops[size_main_states].small_model.create_initial_state, return_shape=True)(jax.random.PRNGKey(0))
    dummy_shape = dummy_init_treedef.dqhdt.shape
    dummy_dtype = dummy_init_treedef.dqhdt.dtype

    def make_traj(big_initial_step):

        def _step_forward(carry, x):
            # Unpack carry
            prev_big_state, small_dqhdt, small_dqhdt_p = carry
            # Produce new "main size" state for output
            prev_small_state = coarse_ops[size_main_states]._coarsen_step(prev_big_state)
            # Shift over dqhdts
            dqhdt_p = small_dqhdt
            # Step the large state forward
            next_big_state = big_model.step_forward(prev_big_state)
            # Return carry, y
            return (next_big_state, prev_small_state.dqhdt, dqhdt_p), CoarseTrajResult(
                q=prev_small_state.q,
                t=prev_small_state.t,
                tc=prev_small_state.tc,
                ablevel=prev_small_state.ablevel,
                dqhdt=prev_small_state.dqhdt,
                q_total_forcings={
                    k: op._coarsen_step(prev_big_state).q_total_forcing
                    for k, op in coarse_ops.items()
                },
            )

        dummy_dqhdt_single = jnp.zeros(shape=dummy_shape, dtype=dummy_dtype)
        dummy_dqhdt_double = jnp.zeros(shape=((2, ) + dummy_shape), dtype=dummy_dtype)

        _carry, results = jax_utils.strided_scan(
            _step_forward,
            (big_initial_step, dummy_dqhdt_single, dummy_dqhdt_single),
            None,
            length=num_steps,
            stride=subsample,
        )
        results = dataclasses.replace(
            results,
            dqhdt=jnp.concatenate(
                [
                    dummy_dqhdt_double,
                    results.dqhdt,
                ],
            )
        )
        return results

    return make_traj


def make_generate_final_step(big_model, coarse_ops, num_steps):

    def step_forward(carry, _x):
        prev_big_state = carry
        next_big_state = big_model.step_forward(prev_big_state)
        return next_big_state, None

    def make_traj(rng):
        big_initial_step = big_model.create_initial_state(rng)
        final_state, _ys = jax.lax.scan(step_forward, big_initial_step, None, length=num_steps - 1)
        # Coarsen the final state
        coarsened = {
            op_name: op._coarsen_step(final_state).q
            for op_name, op in coarse_ops.items()
        }
        return coarsened

    return make_traj


def chunk_array(arr, chunk, axis=0):
    if chunk < 1:
        raise ValueError(f"invalid array chunk size {chunk}")
    return jnp.split(arr, jnp.arange(chunk, arr.shape[axis], chunk), axis=axis)


def gen_qg_final_step(out_dir, args, base_logger):
    out_dir = pathlib.Path(out_dir)
    logger = base_logger.getChild("qg")
    logger.info("Generating final step snapshots for QG with seed %d", args.seed)
    # Initialize models
    big_model = QGModel(nx=args.big_size, ny=args.big_size, dt=args.dt, tmax=args.tmax, **CONFIG_VARS[args.config])
    # Initialize coarsening operators
    coarsen_operators = {
        "op1": coarsen.Operator1(big_model=big_model, small_nx=args.small_size),
        "op2": coarsen.Operator2(big_model=big_model, small_nx=args.small_size),
    }
    # Set up data generator
    rng_ctr = jax.random.PRNGKey(seed=args.seed)
    rng_ctr, step_rng = jnp.split(jax.random.split(rng_ctr, num=args.num_trajs + 1), [1])
    # Do computations
    num_steps = math.ceil(args.tmax / args.dt)
    gen_func = jax.jit(jax.vmap(make_generate_final_step(big_model=big_model, coarse_ops=coarsen_operators, num_steps=num_steps)))
    # Create output files and directories
    op_directories = {}
    for op_name in coarsen_operators.keys():
        op_directories[op_name] = out_dir / op_name
        op_directories[op_name].mkdir(exist_ok=True)
    with contextlib.ExitStack() as exit_stack:
        # Create output files
        op_files = {
            op_name: exit_stack.enter_context(h5py.File(op_directories[op_name] / "data.hdf5", "w", libver="latest"))
            for op_name in coarsen_operators.keys()
        }
        # Store model parameters in each file
        for op_name, op in coarsen_operators.items():
            out_file = op_files[op_name]
            param_group = out_file.create_group("params")
            param_group.create_dataset("big_model", data=big_model.param_json())
            param_group.create_dataset("small_model", data=op.small_model.param_json())
            param_group.create_dataset("num_steps", data=num_steps)
        # Create a dataset in each file
        op_datasets = {}
        for op_name in coarsen_operators.keys():
            # Get dtype
            dummy_q = coarsen_operators[op_name].small_model.create_initial_state(jax.random.PRNGKey(0)).q
            op_datasets[op_name] = op_files[op_name].create_dataset("final_q_steps", shape=(args.num_trajs, ) + dummy_q.shape, dtype=dummy_q.dtype)
        # Create the dataset in the output file
        for batch_num, rng_chunk in enumerate(chunk_array(step_rng, chunk=args.batch_size)):
            idx_start = args.batch_size * batch_num
            idx_end = min(args.batch_size * (batch_num + 1), args.num_trajs)
            logger.info("Starting batch %d with trajectories %d to %d of %d", batch_num, idx_start, idx_end, args.num_trajs)
            final_q_steps = jax.device_get(gen_func(rng_chunk))
            logger.info("Finished batch %d", batch_num)
            # Store results
            for op_name in coarsen_operators.keys():
                op_files[op_name]["final_q_steps"][idx_start:idx_end] = final_q_steps[op_name]
            logger.info("Finished storing batch %d", batch_num)


def gen_qg(out_dir, args, base_logger):
    out_dir = pathlib.Path(out_dir)
    logger = base_logger.getChild("qg")
    logger.info("Generating trajectory for QG with seed %d", args.seed)
    # Initialize model and locate coarsening class
    small_sizes = sorted(set(map(operator.index, args.small_size)))
    main_small_size = max(map(operator.index, small_sizes))
    big_model = QGModel(nx=args.big_size, ny=args.big_size, dt=args.dt, tmax=args.tmax, **CONFIG_VARS[args.config])
    initial_state_fn = jax.jit(big_model.create_initial_state)
    op_name = args.coarse_op
    coarse_cls = coarsen.COARSEN_OPERATORS[op_name]
    # Set up data generator
    rng_ctr = jax.random.PRNGKey(seed=args.seed)
    # Create trajectory generation function
    num_steps = math.ceil(args.tmax / args.dt)
    traj_gen_func = jax.jit(
        make_generate_coarse_traj(
            big_model=big_model,
            small_sizes=small_sizes,
            coarse_op_cls=coarse_cls,
            num_steps=num_steps,
            subsample=args.subsample,
        )
    )
    coarsen_fn = jax.jit(
        jax.vmap(
            make_coarsen_to_size(
                coarse_op=coarse_cls,
                big_model=coarse_cls(big_model=big_model, small_nx=main_small_size).small_model,
            ),
            in_axes=(0, None),
        ),
        static_argnums=(1, ),
    )
    # Do computations
    logger.info("Generating %d trajectories with %d steps (subsampled by a factor of %d to %d steps)", args.num_trajs, num_steps, args.subsample, num_steps // args.subsample)
    # Create directory for this operator
    op_directory = out_dir / op_name
    op_directory.mkdir(exist_ok=True)
    # Prep stats computers
    forcing_stats = {sz: compute_stats.QGStatAccumulator() for sz in small_sizes}
    q_stats = {sz: compute_stats.QGStatAccumulator() for sz in small_sizes}
    with h5py.File(op_directory / "data.hdf5", "w", libver="latest") as out_file:
        # Store model parameters in each file
        param_group = out_file.create_group("params")
        param_group.create_dataset("big_model", data=big_model.param_json())
        for size in set(small_sizes):
            size = operator.index(size)
            size_coarse_cls = coarse_cls if size != args.big_size else coarsen.NoOpCoarsener
            param_group.create_dataset(f"small_model_{size}", data=size_coarse_cls(big_model=big_model, small_nx=size).small_model.param_json())
        param_group["small_model"] = h5py.SoftLink(f"/params/small_model_{main_small_size}")
        # Generate trajectories
        compound_dtype = None
        out_file.create_group("trajs")
        for traj_num in range(args.num_trajs):
            rng, rng_ctr = jax.random.split(rng_ctr, 2)
            logger.info("Starting trajectory %d", traj_num)
            result = traj_gen_func(initial_state_fn(rng))
            logger.info("Finished generating trajectory %d", traj_num)
            if traj_num == 0:
                # First trajectory, store t, tc, ablevel
                out_file["trajs"].create_dataset("t", data=np.asarray(result.t))
                out_file["trajs"].create_dataset("tc", data=np.asarray(result.tc))
                out_file["trajs"].create_dataset("ablevel", data=np.asarray(result.ablevel))
            # Store q and dqhdt
            out_file["trajs"].create_dataset(f"traj{traj_num:05d}_q", data=np.asarray(result.q))
            out_file["trajs"].create_dataset(f"traj{traj_num:05d}_dqhdt", data=np.asarray(result.dqhdt))
            # Store forcings
            for size in small_sizes:
                forcing_dataset = out_file["trajs"].create_dataset(f"traj{traj_num:05d}_q_total_forcing_{size}", data=np.asarray(result.q_total_forcings[size]))
            out_file["trajs"][f"traj{traj_num:05d}_q_total_forcing"] = h5py.SoftLink(f"/trajs/traj{traj_num:05d}_q_total_forcing_{main_small_size}")
            logger.info("Finished storing trajectory %d", traj_num)
            # Keep only q and forcings and discard the rest
            logger.info("Updating statistics for trajectory %d", traj_num)
            traj_q = result.q
            traj_forcings = result.q_total_forcings
            result = None
            # Update statistics
            #   First, the forcings
            for size in small_sizes:
                forcing_stats[size].add_batch(np.asarray(traj_forcings[size]).astype(np.float64))
            traj_forcings = None
            #   Next, the q values which require their own coarsening steps
            for size in small_sizes:
                q_stats[size].add_batch(np.asarray(coarsen_fn(traj_q, size)).astype(np.float64))
            logger.info("Finished staticstics for trajectory %d", traj_num)
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



if __name__ == "__main__":
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    if out_dir.is_file():
        raise ValueError(f"Path must be a directory, not a file: {args.out_dir}")
    out_dir.mkdir(exist_ok=True)
    utils.set_up_logging(level=args.log_level, out_file=out_dir/"run.log")
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
        case "qg_snap":
            gen_qg_final_step(out_dir, args, logger)
        case _:
            raise ValueError(f"invalid system: {args.system}")
    logger.info("Finished generating data")
