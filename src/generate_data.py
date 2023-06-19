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
import jax_utils
import pyqg_jax


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
parser_qg.add_argument("--config", type=str, default="eddy", help="Eddy or jet configuration", choices=sorted(CONFIG_VARS.keys()))
parser_qg.add_argument("--coarse_op", type=str, default="op1", help="Which coarsening operators to apply", choices=sorted(coarsen.COARSEN_OPERATORS.keys()))
parser_qg.add_argument("--max_gen_tries", type=int, default=5, help="Number of retries to generate a trajectory")


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



@jax_utils.register_pytree_dataclass
@dataclasses.dataclass
class CoarseTrajResult:
    q: jax.Array
    t: jax.Array
    tc: jax.Array
    ablevel: jax.Array
    dqhdt: jax.Array
    q_total_forcings: dict[int, jax.Array]


def make_generate_coarse_traj(big_model, small_sizes, coarse_op_cls, num_warmup_steps, num_steps, subsample):
    if num_warmup_steps >= num_steps:
        raise ValueError("warmup steps {num_warmup_steps} larger than total steps {num_steps}")
    coarse_ops = {}
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

    def make_traj(big_initial_step):

        def _step_until_warmup(carry, _x):
            prev_big_state = carry
            next_big_state = big_model.step_model(prev_big_state)
            return next_big_state, None

        def _step_forward(carry, x):
            prev_big_state = carry
            # Produce new "main size" state for output
            prev_small_q = coarse_ops[size_main_states].coarsen(prev_big_state.state.q)
            prev_big_dqhdt = big_model.get_full_state(prev_big_state).dqhdt
            prev_small_dqhdt = coarse_ops[size_main_states].coarsen(prev_big_dqhdt)
            # Step the large model forward
            next_big_state = big_model.step_model(prev_big_state)
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


def gen_qg(out_dir, args, base_logger):
    out_dir = pathlib.Path(out_dir)
    logger = base_logger.getChild("qg")
    logger.info("Generating trajectory for QG with seed %d", args.seed)
    # Initialize model and locate coarsening class
    small_sizes = sorted(set(map(operator.index, args.small_size)))
    main_small_size = max(map(operator.index, small_sizes))
    big_model = pyqg_jax.steppers.SteppedModel(
        model=pyqg_jax.qg_model.QGModel(
            nx=args.big_size,
            ny=args.big_size,
            **CONFIG_VARS[args.config],
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
    with h5py.File(op_directory / "data.hdf5", "w", libver="latest") as out_file:
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
        for traj_num in range(args.num_trajs):
            logger.info("Starting trajectory %d", traj_num)
            for generation_trial in range(arg.max_gen_tries):
                logger.info("Generation attempt %d of %d", generation_trial + 1, arg.max_gen_tries)
                rng, rng_ctr = jax.random.split(rng_ctr, 2)
                result = traj_gen_func(initial_state_fn(rng))
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
            out_file["trajs"].create_dataset(f"traj{traj_num:05d}_q", data=np.asarray(result.q))
            out_file.flush()
            out_file["trajs"].create_dataset(f"traj{traj_num:05d}_dqhdt", data=np.asarray(result.dqhdt))
            out_file.flush()
            # Store forcings
            for size in small_sizes:
                forcing_dataset = out_file["trajs"].create_dataset(f"traj{traj_num:05d}_q_total_forcing_{size}", data=np.asarray(result.q_total_forcings[size]))
                out_file.flush()
            out_file["trajs"][f"traj{traj_num:05d}_q_total_forcing"] = h5py.SoftLink(f"/trajs/traj{traj_num:05d}_q_total_forcing_{main_small_size}")
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
        case _:
            raise ValueError(f"invalid system: {args.system}")
    logger.info("Finished generating data")
