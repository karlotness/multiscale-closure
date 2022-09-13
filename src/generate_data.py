import argparse
import pathlib
import dataclasses
import utils
import logging
import contextlib
import math
import jax
import jax.numpy as jnp
import numpy as np
import h5py
from systems.qg import utils as qg_utils
from systems.qg.qg_model import QGModel
from systems.qg import coarsen

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
parser_qg.add_argument("--small_size", type=int, default=64, help="Scale of small model")
parser_qg.add_argument("--num_trajs", type=int, default=1, help="Number of trajectories to generate")
parser_qg.add_argument("--config", type=str, default="eddy", help="Eddy or jet configuration", choices=["eddy", "jet"])

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


def make_generate_coarse_traj(coarse_op, num_steps):
    dummy_init = coarse_op.small_model.create_initial_state(jax.random.PRNGKey(0))
    dummy_dqhdt = jnp.zeros_like(dummy_init.dqhdt)

    def make_traj(big_initial_step):

        def _step_forward(carry, x):
            prev_big_state, small_dqhdt, small_dqhdt_p = carry
            prev_small_state = coarse_op._coarsen_step(prev_big_state)
            prev_small_state.dqhdt_p = small_dqhdt
            prev_small_state.dqhdt_pp = small_dqhdt_p
            next_big_state = coarse_op.big_model.step_forward(prev_big_state)
            return (next_big_state, prev_small_state.dqhdt, prev_small_state.dqhdt_p), prev_small_state

        _carry, coarse_states = jax.lax.scan(
            _step_forward,
            (big_initial_step, dummy_dqhdt, dummy_dqhdt),
            None,
            length=num_steps,
        )
        return coarse_states

    return make_traj


def gen_qg(out_dir, args, base_logger):
    out_dir = pathlib.Path(out_dir)
    logger = base_logger.getChild("qg")
    logger.info("Generating trajectory for QG with seed %d", args.seed)
    # Initialize models
    big_model = QGModel(nx=args.big_size, ny=args.big_size, dt=args.dt, tmax=args.tmax, **CONFIG_VARS[args.config])
    # Initialize coarsening operators
    coarsen_operators = {
        "op1": coarsen.Operator1(big_model=big_model, small_nx=args.small_size),
        "op2": coarsen.Operator2(big_model=big_model, small_nx=args.small_size),
    }
    # Set up data generator
    rng_ctr = jax.random.PRNGKey(seed=args.seed)
    # Do computations
    num_steps = math.ceil(args.tmax / args.dt)
    traj_gen_ops = {
        op_name: jax.jit(
            make_generate_coarse_traj(op, num_steps)
        )
        for op_name, op in coarsen_operators.items()
    }
    logger.info("Generating %d trajectories with %d steps", args.num_trajs, num_steps)
    # Create directories for each operator
    op_directories = {}
    for op_name in coarsen_operators.keys():
        op_directories[op_name] = out_dir / op_name
        op_directories[op_name].mkdir(exist_ok=True)
    with contextlib.ExitStack() as exit_stack:
        # Create and open files for each operator
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
            out_file.create_group("trajs")
        # Generate trajectories
        compound_dtype = None
        for traj_num in range(args.num_trajs):
            rng, rng_ctr = jax.random.split(rng_ctr, 2)
            initial_step = big_model.create_initial_state(rng)
            for op_name, op in coarsen_operators.items():
                out_file = op_files[op_name]
                traj_gen = traj_gen_ops[op_name]
                logger.info("Starting trajectory %d for operator %s", traj_num, op_name)
                traj = jax.device_get(traj_gen(initial_step))
                logger.info("Finished generating trajectory %d for operator %s", traj_num, op_name)
                # First time: determine the compound dtype
                if compound_dtype is None:
                    dtype_fields = []
                    for k, v in dataclasses.asdict(traj).items():
                        if v.ndim > 1:
                            dtype_fields.append((k, v.dtype, v.shape[1:]))
                        else:
                            dtype_fields.append((k, v.dtype))
                    compound_dtype = np.dtype(dtype_fields)
                # Combine the data
                data_arr = np.empty(num_steps, dtype=compound_dtype)
                for k, v in dataclasses.asdict(traj).items():
                    data_arr[k] = v
                del traj
                out_file["trajs"].create_dataset(f"traj{traj_num:05d}", data=data_arr)
                del data_arr
                logger.info("Finished storing trajectory %d for operator %s", traj_num, op_name)


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
    if git_info is not None:
        logger.info(
            "Running on commit %s (%s worktree)",
            git_info.hash,
            "clean" if git_info.clean_worktree else "dirty"
        )
    if args.system == "qg":
        gen_qg(out_dir, args, logger)
    else:
        raise ValueError(f"invalid system: {args.system}")
    logger.info("Finished generating data")
