import argparse
import pathlib
import dataclasses
import utils
import logging
import math
import jax
import jax.numpy as jnp
import numpy as np
import h5py
from systems.qg import utils as qg_utils
from systems.qg.qg_model import QGModel

parser = argparse.ArgumentParser(description="Generate data for a variety of systems")
parser.add_argument("out_dir", type=str, help="Directory to store output (created if non-existing)")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
subparsers = parser.add_subparsers(help="Choice of system to generate", dest="system", required=True)

# QG options
parser_qg = subparsers.add_parser("qg", help="Generate training data like PyQG")
parser_qg.add_argument("seed", type=int, help="RNG seed, must be unique for unique trajectory")
parser_qg.add_argument("--dt", type=float, default=7200.0, help="Time step size")
parser_qg.add_argument("--tmax", type=float, default=157680000.0 * 2, help="End time for the model")
parser_qg.add_argument("--big_size", type=int, default=64, help="Scale of large model")
parser_qg.add_argument("--small_size", type=int, default=16, help="Scale of small model")
parser_qg.add_argument("--num_trajs", type=int, default=1, help="Number of trajectories to generate")


def gen_qg(out_dir, args, base_logger):
    logger = base_logger.getChild("qg")
    logger.info("Generating trajectory for QG with seed %d", args.seed)
    # Initialize models
    big_model = QGModel(nx=args.big_size, ny=args.big_size, dt=args.dt, tmax=args.tmax)
    small_model = QGModel(nx=args.small_size, ny=args.small_size, dt=args.dt, tmax=args.tmax)
    # Set up data generator
    rng_ctr = jax.random.PRNGKey(seed=args.seed)
    spectral_coarsener = qg_utils.SpectralCoarsener(big_model=big_model, small_model=small_model)
    # Do computations
    num_steps = math.ceil(args.tmax / args.dt)
    traj_gen = qg_utils.make_gen_traj(big_model=big_model, spectral_coarsener=spectral_coarsener)
    logger.info("Generating %d trajectories with %d steps", args.num_trajs, num_steps)
    with h5py.File(out_dir / "data.hdf5", "w", libver="latest") as out_file:
        # Store model parameters
        param_group = out_file.create_group("params")
        param_group.create_dataset("big_model", data=big_model.param_json())
        param_group.create_dataset("small_model", data=small_model.param_json())
        # Generate trajectories
        root_group = out_file.create_group("trajs")
        compound_dtype = None
        for traj_num in range(args.num_trajs):
            rng, rng_ctr = jax.random.split(rng_ctr, 2)
            logger.info("Starting trajectory %d", traj_num)
            traj = traj_gen(rng=rng, num_steps=num_steps)
            logger.info("Finished generating trajectory %d", traj_num)
            if compound_dtype is None:
                dtype_fields = []
                for k, v in dataclasses.asdict(traj).items():
                    if v.ndim > 1:
                        dtype_fields.append((k, v.dtype, v.shape[1:]))
                    else:
                        dtype_fields.append((k, v.dtype))
                compound_dtype = np.dtype(dtype_fields)
            data_arr = np.empty(num_steps, dtype=compound_dtype)
            for k, v in dataclasses.asdict(traj).items():
                data_arr[k] = v
            # Store results
            root_group.create_dataset(f"traj{traj_num:05d}", data=data_arr)
            logger.info("Finished storing trajectory %d", traj_num)


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
