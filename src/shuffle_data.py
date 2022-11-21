# Pre-shuffle data for faster loading as snapshots (slow to pre-process, then faster to load)
# Handles fields q and q_total_forcing (includes references for steps and traj). No other data copied
# Intended to be applied to training set only
import argparse
import pathlib
import itertools
import random
import numpy as np
import h5py
import re
import logging
import math
import utils

parser = argparse.ArgumentParser(description="Pre-shuffle data for faster training")
parser.add_argument("data_set", type=str, help="Directory containing original data to shuffle")
parser.add_argument("--seed", type=int, default=None, help="Seed for shuffling")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])


CHUNK_SIZE = 1000
FIELD_Q_RE = re.compile(rf"^traj\d{{5}}_q$")
FIELD_Q_FORCING_RE = re.compile(rf"^traj\d{{5}}_q_total_forcing$")


def main():
    args = parser.parse_args()
    data_dir = pathlib.Path(args.data_set)
    utils.set_up_logging(level=args.log_level, out_file=data_dir/"shuffle_run.log")
    logger = logging.getLogger("main")
    logger.info("Arguments: %s", vars(args))
    git_info = utils.get_git_info(base_logger=logger)
    if git_info is not None:
        logger.info(
            "Running on commit %s (%s worktree)",
            git_info.hash,
            "clean" if git_info.clean_worktree else "dirty"
        )
    in_path = data_dir / "data.hdf5"
    out_path = data_dir / "shuffled.hdf5"
    if args.seed is None:
        seed = random.SystemRandom().randint(0, 2**32)
    else:
        seed = args.seed
    logger.info("Using seed %d", seed)
    rng = np.random.default_rng(seed=seed)
    with h5py.File(in_path, "r") as in_data:
        # Compute number of trajectories and steps
        num_trajs = sum(1 for k in in_data["trajs"].keys() if FIELD_Q_RE.match(k))
        num_steps = in_data["trajs"]["traj00000_q"].shape[0]
        total_steps = num_trajs * num_steps
        logger.info("Shuffling %d trajs of %d steps -> %d total items", num_trajs, num_steps, total_steps)
        # Shuffle indices
        indices = np.arange(total_steps, dtype=np.uint64)
        rng.shuffle(indices)
        trajs, steps = np.divmod(indices, num_steps)
        # Create compound dtypes
        data_dtype = np.dtype(
            [
                ("q", in_data["trajs"]["traj00000_q"].dtype, in_data["trajs"]["traj00000_q"].shape[1:]),
                ("q_total_forcing", in_data["trajs"]["traj00000_q_total_forcing"].dtype, in_data["trajs"]["traj00000_q_total_forcing"].shape[1:]),
            ]
        )
        source_record_dtype = np.dtype(
            [
                ("traj", np.uint16),
                ("step", np.uint32),
            ]
        )
        num_chunks, rem = divmod(total_steps, CHUNK_SIZE)
        num_chunks += (1 if rem != 0 else 0)
        # Create output record (empty)
        with h5py.File(out_path, "w") as out_data:
            data_dataset = out_data.create_dataset("shuffled", shape=(total_steps, ), dtype=data_dtype)
            source_dataset = out_data.create_dataset("source", shape=(total_steps, ), dtype=source_record_dtype)
            in_trajs = in_data["trajs"]
            for chunk, (start, end) in enumerate(itertools.pairwise(itertools.chain(range(0, total_steps, CHUNK_SIZE), [None]))):
                logger.info("Starting chunk %d of %d", chunk, num_chunks)
                q_chunk = []
                q_forcing_chunk = []
                source_traj_chunk = []
                source_step_chunk = []
                # Load data
                for traj, step in zip(trajs[start:end], steps[start:end], strict="true"):
                    q_chunk.append(in_trajs[f"traj{traj:05d}_q"][step])
                    q_forcing_chunk.append(in_trajs[f"traj{traj:05d}_q_total_forcing"][step])
                    source_traj_chunk.append(np.uint16(traj))
                    source_step_chunk.append(np.uint32(step))
                # Stack and write to file
                data_dataset[start:end] = np.core.records.fromarrays(
                    [
                        np.stack(q_chunk),
                        np.stack(q_forcing_chunk),
                    ],
                    dtype=data_dtype,
                )
                source_dataset[start:end] = np.core.records.fromarrays(
                    [
                        np.stack(source_traj_chunk),
                        np.stack(source_step_chunk),
                    ],
                    dtype=source_record_dtype,
                )
    logger.info("Finished shuffling data")


if __name__ == "__main__":
    main()
