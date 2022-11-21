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
import asyncio
import operator
import sys
import utils

parser = argparse.ArgumentParser(description="Pre-shuffle data for faster training")
parser.add_argument("data_set", type=str, help="Directory containing original data to shuffle")
parser.add_argument("--seed", type=int, default=None, help="Seed for shuffling")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])


CHUNK_SIZE = 1000
NUM_WORKERS = 32
FIELD_Q_RE = re.compile(rf"^traj\d{{5}}_q$")
FIELD_Q_FORCING_RE = re.compile(rf"^traj\d{{5}}_q_total_forcing$")


async def proc_worker(in_queue, out_queue, data_file, data_dtype):
    try:
        q_shape = data_dtype["q"].shape
        forcing_shape = data_dtype["q_total_forcing"].shape
        q_dtype = np.zeros(1, dtype=data_dtype)["q"].dtype
        forcing_dtype = np.zeros(1, dtype=data_dtype)["q_total_forcing"].dtype
        q_bytes = np.zeros(1, dtype=data_dtype)["q"].nbytes
        forcing_bytes = np.zeros(1, dtype=data_dtype)["q_total_forcing"].nbytes
        total_bytes = q_bytes + forcing_bytes
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(pathlib.Path(__file__).resolve().parent / "systems" / "qg" / "_loader.py"),
            str(data_file),
            "1",
            "--fields",
            "q",
            "q_total_forcing",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        while True:
            job = await in_queue.get()
            if job is None:
                break
            job_id, traj, step = job
            proc.stdin.write(f"{traj:d} {step:d}\n".encode("utf8"))
            await proc.stdin.drain()
            data_bytes = await proc.stdout.readexactly(total_bytes)
            q_result = np.frombuffer(data_bytes[:q_bytes], dtype=q_dtype).reshape(q_shape)
            forcing_result = np.frombuffer(data_bytes[-forcing_bytes:], dtype=forcing_dtype).reshape(forcing_shape)
            data_bytes = None
            await out_queue.put((job_id, q_result, forcing_result))
            q_result = None
            forcing_result = None
        # Cleanup the process
        proc.stdin.write_eof()
        return_code = await proc.wait()
        if return_code != 0:
            raise RuntimeError(f"worker process exited abnormally {return_code}")
    finally:
        await out_queue.put(None)


async def do_work(out_path, in_path, data_dtype, source_record_dtype, num_chunks, trajs, steps, total_steps):
    logger = logging.getLogger("worker")
    # Create output record (empty)
    ready_batch_queue = asyncio.Queue()
    batch_idx_queue = asyncio.Queue()
    # Spawn workers
    workers = {
        asyncio.create_task(
            proc_worker(
                in_queue=batch_idx_queue,
                out_queue=ready_batch_queue,
                data_file=in_path,
                data_dtype=data_dtype,
            )
        )
        for _ in range(NUM_WORKERS)
    }
    try:
        with h5py.File(out_path, "w") as out_data:
            data_dataset = out_data.create_dataset("shuffled", shape=(total_steps, ), dtype=data_dtype)
            source_dataset = out_data.create_dataset("source", shape=(total_steps, ), dtype=source_record_dtype)
            for chunk, (start, end) in enumerate(itertools.pairwise(itertools.chain(range(0, total_steps, CHUNK_SIZE), [None]))):
                logger.info("Starting chunk %d of %d", chunk, num_chunks)
                # Submit jobs to workers
                batch_size = 0
                for i, (traj, step) in enumerate(zip(trajs[start:end], steps[start:end], strict="true")):
                    await batch_idx_queue.put((i, traj, step))
                    batch_size += 1
                batch_results = []
                for _batch in range(batch_size):
                    result = await ready_batch_queue.get()
                    if result is None:
                        raise RuntimeError("worker exited unexpectedly")
                    batch_results.append(result)
                batch_results.sort(key=operator.itemgetter(0))
                # Stack results and write
                data_dataset[start:end] = np.core.records.fromarrays(
                    [
                        np.stack([b[1] for b in batch_results], axis=0),
                        np.stack([b[2] for b in batch_results], axis=0),
                    ],
                    dtype=data_dtype,
                )
                source_dataset[start:end] = np.core.records.fromarrays(
                    [
                        trajs[start:end],
                        steps[start:end],
                    ],
                    dtype=source_record_dtype,
                )
                batch_results = None
    finally:
        # Stop our remaining workers
        for _ in range(NUM_WORKERS):
            await batch_idx_queue.put(None)
        await asyncio.gather(*workers, return_exceptions=True)


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
    # Done with input file in this process
    num_chunks, rem = divmod(total_steps, CHUNK_SIZE)
    num_chunks += (1 if rem != 0 else 0)
    asyncio.run(
        do_work(
            out_path=out_path,
            in_path=in_path,
            data_dtype=data_dtype,
            source_record_dtype=source_record_dtype,
            num_chunks=num_chunks,
            trajs=trajs,
            steps=steps,
            total_steps=total_steps,
        )
    )


if __name__ == "__main__":
    main()
