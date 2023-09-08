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


def get_field_info(data_dtype, field_name):
    dummy = np.zeros(1, dtype=data_dtype)[field_name].copy()
    shape = data_dtype[field_name].shape
    return shape, dummy.dtype, dummy.nbytes


async def proc_worker(in_queue, out_queue, data_file, data_dtype):
    try:
        total_bytes = 0
        field_infos = []
        for field_name in data_dtype.names:
            field_shape, field_dtype, field_bytes = get_field_info(data_dtype, field_name)
            total_bytes += field_bytes
            field_infos.append((field_name, field_shape, field_dtype, field_bytes))
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(pathlib.Path(__file__).resolve().parent / "systems" / "qg" / "_loader.py"),
            str(data_file),
            "1",
            "--fields",
            *[name for (name, _, _, _) in field_infos],
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
            ret = np.empty((), dtype=data_dtype)
            byte_cursor = 0
            for field_name, field_shape, field_dtype, field_bytes in field_infos:
                ret[field_name] = np.frombuffer(data_bytes[byte_cursor:byte_cursor + field_bytes], dtype=field_dtype).reshape(field_shape)
                byte_cursor += field_bytes
            data_bytes = None
            await out_queue.put((job_id, ret))
            ret = None
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
                data_dataset[start:end] = np.stack([b[1] for b in batch_results], axis=0)
                source_dataset[start:end] = np.core.records.fromarrays(
                    [
                        trajs[start:end],
                        steps[start:end],
                    ],
                    dtype=source_record_dtype,
                )
                batch_results = None
            # After shuffling, copy over statistics from source file
            with h5py.File(in_path, "r") as in_data:
                logger.info("Copying statistics and parameters")
                in_data.copy(source=in_data["/stats"], dest=out_data["/"], shallow=False, expand_soft=True, expand_external=True, expand_refs=True, without_attrs=False)
                in_data.copy(source=in_data["/params"], dest=out_data["/"], shallow=False, expand_soft=True, expand_external=True, expand_refs=True, without_attrs=False)
                logger.info("Finished copying statistics")
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
        small_sizes = set()
        forcing_prefix = "traj00000_q_total_forcing_"
        for key in in_data["trajs"].keys():
            if key.startswith(forcing_prefix):
                small_sizes.add(int(key[len(forcing_prefix):]))
        small_sizes = sorted(small_sizes)
        # Base q data
        data_dtype_fields = [
            ("q", in_data["trajs"]["traj00000_q"].dtype, in_data["trajs"]["traj00000_q"].shape[1:])
        ]
        # System parameters
        for field in ["rek", "delta", "beta"]:
            data_dtype_fields.append((field, in_data["trajs"]["traj00000_sysparams"][field].dtype, (1, 1, 1)))
        for size in small_sizes:
            data_dtype_fields.append(
                (f"q_total_forcing_{size}", in_data["trajs"][f"traj00000_q_total_forcing_{size}"].dtype, in_data["trajs"][f"traj00000_q_total_forcing_{size}"].shape[1:])
            )
        data_dtype = np.dtype(data_dtype_fields)
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
    logger.info("Finished shuffling process")


if __name__ == "__main__":
    main()
