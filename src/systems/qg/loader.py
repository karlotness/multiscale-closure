import threading
import sys
import dataclasses
import pathlib
import logging
import queue
import operator
import subprocess
import contextlib
import jax
import jax.numpy as jnp
import numpy as np
import h5py
from . import kernel
from .qg_model import QGModel


__all__ = ["ThreadedQGLoader", "qg_model_from_hdf5"]


def qg_model_from_hdf5(file_path, model="small"):
    with h5py.File(file_path, "r") as h5_file:
        params = h5_file["params"][f"{model}_model"].asstr()[()]
        return QGModel.from_param_json(params)


def _get_series_details(file_path, rollout_steps):
    with h5py.File(file_path, "r") as h5_file:
        arr = h5_file["trajs"]["traj00000"][:int(rollout_steps)]
        return arr.dtype, arr.nbytes


def _proc_worker_func(
        in_queue,
        out_queue,
        logger,
        file_path,
        rollout_steps,
        arr_dtype,
        arr_byte_size,
):
    logger.debug("process worker started")
    # Spawn worker's process
    proc = subprocess.Popen(
                [
                    sys.executable,
                    str(pathlib.Path(__file__).parent / "_loader.py"),
                    str(file_path),
                    f"{rollout_steps:d}",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
    logger.debug("spawned subprocess")
    try:
        while True:
            job = in_queue.get()
            if job is None:
                # This is our stop signal
                logger.debug("process worker signaled to stop")
                return
            traj, step = job
            proc.stdin.write(f"{traj:d} {step:d}\n".encode("utf8"))
            proc.stdin.flush()
            text = proc.stdout.readline()
            if text != b"done\n":
                raise ValueError(f"got invalid result from worker {text}")
            out_queue.put(np.frombuffer(proc.stdout.read(arr_byte_size), dtype=arr_dtype))
    except Exception:
        logger.exception("error in process worker")
        raise
    finally:
        proc.stdin.write(b"exit\n")
        proc.stdin.flush()
        return_code = proc.wait()
        if return_code != 0:
            logger.error("worker exited abnormally with code %d", return_code)
        else:
            logger.debug("stopped worker process")

def _worker_func(
        out_queue,
        logger,
        file_path,
        stop_event,
        batch_size,
        rollout_steps,
        seed,
        num_trajs,
        num_steps,
        num_procs=10,
):
    try:
        logger.debug("worker thread started")
        arr_dtype, arr_byte_size = _get_series_details(file_path=file_path, rollout_steps=rollout_steps)
        submit_queue = queue.SimpleQueue()
        recieve_queue = queue.SimpleQueue()
        sub_threads = [
            threading.Thread(
                target=_proc_worker_func,
                kwargs={
                    "in_queue": submit_queue,
                    "out_queue": recieve_queue,
                    "logger": logger.getChild(f"proc{i}"),
                    "file_path": file_path,
                    "rollout_steps": rollout_steps,
                    "arr_dtype": arr_dtype,
                    "arr_byte_size": arr_byte_size,
                },
                name=f"loader_proc_worker_{i:d}",
                daemon=True,
            )
            for i in range(num_procs)
        ]
        for thread in sub_threads:
            thread.start()
        try:
            # Compute needed information
            rng = np.random.default_rng(seed=seed)
            per_traj_valid_steps = num_steps - rollout_steps
            total_valid_starts = (per_traj_valid_steps + 1) * num_trajs
            data_fields = frozenset(f.name for f in dataclasses.fields(kernel.PseudoSpectralKernelState))
            while True:
                if stop_event.is_set():
                    logger.debug("worker was signaled to quit")
                    return
                # Continuing with data load
                samples = rng.integers(0, total_valid_starts, size=batch_size, dtype=np.uint64)
                batch_trajs, batch_steps = np.divmod(samples, per_traj_valid_steps + 1)
                # Assign each process a trajectory
                for traj, step in zip(batch_trajs, batch_steps, strict=True):
                    submit_queue.put((traj, step))
                # Retrieve results
                arr_stack = []
                for _i in range(batch_size):
                    arr_stack.append(recieve_queue.get())
                # Stack and split arrays, push to GPU and place in queue
                arr_stack = np.stack(arr_stack)
                out_queue.put(kernel.PseudoSpectralKernelState(**{k: jax.device_put(arr_stack[k]) for k in data_fields}))
        except Exception:
            logger.exception("exception inside worker thread, terminating worker and processes")
            raise
        finally:
            # Signal our workers to stop
            for _i in range(num_procs):
                submit_queue.put(None)
            for thread in sub_threads:
                thread.join()
            logger.debug("finished stopping all workers")
    except Exception:
        logger.exception("exception inside worker thread, terminating worker")
        raise


class ThreadedQGLoader:
    def __init__(
            self,
            file_path,
            batch_size,
            rollout_steps,
            split_name=None,
            base_logger=None,
            buffer_size=1,
            seed=None,
            num_workers=10,
    ):
        self.split_name = split_name
        if split_name is not None:
            logger_name = f"qgloader_{split_name}"
        else:
            logger_name = "qgloader"
        if base_logger is not None:
            self._logger = base_logger.getChild(logger_name)
        else:
            self._logger = logging.getLogger(logger_name)
        if operator.index(buffer_size) < 1:
            raise ValueError(f"invalid buffer size {buffer_size}, must be at least 1")
        self._queue = queue.Queue(maxsize=operator.index(buffer_size))
        self._stop_event = threading.Event()

        # Determine some basic statistics
        with h5py.File(file_path, "r") as h5_file:
            num_steps = h5_file["trajs"]["traj00000"]["q"].shape[0]
            num_trajs = 0
            for k in h5_file["trajs"].keys():
                if k.startswith("traj"):
                    num_trajs += 1
        self._logger.info("loading dataset with %d trajectories of %d steps each", num_trajs, num_steps)
        self.num_trajs = num_trajs
        self.num_steps = num_steps
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size

        self._worker_thread = threading.Thread(
            target=_worker_func,
            kwargs={
                "out_queue": self._queue,
                "logger": self._logger.getChild("worker"),
                "file_path": file_path,
                "stop_event": self._stop_event,
                "batch_size": batch_size,
                "rollout_steps": rollout_steps,
                "seed": seed,
                "num_trajs": num_trajs,
                "num_steps": num_steps,
                "num_procs": num_workers,
            },
            name=f"{logger_name}_worker",
            daemon=True,
        )
        self._worker_thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if not self._stop_event.is_set():
            # We didn't close before, salvage things
            self.close()

    def next_batch(self):
        if self._stop_event.is_set():
            raise ValueError("Closed dataset, cannot load batches")
        return self._queue.get()

    def iter_batches(self):
        # A generator over the batches
        while True:
            yield self.next_batch()

    def close(self):
        self._stop_event.set()
        # Consume all items in the queue
        # The worker will then notice the flag
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            # Ignore the exception now that queue is empty
            pass
        self._worker_thread.join()
