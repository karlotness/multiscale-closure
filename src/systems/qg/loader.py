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


# Create the worker thread
def _worker_func(
        queue,
        logger,
        file_path,
        stop_event,
        batch_size,
        rollout_steps,
        seed,
        num_trajs,
        num_steps,
):
    try:
        logger.debug("worker thread started")
        num_procs = batch_size
        arr_dtype, arr_byte_size = _get_series_details(file_path=file_path, rollout_steps=rollout_steps)
        # Spawn processes
        procs = [
            subprocess.Popen(
                [
                    sys.executable,
                    str(pathlib.Path(__file__).parent / "_loader.py"),
                    str(file_path),
                    f"{rollout_steps:d}",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            for _ in range(num_procs)
        ]
        logger.debug("spawned %d processes", num_procs)
        # Compute needed information
        rng = np.random.default_rng(seed=seed)
        per_traj_valid_steps = num_steps - rollout_steps
        total_valid_starts = (per_traj_valid_steps + 1) * num_trajs
        data_fields = frozenset(f.name for f in dataclasses.fields(kernel.PseudoSpectralKernelState))
        while True:
            if stop_event.is_set():
                logger.debug("worker was signaled to quit")
                # Ask all processes to stop
                for proc in procs:
                    proc.stdin.write(b"exit\n")
                    proc.stdin.flush()
                    return_code = proc.wait()
                    if return_code != 0:
                        logger.error("worker exited abnormally with code %d", return_code)
                logger.debug("done stopping workers, exiting")
                return
            # Continuing with data load
            samples = rng.integers(0, total_valid_starts, size=batch_size, dtype=np.uint64)
            batch_trajs, batch_steps = np.divmod(samples, per_traj_valid_steps + 1)
            # Assign each process a trajectory
            for proc, traj, step in zip(procs, batch_trajs, batch_steps, strict=True):
                proc.stdin.write(f"{traj:d} {step:d}\n".encode("utf8"))
                proc.stdin.flush()
            arr_stack = []
            # For each process, wait until it's ready then read the result
            for proc in procs:
                text = proc.stdout.readline()
                if text != b"done\n":
                    raise ValueError(f"got invalid result from worker {text}")
                arr_stack.append(np.frombuffer(proc.stdout.read(arr_byte_size), dtype=arr_dtype))
            # Stack and split arrays, push to GPU and place in queue
            arr_stack = np.stack(arr_stack)
            queue.put(kernel.PseudoSpectralKernelState(**{k: jax.device_put(arr_stack[k]) for k in data_fields}))
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
                "queue": self._queue,
                "logger": self._logger.getChild("worker"),
                "file_path": file_path,
                "stop_event": self._stop_event,
                "batch_size": batch_size,
                "rollout_steps": rollout_steps,
                "seed": seed,
                "num_trajs": num_trajs,
                "num_steps": num_steps,
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
