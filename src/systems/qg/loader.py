import threading
import dataclasses
import logging
import queue
import operator
import jax
import jax.numpy as jnp
import numpy as np
import h5py
from . import kernel


__all__ = ["ThreadedQGLoader"]


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
    logger.debug("worker thread started")
    rng = np.random.default_rng(seed=seed)
    per_traj_valid_steps = num_steps - rollout_steps
    total_valid_starts = (per_traj_valid_steps + 1) * num_trajs
    data_fields = frozenset(f.name for f in dataclasses.fields(kernel.PseudoSpectralKernelState))
    with h5py.File(file_path, "r") as h5_file:
        logger.debug("opened hdf5 file %s", file_path)
        trajs_group = h5_file["trajs"]
        while True:
            if stop_event.is_set():
                logger.debug("worker was signaled to quit")
                # We are signaled to stop
                return
            # Produce next samples
            samples = rng.integers(0, total_valid_starts, size=batch_size, dtype=np.uint64)
            batch_trajs, batch_steps = np.divmod(samples, per_traj_valid_steps + 1)
            # Load samples from the dataset
            np_dict = {k: [] for k in data_fields}
            for traj, step in zip(batch_trajs, batch_steps):
                traj_group = trajs_group[f"traj{traj:05d}"]
                for field in data_fields:
                    if field in {"dqhdt", "dqhdt_p", "dqhdt_pp"}:
                        # Special handling
                        continue
                    np_dict[field].append(traj_group[field][int(step):int(step + rollout_steps)])
                # Do special handling of dqhdt fields
                full_dqhdt_ds = traj_group["full_dqhdt"]
                dqhdt_and_p = full_dqhdt_ds[int(step+1):int(step+rollout_steps+1)]
                np_dict["dqhdt"].append(dqhdt_and_p)
                np_dict["dqhdt_p"].append(dqhdt_and_p)
                np_dict["dqhdt_pp"].append(full_dqhdt_ds[int(step):int(step + rollout_steps)])
            # Stack all samples
            np_dict = {k: np.stack(v) for k, v in np_dict.items()}
            # Move to device, pack and add to queue
            queue.put(kernel.PseudoSpectralKernelState(**{k: jax.device_put(v) for k, v in np_dict.items()}))


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
