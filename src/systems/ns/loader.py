import dataclasses
import random
import logging
import queue
import re
import operator
import threading
import contextlib
import numpy as np
import jax
import jax.numpy as jnp
import h5py
from ..qg.utils import register_pytree_dataclass



@register_pytree_dataclass
@dataclasses.dataclass
class ParamStat:
    mean: float
    var: float


@register_pytree_dataclass
@dataclasses.dataclass
class CoreSystemParams:
    u_offset: jnp.ndarray
    v_offset: jnp.ndarray
    u_corr_offset: jnp.ndarray
    v_corr_offset: jnp.ndarray
    u_stats: ParamStat
    v_stats: ParamStat
    u_corr_stats: ParamStat
    v_corr_stats: ParamStat

    def field_stats(self, field: str) -> ParamStat:
        match field:
            case "u":
                return self.u_stats
            case "v":
                return self.v_stats
            case "u_corr":
                return self.u_corr_stats
            case "v_corr":
                return self.v_corr_stats
            case _:
                raise ValueError(f"Invalid field {field}")

    def field_offsets(self, field: str) -> jnp.ndarray:
        match field:
            case "u":
                return self.u_offset
            case "v":
                return self.v_offset
            case "u_corr":
                return self.u_corr_offset
            case "v_corr":
                return self.v_corr_offset
            case _:
                raise ValueError(f"Invalid field {field}")


@register_pytree_dataclass
@dataclasses.dataclass
class LoadedState:
    u: jnp.ndarray | None
    v: jnp.ndarray | None
    u_corr: jnp.ndarray | None
    v_corr: jnp.ndarray | None

    def field(self, field: str) -> jnp.ndarray:
        match field:
            case "u":
                val = self.u
            case "v":
                val = self.v
            case "u_corr":
                val = self.u_corr
            case "v_corr":
                val = self.v_corr
            case _:
                raise ValueError(f"Invalid field {field}")
        if val is None:
            raise ValueError(f"Missing field {field} (was not loaded)")
        return val


class NSThreadedPreShuffledSnapshotLoader:
    def __init__(
        self,
        file_path,
        batch_size,
        buffer_size=10,
        chunk_size=5425*2,
        seed=None,
        base_logger=None,
        fields=("ns_u_64", "ns_v_64", "ns_u_corr_64", "ns_v_corr_64"),
    ):
        if base_logger is None:
            self._logger = logging.getLogger("ns_preshuffle_loader")
        else:
            self._logger = base_logger.getChild("ns_preshuffle_loader")
        self.fields = sorted(set(fields))
        self.batch_size = batch_size
        sizes = set()
        load_fields = set()
        # Check the fields and determine size
        for field in self.fields:
            if m := re.fullmatch(r"ns_(?P<basename>.+?)_(?P<size>\d+)", field):
                sizes.add(int(m.group("size")))
                base_name = m.group("basename")
                if base_name not in {"u", "v", "u_corr", "v_corr"}:
                    raise ValueError(f"Requested invalid field {base_name} not in file")
                load_fields.add(base_name)
            else:
                raise ValueError(f"invalid field {field}")
        if len(sizes) != 1:
            raise ValueError(f"Requested loading inconsistent sizes {fields}")
        self.size = sizes.pop()
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**32)
        # Create communication queues
        self._chunk_load_queue = queue.Queue(maxsize=1)
        self._batch_queue = queue.Queue(maxsize=max(buffer_size, 1))
        self._stop_event = threading.Event()
        # Spawn our threads
        self._chunk_load_thread = threading.Thread(
            target=self._load_chunks,
            daemon=True,
            kwargs={
                "file_path": file_path,
                "chunk_size": chunk_size,
                "chunk_load_queue": self._chunk_load_queue,
                "stop_event": self._stop_event,
                "size": self.size,
                "fields": load_fields,
                "seed": seed,
                "logger": self._logger.getChild("chunk_loader"),
            },
        )
        self._batch_thread = threading.Thread(
            target=self._batch_chunks,
            daemon=True,
            kwargs={
                "chunk_load_queue": self._chunk_load_queue,
                "batch_queue": self._batch_queue,
                "batch_size": batch_size,
                "stop_event": self._stop_event,
                "fields": load_fields,
                "logger": self._logger.getChild("batcher"),
            },
        )
        # Compute number of samples
        with h5py.File(file_path, "r") as in_file:
            self._num_samples = operator.index(in_file[f"sz{self.size}"]["shuffled"].shape[0])
            # Check system type
            if "model_type" not in in_file.keys() or in_file["model_type"].asstr()[()] != "ns":
                raise ValueError("provided dataset is not for NS system")
        # Start the threads
        self._chunk_load_thread.start()
        self._batch_thread.start()


    @staticmethod
    def _load_chunks(file_path, chunk_size, chunk_load_queue, stop_event, size, fields, seed, logger):
        logger.debug("Starting chunk loader for chunk_size %d", chunk_size)
        rng = np.random.default_rng(seed=seed)

        def load_chunk(dataset, start, end):
            chunk = dataset[start:end]
            rng.shuffle(chunk, axis=0)
            return tuple(
                chunk[field].copy().astype(np.float32)
                for field in fields
            )

        try:
            with h5py.File(file_path, "r") as in_file:
                dataset = in_file[f"sz{size}"]["shuffled"]
                num_steps = dataset.shape[0]
                chunk_size = min(chunk_size, num_steps)
                valid_range = num_steps - chunk_size
                while not stop_event.is_set():
                    start = int(rng.integers(valid_range, dtype=np.uint64, endpoint=True).item())
                    end = start + chunk_size
                    logger.debug("Loading chunk from %d to %d", start, end)
                    chunk_load_queue.put(load_chunk(dataset, start, end))
        except Exception:
            logger.exception("Error in background chunk loader")
            chunk_load_queue.put(None)
            raise
        finally:
            logger.debug("Chunk loader exiting")

    @staticmethod
    def _batch_chunks(chunk_load_queue, batch_queue, batch_size, stop_event, fields, logger):
        logger.debug("Starting batch producer")

        def apportion_batches(construct_batch, chunk, cursor, remaining_steps):
            slicer = slice(cursor, cursor + remaining_steps)
            for dest, field_chunk in zip(construct_batch, chunk, strict=True):
                sliced = field_chunk[slicer]
                dest.append(sliced)
            return sliced.shape[0]

        def build_snapshot_states(fields, non_fields, construct_batch):
            args = {}
            for field, field_stack in zip(fields, construct_batch, strict=True):
                args[field] = np.concatenate(field_stack, axis=0)
                field_stack.clear()
            for field in non_fields:
                args[field] = None
            return LoadedState(**args)

        try:
            non_fields = {"u", "v", "u_corr", "v_corr"} - set(fields)
            construct_batch = tuple([] for _ in fields)
            batch_steps = 0
            while not stop_event.is_set():
                # Get a chunk
                chunk = chunk_load_queue.get()
                cursor = 0
                if chunk is None:
                    # Time to exit
                    logger.debug("Got a None chunk, chunk loader exited")
                    break
                chunk_size = chunk[0].shape[0]
                while cursor < chunk_size and not stop_event.is_set():
                    # Keep consuming from this chunk as long as we can
                    remaining_steps = batch_size - batch_steps
                    consumed = apportion_batches(construct_batch, chunk, cursor, remaining_steps)
                    cursor += consumed
                    batch_steps += consumed
                    if batch_steps >= batch_size:
                        # A new batch is ready
                        batch_queue.put(
                            jax.device_put(
                                build_snapshot_states(fields, non_fields, construct_batch)
                            )
                        )
                        construct_batch = tuple([] for _ in fields)
                        batch_steps = 0
        except Exception:
            logger.exception("Error in background batch producer")
            batch_queue.put(None)
            raise
        finally:
            logger.debug("Batch producer exiting")

    def next_batch(self):
        if self._stop_event.is_set():
            raise ValueError("closed dataset, cannot load batches")
        res = self._batch_queue.get()
        if res is None:
            self.close()
            raise RuntimeError("background worker stopped prematurely")
        return res

    def iter_batches(self):
        while True:
            yield self.next_batch()

    def num_samples(self):
        return self._num_samples

    def close(self):
        if self._stop_event.is_set():
            # Already cleaned up
            return
        self._stop_event.set()
        # Stop the chunk load thread first
        with contextlib.suppress(queue.Empty):
            while True:
                # Clear its queue
                self._chunk_load_queue.get_nowait()
        self._chunk_load_thread.join()
        # Now that it is stopped, clear the queue again and place a None
        # This will ensure the batch thread exits if it hasn't already
        with contextlib.suppress(queue.Empty):
            while True:
                self._chunk_load_queue.get_nowait()
        self._chunk_load_queue.put(None)
        # Next, stop the batch thread
        with contextlib.suppress(queue.Empty):
            while True:
                self._batch_queue.get_nowait()
        # Join the second thread
        self._batch_thread.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


class SimpleNSLoader:
    def __init__(self, file_path, fields=("ns_u_64", "ns_v_64", "ns_u_corr_64", "ns_v_corr_64")):
        self.fields = sorted(set(fields))
        sizes = set()
        load_fields = set()
        self._h5_file = h5py.File(file_path, "r")
        # Check the fields and determine size
        for field in self.fields:
            if m := re.fullmatch(r"ns_(?P<basename>.+?)_(?P<size>\d+)", field):
                sizes.add(int(m.group("size")))
                base_name = m.group("basename")
                if base_name not in {"u", "v", "u_corr", "v_corr"}:
                    raise ValueError(f"Requested invalid field {base_name} not in file")
                load_fields.add(base_name)
            else:
                raise ValueError(f"invalid field {field}")
        self._load_fields = load_fields
        self._non_fields = {"u", "v", "u_corr", "v_corr"} - set(self._load_fields)
        if len(sizes) != 1:
            raise ValueError(f"Requested loading inconsistent sizes {fields}")
        self.size = sizes.pop()
        self._trajs_group = self._h5_file[f"sz{self.size}"]["trajs"]
        num_traj = 0
        for k in self._trajs_group.keys():
            if k.startswith("traj") and k.endswith("_u_corr"):
                num_traj += 1
        self.num_trajectories = num_traj
        self.num_steps = self._trajs_group["traj00000_u"].shape[0]
        self.num_trajs = self.num_trajectories
        if "model_type" not in self._h5_file.keys() or self._h5_file["model_type"].asstr()[()] != "ns":
            raise ValueError("provided dataset is not for NS system")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def num_samples(self):
        return self.num_steps * self.num_trajs

    def close(self):
        if self._trajs_group is not None:
            self._trajs_group = None
            self._h5_file.close()

    def get_trajectory(self, traj, start=0, end=None):
        start = operator.index(start)
        if end is not None:
            end = operator.index(end)
        slicer = slice(start, end)
        idx_start, idx_stop, _ = slicer.indices(self.num_steps)
        num_steps = idx_stop - idx_start
        result_fields = {k: None for k in self._non_fields}
        for field in self._load_fields:
            result_fields[field] = self._trajs_group[f"traj{traj:05d}_{field}"][slicer].astype(np.float32)
        return jax.device_put(LoadedState(**result_fields))


class NSAggregateLoader:
    def __init__(self, loaders, batch_size, seed=None):
        self.loaders = tuple(loaders)
        self.batch_size = batch_size
        self._closed = False
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**32)
        self._np_rng = np.random.default_rng(seed=seed)
        if any(loader.batch_size < self.batch_size for loader in self.loaders):
            raise ValueError("Batch size too small for aggregation")

    def num_samples(self):
        return sum(l.num_samples() for l in self.loaders)

    def next_batch(self):
        if self._closed:
            raise ValueError("Closed loader, cannot load batches")
        active_loaders = []
        weights = []
        for loader in self.loaders:
            num_samps = loader.num_samples()
            if num_samps > 0:
                active_loaders.append(loader)
                weights.append(num_samps)
        weights = np.asarray([loader.num_samples() for loader in active_loaders])
        weights = weights / weights.sum()
        samples_per_loader = self._np_rng.multinomial(self.batch_size, weights)
        batches = [loader.next_batch() for loader in active_loaders]
        extension = len(self.loaders) - len(batches)
        batches.extend([None] * extension)
        samples_per_loader = np.concatenate([samples_per_loader, np.zeros(extension, dtype=samples_per_loader.dtype)], axis=0)
        return batches, samples_per_loader

    def iter_batches(self):
        # A generator over the batches
        while True:
            yield self.next_batch()

    def close(self):
        if self._closed:
            return
        for loader in self.loaders:
            loader.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
