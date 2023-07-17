import threading
import dataclasses
import logging
import queue
import operator
import contextlib
import tempfile
import pathlib
import jax
import jax.numpy as jnp
import numpy as np
import random
import h5py
import zarr
import re
from . import kernel
from . import utils as qg_utils
from .qg_model import QGModel


__all__ = ["ThreadedQGLoader", "qg_model_from_hdf5"]


@dataclasses.dataclass
class CoreTrajData:
    t: np.ndarray
    tc: np.ndarray
    ablevel: np.ndarray


@kernel.register_dataclass_pytree
@dataclasses.dataclass
class PartialState:
    q: jnp.ndarray
    dqhdt_seq: jnp.ndarray
    t: jnp.ndarray
    tc: jnp.ndarray
    ablevel: jnp.ndarray
    q_total_forcings: dict[int, jnp.ndarray]
    sys_params: dict[str, object] = dataclasses.field(default_factory=dict)


def qg_model_from_hdf5(file_path, model="small"):
    with h5py.File(file_path, "r") as h5_file:
        params = h5_file["params"][f"{model}_model"].asstr()[()]
        return qg_utils.qg_model_from_param_json(params)


def _get_core_traj_data(file_path):
    with h5py.File(file_path, "r") as h5_file:
        t = h5_file["trajs"]["t"][:]
        tc = h5_file["trajs"]["tc"][:]
        ablevel = h5_file["trajs"]["ablevel"][:]
        return CoreTrajData(t=t, tc=tc, ablevel=ablevel)


class SimpleQGLoader:
    def __init__(self, file_path, fields=("q", "dqhdt", "t", "tc", "ablevel")):
        self._fields = sorted(set(fields))
        self._core_fields = [f for f in self._fields if f in {"t", "tc", "ablevel"}]
        self._non_core_fields = [f for f in self._fields if f not in {"t", "tc", "ablevel"}]
        self._h5_file = h5py.File(file_path, "r")
        self._trajs_group = self._h5_file["trajs"]
        self._data_fields = frozenset(f.name for f in dataclasses.fields(kernel.PseudoSpectralKernelState))
        num_traj = 0
        for k in self._trajs_group.keys():
            if k.startswith("traj") and k.endswith("_q"):
                num_traj += 1
        self.num_trajectories = num_traj
        self.num_steps = self._trajs_group["traj00000_q"].shape[0]
        self.num_trajs = self.num_trajectories
        self._core_traj_data = _get_core_traj_data(file_path) # Source for t, tc, ablevel

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
        result_fields = {k: None for k in ("q", "dqhdt_seq", "t", "tc", "ablevel")}
        result_fields["q_total_forcings"] = {}
        result_fields["sys_params"] = {}
        for field in self._core_fields:
            result_fields[field] = getattr(self._core_traj_data, field)[slicer]
        for field in self._non_core_fields:
            if field.startswith("q_total_forcing_"):
                # Handle forcings specially
                size = int(field[len("q_total_forcing_"):])
                result_fields["q_total_forcings"][size] = self._trajs_group[f"traj{traj:05d}_{field}"][slicer]
            elif field in {"rek", "delta", "beta"}:
                # This is a system parameter field
                result_fields["sys_params"][field] = np.full(
                    shape=(num_steps, 1, 1, 1),
                    fill_value=self._trajs_group[f"traj{traj:05d}_sysparams"][field][()].item(),
                    dtype=np.float32
                )
            else:
                result_fields[field if field != "dqhdt" else "dqhdt_seq"] = jax.device_put(
                    self._trajs_group[f"traj{traj:05d}_{field}"][slicer]
                )
        return jax.device_put(PartialState(**result_fields))


@kernel.register_dataclass_pytree
@dataclasses.dataclass
class SnapshotStates:
    q: jnp.ndarray = None
    q_total_forcings: dict[int, jnp.ndarray] = dataclasses.field(default_factory=dict)
    sys_params: dict[str, jnp.ndarray] = dataclasses.field(default_factory=dict)


class ThreadedPreShuffledSnapshotLoader:
    def __init__(
            self,
            file_path,
            batch_size,
            buffer_size=10,
            chunk_size=5425*2,
            seed=None,
            base_logger=None,
            fields=("q", "q_total_forcing_64"),
    ):
        if base_logger is None:
            base_logger = logging.getLogger("preshuffle_loader")
        # Note: Loads only q and q_total_forcing
        self.fields = sorted(set(fields))
        self.batch_size = batch_size
        if not all(f.startswith("q_total_forcing_") or f in {"q", "rek", "delta", "beta"} for f in self.fields):
            raise ValueError("invalid field, can only load q, q_total_forcing, and system parameters")
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**32)
        # Create queues
        self._chunk_load_queue = queue.Queue(maxsize=1)
        self._batch_queue = queue.Queue(maxsize=max(buffer_size, 1))
        self._stop_event = threading.Event()
        # Spawn threads
        self._chunk_load_thread = threading.Thread(
            target=self._load_chunks,
            daemon=True,
            kwargs={
                "file_path": file_path,
                "chunk_size": chunk_size,
                "chunk_load_queue": self._chunk_load_queue,
                "stop_event": self._stop_event,
                "fields": self.fields,
                "seed": seed,
                "logger": base_logger.getChild("chunk_loader"),
            },
        )
        self._batch_thread = threading.Thread(
            target=self._batch_chunks,
            daemon=True,
            kwargs={
                "chunk_load_queue": self._chunk_load_queue,
                "batch_queue": self._batch_queue,
                "batch_size": batch_size,
                "fields": self.fields,
                "stop_event": self._stop_event,
                "logger": base_logger.getChild("batcher"),
            },
        )
        # Compute number of samples
        with h5py.File(file_path, "r") as in_file:
            self._num_samples = operator.index(in_file["shuffled"].shape[0])
        self._chunk_load_thread.start()
        self._batch_thread.start()

    @staticmethod
    def _load_chunks(file_path, chunk_size, chunk_load_queue, stop_event, fields, seed, logger):
        logger.debug("starting chunk loader for size %d", chunk_size)

        def load_chunk(dataset, start, end):
            chunk = dataset[start:end]
            rng.shuffle(chunk, axis=0)
            return tuple(
                (
                    chunk[field].copy().astype(np.float32)
                    if field in {"rek", "delta", "beta"}
                    else chunk[field].copy()
                )
                for field in fields
            )

        try:
            rng = np.random.default_rng(seed=seed)
            with h5py.File(file_path, "r") as in_file:
                dataset = in_file["shuffled"]
                num_steps = dataset.shape[0]
                chunk_size = min(chunk_size, num_steps)
                valid_range = num_steps - chunk_size
                while not stop_event.is_set():
                    start = int(rng.integers(valid_range, dtype=np.uint64, endpoint=True).item())
                    end = start + chunk_size
                    logger.debug("loading chunk from %d to %d", start, end)
                    chunk_load_queue.put(
                        load_chunk(dataset, start, end)
                    )
        except Exception:
            logger.exception("error in background chunk loader")
            chunk_load_queue.put(None)
            raise
        finally:
            logger.debug("chunk loader exiting")

    @staticmethod
    def _batch_chunks(chunk_load_queue, batch_queue, batch_size, stop_event, fields, logger):
        logger.debug("starting batch producer")
        len_q_total_forcing = len("q_total_forcing_")

        def apportion_batches(construct_batch, chunk, cursor, remaining_steps):
            slicer = slice(cursor, cursor+remaining_steps)
            for dest, field_chunk in zip(construct_batch, chunk, strict=True):
                sliced = field_chunk[slicer]
                dest.append(sliced)
            return sliced.shape[0]

        def build_snapshot_states(fields, construct_batch):
            q = None
            q_total_forcings = {}
            sys_params = {}
            for field, field_stack in zip(fields, construct_batch, strict=True):
                if field.startswith("q_total_forcing_"):
                    q_total_forcings[int(field[len_q_total_forcing:])] = np.concatenate(field_stack, axis=0)
                elif field in {"rek", "delta", "beta"}:
                    # This is a system parameter field
                    sys_params[field] = np.concatenate(field_stack, axis=0)
                else:
                    # This must be q
                    q = np.concatenate(field_stack, axis=0)
                field_stack.clear()
            return SnapshotStates(
                q=q,
                q_total_forcings=q_total_forcings,
                sys_params=sys_params,
            )

        try:
            construct_batch = tuple([] for _ in fields)
            batch_steps = 0
            while not stop_event.is_set():
                # Get a chunk
                chunk = chunk_load_queue.get()
                cursor = 0
                if chunk is None:
                    # Time to exit
                    logger.debug("got None chunk, chunk loader exited")
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
                                build_snapshot_states(fields, construct_batch)
                            )
                        )
                        construct_batch = tuple([] for _ in fields)
                        batch_steps = 0
        except Exception:
            logger.exception("error in background batch producer")
            batch_queue.put(None)
            raise
        finally:
            logger.debug("batch producer exiting")

    def next_batch(self):
        if self._stop_event.is_set():
            raise ValueError("Closed dataset, cannot load batches")
        res = self._batch_queue.get()
        if res is None:
            self.close()
            raise RuntimeError("background worker stopped prematurely")
        return res

    def iter_batches(self):
        # A generator over the batches
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


class AggregateLoader:
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
        active_loaders = [loader for loader in self.loaders if loader.num_samples() > 0]
        weights = np.asarray([loader.num_samples() for loader in active_loaders])
        weights = weights / weights.sum()
        samples_per_loader = self._np_rng.multinomial(self.batch_size, weights)
        return jax.device_put(
            jax.tree_util.tree_map(
                lambda *loader_batches: jnp.concatenate(
                    [
                        AggregateLoader._slice_batch(batch, num_samples)
                        for num_samples, batch in zip(samples_per_loader, loader_batches, strict=True)
                    ]
                ),
                *(loader.next_batch() for loader in active_loaders),
            )
        )

    @staticmethod
    def _slice_batch(batch, num_samples):
        assert batch.shape[0] >= num_samples
        return batch[:num_samples]


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


class FillableDataLoader:
    def __init__(self, batch_size, fields=("q", "q_total_forcing_64"), seed=None):
        self.batch_size = batch_size
        self.fields = sorted(set(fields))
        self._temp_dir = tempfile.TemporaryDirectory(prefix="temp-fill-loader-")
        self._data = None
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**32)
        self._np_rng = np.random.default_rng(seed=seed)

    def _create_struct_dtype(self, sample):
        struct_dtype = []
        for field in self.fields:
            if field == "q":
                struct_dtype.append((field, sample.q.dtype, sample.q.shape[1:]))
            elif m := re.match(r"^q_total_forcing_(?P<size>\d+)$", field):
                size = int(m.group("size"))
                struct_dtype.append((field, sample.q_total_forcings[size].dtype, sample.q_total_forcings[size].shape[1:]))
            elif field in {"rek", "delta", "beta"}:
                struct_dtype.append((field, sample.sys_params[field].dtype, (1, 1, 1)))
            else:
                raise ValueError(f"unsupported field {field}")
        return np.dtype(struct_dtype)

    def _combine_array(self, sample, dtype):
        if sample.q is not None:
            num_steps = sample.q.shape[0]
        else:
            num_steps = next(iter(sample.q_total_forcings.values())).shape[0]
        arr = np.empty(shape=num_steps, dtype=dtype)
        for field in self.fields:
            if field == "q":
                value = sample.q
            elif m := re.match(r"^q_total_forcing_(?P<size>\d+)$", field):
                size = int(m.group("size"))
                value = m.q_total_forcings[size]
            elif field in {"rek", "delta", "beta"}:
                value = np.asarray(sample.sys_params[field]).item()
            else:
                raise ValueError(f"unsupported field {field}")
            arr[field] = value
        return arr

    def add_data(self, sample):
        if self._temp_dir is None:
            raise ValueError("Closed loader, cannot add data")
        if self._data is None:
            self._data = zarr.open_array(
                pathlib.Path(self._temp_dir.name) / "data.zarr",
                dtype=self._create_struct_dtype(sample),
                fill_value=None,
                shape=(0,),
                chunks=(25,),
                mode="w-",
                compressor=None,
            )
        self._data.append(self._combine_array(sample, self._data.dtype), axis=0)

    def num_samples(self):
        if self._data is None:
            return 0
        return self._data.shape[0]

    def next_batch(self):
        if self._temp_dir is None:
            raise ValueError("Closed loader, cannot load batches")
        num_samps = self.num_samples()
        if num_samps <= 0:
            raise ValueError("Empty fillable dataset, insert data first")
        indices = self._np_rng.integers(0, num_samps, size=self.batch_size)
        samples = self._data[indices]
        # Bundle samples into batch
        q = None
        q_total_forcings = {}
        sys_params = {}
        for field in self.fields:
            if field in {"rek", "delta", "beta"}:
                sys_params[field] = samples[field].astype(np.float32)
            elif field.startswith("q_total_forcing_"):
                size = int(field[16:])
                q_total_forcings[size] = samples[field]
            elif field == "q":
                q = samples["q"]
            else:
                raise ValueError(f"invalid field {field}")
        return jax.device_put(
            SnapshotStates(
                q=q,
                q_total_forcings=q_total_forcings,
                sys_params=sys_params,
            )
        )

    def iter_batches(self):
        # A generator over the batches
        while True:
            yield self.next_batch()

    def close(self):
        if self._temp_dir is None:
            return
        self._data = None
        self._temp_dir.cleanup()
        self._temp_dir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
