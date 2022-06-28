import threading
import sys
import dataclasses
import pathlib
import logging
import queue
import operator
import asyncio
import contextlib
import jax
import jax.numpy as jnp
import numpy as np
import h5py
from . import kernel
from .qg_model import QGModel


__all__ = ["ThreadedQGLoader", "qg_model_from_hdf5"]


_WORK_LOOP = None
_WORK_LOOP_MUTEX = threading.Lock()
_PENDING_TASKS = set()


def qg_model_from_hdf5(file_path, model="small"):
    with h5py.File(file_path, "r") as h5_file:
        params = h5_file["params"][f"{model}_model"].asstr()[()]
        return QGModel.from_param_json(params)


def _get_series_details(file_path, rollout_steps):
    with h5py.File(file_path, "r") as h5_file:
        arr = h5_file["trajs"]["traj00000"][:int(rollout_steps)]
        return arr.dtype, arr.nbytes


def _get_work_loop():
    global _WORK_LOOP

    def _start_loop(loop, logger):
        try:
            asyncio.set_event_loop(loop)
            loop.run_forever()
        except Exception:
            logger.exception("event loop quit unexpectedly")
            raise
        finally:
            with _WORK_LOOP_MUTEX:
                _WORK_LOOP = None
            try:
                # Cancel tasks
                to_cancel = asyncio.all_tasks(loop)
                if to_cancel:
                    for task in to_cancel:
                        task.cancel()
                    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))
                # Shut down everything else
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    with _WORK_LOOP_MUTEX:
        if _WORK_LOOP is None:
            _WORK_LOOP = asyncio.new_event_loop()
            thread = threading.Thread(
                target=_start_loop,
                kwargs={
                    "loop": _WORK_LOOP,
                    "logger": logging.getLogger("loader_event_loop"),
                },
                name="loader_worker",
                daemon=True,
            )
            thread.start()
        return _WORK_LOOP


def _create_work_loop_cond(loop):

    async def _create_cond():
        return asyncio.Condition()

    future = asyncio.run_coroutine_threadsafe(_create_cond(), loop)
    return future.result()


async def _signal_condition_cb(cond):
    async with cond:
        cond.notify()

def _add_signal_task(cond):
    task = asyncio.create_task(_signal_condition_cb(cond))
    _PENDING_TASKS.add(task)
    task.add_done_callback(_PENDING_TASKS.discard)

def _signal_work_loop_cond(loop, cond):
    loop.call_soon_threadsafe(_add_signal_task, cond)

async def _proc_worker_task(
        in_queue,
        out_queue,
        logger,
        file_path,
        rollout_steps,
        arr_dtype,
        arr_byte_size,
):
    logger.debug("process worker started")
    try:
        # Spawn worker's process
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(pathlib.Path(__file__).parent / "_loader.py"),
            str(file_path),
            f"{rollout_steps:d}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        logger.debug("spawned subprocess")
        try:
            while True:
                job = await in_queue.get()
                if job is None:
                    # This is our stop signal
                    logger.debug("process worker signaled to stop")
                    return
                i, traj, step = job
                proc.stdin.write(f"{traj:d} {step:d}\n".encode("utf8"))
                await proc.stdin.drain()
                await out_queue.put((i, np.frombuffer(await proc.stdout.readexactly(arr_byte_size), dtype=arr_dtype)))
        finally:
            proc.stdin.write_eof()
            return_code = await proc.wait()
            if return_code != 0:
                logger.error("worker exited abnormally with code %d", return_code)
            else:
                logger.debug("stopped worker process")
    except Exception:
        logger.exception("error in process worker")
        raise
    finally:
        # Signal our exit
        await out_queue.put(None)

async def _worker_coro(
        out_queue,
        logger,
        file_path,
        stop_event,
        batch_size,
        rollout_steps,
        seed,
        num_trajs,
        num_steps,
        arr_dtype,
        arr_byte_size,
        queue_wait_cond,
        num_procs=10,
):
    try:
        logger.debug("worker thread started")
        submit_queue = asyncio.Queue()
        recieve_queue = asyncio.Queue()
        sort_key_func = operator.itemgetter(0)
        undecorate_func = operator.itemgetter(1)
        sub_tasks = {
            asyncio.create_task(
                _proc_worker_task(
                    in_queue=submit_queue,
                    out_queue=recieve_queue,
                    logger=logger.getChild(f"proc{i}"),
                    file_path=file_path,
                    rollout_steps=rollout_steps,
                    arr_dtype=arr_dtype,
                    arr_byte_size=arr_byte_size,
                )
            )
            for i in range(num_procs)
        }
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
                for i, (traj, step) in enumerate(zip(batch_trajs, batch_steps, strict=True)):
                    await submit_queue.put((i, traj, step))
                # Retrieve results
                arr_stack = []
                for _i in range(batch_size):
                    res = await recieve_queue.get()
                    if not res:
                        logger.error("worker died prematurely, exiting")
                        return
                    arr_stack.append(res)
                arr_stack.sort(key=sort_key_func)
                # Stack and split arrays, push to GPU and place in queue
                arr_stack = np.stack(list(map(undecorate_func, arr_stack)))
                out_result = jax.device_put(kernel.PseudoSpectralKernelState(**{k: arr_stack[k] for k in data_fields}))
                async with queue_wait_cond:
                    while True:
                        try:
                            out_queue.put_nowait(out_result)
                            break
                        except queue.Full:
                            await queue_wait_cond.wait()
        except Exception:
            logger.exception("exception inside worker thread, terminating worker and processes")
            raise
        finally:
            # Signal our workers to stop
            for _i in range(num_procs):
                await submit_queue.put(None)
            await asyncio.gather(*sub_tasks, return_exceptions=True)
            logger.debug("finished stopping all workers")
    except Exception:
        logger.exception("exception inside worker thread, terminating worker")
        raise
    finally:
        # Signal that we've exited by placing None in the queue
        async with queue_wait_cond:
            while True:
                try:
                    out_queue.put_nowait(None)
                    break
                except queue.Full:
                    await queue_wait_cond.wait()


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
            num_steps = h5_file["trajs"]["traj00000"].shape[0]
            num_trajs = 0
            for k in h5_file["trajs"].keys():
                if k.startswith("traj"):
                    num_trajs += 1
        self._logger.info("loading dataset with %d trajectories of %d steps each", num_trajs, num_steps)
        self.num_trajs = num_trajs
        self.num_steps = num_steps
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self._arr_dtype, self._arr_byte_size = _get_series_details(
            file_path=file_path,
            rollout_steps=self.rollout_steps
        )

        self._work_loop = _get_work_loop()
        self._queue_wait_cond = _create_work_loop_cond(self._work_loop)
        self._worker_future = asyncio.run_coroutine_threadsafe(
            _worker_coro(
                out_queue=self._queue,
                logger=self._logger.getChild("worker"),
                file_path=file_path,
                stop_event=self._stop_event,
                batch_size=batch_size,
                rollout_steps=rollout_steps,
                seed=seed,
                num_trajs=num_trajs,
                num_steps=num_steps,
                arr_dtype=self._arr_dtype,
                arr_byte_size=self._arr_byte_size,
                queue_wait_cond=self._queue_wait_cond,
                num_procs=num_workers,
            ),
            self._work_loop
        )

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
        res = self._queue.get()
        _signal_work_loop_cond(self._work_loop, self._queue_wait_cond)
        if res is None:
            self._logger.error("background worker stopped prematurely")
            self.close()
            raise RuntimeError("background worker stopped prematurely")
        return res

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
                _signal_work_loop_cond(self._work_loop, self._queue_wait_cond)
        except queue.Empty:
            # Ignore the exception now that queue is empty
            pass
        # Join the worker
        self._worker_future.result()


class SimpleQGLoader:
    def __init__(self, file_path):
        self._h5_file = h5py.File(file_path, "r")
        self._trajs_group = self._h5_file["trajs"]
        self._data_fields = frozenset(f.name for f in dataclasses.fields(kernel.PseudoSpectralKernelState))
        num_traj = 0
        for k in self._trajs_group.keys():
            if k.startswith("traj"):
                num_traj += 1
        self.num_trajectories = num_traj
        self.num_steps = self._trajs_group["traj00000"].shape[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        if self._trajs_group is not None:
            self._trajs_group = None
            self._h5_file.close()

    def get_trajectory(self, traj, start=0, end=None):
        start = operator.index(start)
        if end is not None:
            end = operator.index(end)
        traj_data = self._trajs_group[f"traj{traj:05d}"][start:end]
        return jax.device_put(kernel.PseudoSpectralKernelState(**{k: traj_data[k] for k in self._data_fields}))
