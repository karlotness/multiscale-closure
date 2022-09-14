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


@dataclasses.dataclass
class CoreTrajData:
    t: jnp.ndarray
    tc: jnp.ndarray
    ablevel: jnp.ndarray


def qg_model_from_hdf5(file_path, model="small"):
    with h5py.File(file_path, "r") as h5_file:
        params = h5_file["params"][f"{model}_model"].asstr()[()]
        return QGModel.from_param_json(params)


def _get_series_details(file_path, rollout_steps):
    with h5py.File(file_path, "r") as h5_file:
        q_arr = h5_file["trajs"]["traj00000_q"][:operator.index(rollout_steps)]
        dqhdt_arr = h5_file["trajs"]["traj00000_dqhdt"][:operator.index(rollout_steps) + 2]
        return q_arr.dtype, dqhdt_arr.dtype, q_arr.nbytes, dqhdt_arr.nbytes, q_arr.shape[1:], dqhdt_arr.shape[1:]


def _get_core_traj_data(file_path):
    with h5py.File(file_path, "r") as h5_file:
        t = jax.device_put(h5_file["trajs"]["t"][:])
        tc = jax.device_put(h5_file["trajs"]["tc"][:])
        ablevel = jax.device_put(h5_file["trajs"]["ablevel"][:])
        return CoreTrajData(t=t, tc=tc, ablevel=ablevel)


def _make_small_model_recomputer(small_model):
    def recompute_1(q, t, tc, ablevel):
        small_state = small_model.create_initial_state(jax.random.PRNGKey(0))
        small_state.t = t
        small_state.tc = tc
        small_state.ablevel = ablevel
        small_state.q = q
        # Initialize other values
        small_state = small_model.invert(small_state) # Recompute ph, u, v
        small_state = small_model.do_advection(small_state) # Recompute uq, vq, dqhdt
        small_state = small_model.do_friction(small_state) # Recompute dqhdt
        return small_state

    def recompute(q, dqhdt, t, tc, ablevel):
        small_state = jax.vmap(recompute_1)(q, t, tc, ablevel)
        # Fix up dqhdt, etc
        small_state.dqhdt = dqhdt[2:]
        small_state.dqhdt_p = dqhdt[1:-1]
        small_state.dqhdt_pp = dqhdt[:-2]
        return small_state

    return recompute


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
                    for task in to_cancel:
                        if task.cancelled():
                            continue
                        if task.exception() is not None:
                            logger.warning("event loop shutdown found a task %r with unhandled exception %r", task, task.exception())
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
        q_dtype,
        dqhdt_dtype,
        q_byte_size,
        dqhdt_byte_size,
        q_shape,
        dqhdt_shape,
        core_traj_data,
):
    logger.debug("process worker started")
    q_shape = (rollout_steps, ) + q_shape
    dqhdt_shape = (rollout_steps + 2, ) + dqhdt_shape
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
                # Prepare sliced core data
                slicer = slice(operator.index(step), operator.index(step) + operator.index(rollout_steps))
                t = core_traj_data.t[slicer]
                tc = core_traj_data.tc[slicer]
                ablevel = core_traj_data.ablevel[slicer]
                # Get bytes from worker (q, dqhdt)
                q = np.frombuffer(await proc.stdout.readexactly(q_byte_size), dtype=q_dtype).reshape(q_shape)
                dqhdt = np.frombuffer(await proc.stdout.readexactly(dqhdt_byte_size), dtype=dqhdt_dtype).reshape(dqhdt_shape)
                # Put results
                await out_queue.put((i, q, dqhdt, t, tc, ablevel))
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
        q_dtype,
        dqhdt_dtype,
        q_byte_size,
        dqhdt_byte_size,
        q_shape,
        dqhdt_shape,
        queue_wait_cond,
        num_procs=10,
):
    try:
        logger.debug("worker thread started")
        core_traj_data = _get_core_traj_data(file_path) # Source for t, tc, ablevel
        small_model = qg_model_from_hdf5(file_path=file_path, model="small") # Source for recomputing other values from q
        state_recomputer = jax.jit(jax.vmap(_make_small_model_recomputer(small_model)))
        submit_queue = asyncio.Queue()
        recieve_queue = asyncio.Queue()
        sort_key_func = operator.itemgetter(0)
        undecorate_q_func = operator.itemgetter(1)
        undecorate_dqhdt_func = operator.itemgetter(2)
        undecorate_t_func = operator.itemgetter(3)
        undecorate_tc_func = operator.itemgetter(4)
        undecorate_ablevel_func = operator.itemgetter(5)
        sub_tasks = {
            asyncio.create_task(
                _proc_worker_task(
                    in_queue=submit_queue,
                    out_queue=recieve_queue,
                    logger=logger.getChild(f"proc{i}"),
                    file_path=file_path,
                    rollout_steps=rollout_steps,
                    q_dtype=q_dtype,
                    dqhdt_dtype=dqhdt_dtype,
                    q_byte_size=q_byte_size,
                    dqhdt_byte_size=dqhdt_byte_size,
                    q_shape=q_shape,
                    dqhdt_shape=dqhdt_shape,
                    core_traj_data=core_traj_data,
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
                q_stack = np.stack(list(map(undecorate_q_func, arr_stack)))
                dqhdt_stack = np.stack(list(map(undecorate_dqhdt_func, arr_stack)))
                t_stack = jnp.stack(list(map(undecorate_t_func, arr_stack)))
                tc_stack = jnp.stack(list(map(undecorate_tc_func, arr_stack)))
                ablevel_stack = jnp.stack(list(map(undecorate_ablevel_func, arr_stack)))
                out_result = state_recomputer(q_stack, dqhdt_stack, t_stack, tc_stack, ablevel_stack)
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
            num_steps = h5_file["trajs"]["traj00000_q"].shape[0]
            num_trajs = 0
            for k in h5_file["trajs"].keys():
                if k.startswith("traj") and k.endswith("_q"):
                    num_trajs += 1
        self._logger.info("loading dataset with %d trajectories of %d steps each", num_trajs, num_steps)
        self.num_trajs = num_trajs
        self.num_steps = num_steps
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self._q_dtype, self._dqhdt_dtype, self._q_byte_size, self._dqhdt_byte_size, self._q_shape, self._dqhdt_shape = _get_series_details(
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
                batch_size=self.batch_size,
                rollout_steps=self.rollout_steps,
                seed=seed,
                num_trajs=self.num_trajs,
                num_steps=self.num_steps,
                q_dtype=self._q_dtype,
                dqhdt_dtype=self._dqhdt_dtype,
                q_byte_size=self._q_byte_size,
                dqhdt_byte_size=self._dqhdt_byte_size,
                q_shape=self._q_shape,
                dqhdt_shape=self._dqhdt_shape,
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
            if k.startswith("traj") and k.endswith("_q"):
                num_traj += 1
        _, _, _, _, q_shape, dqhdt_shape = _get_series_details(
            file_path=file_path,
            rollout_steps=1,
        )
        self.q_shape = q_shape[1:]
        self.dqhdt_shape = dqhdt_shape[1:]
        self.num_trajectories = num_traj
        self.num_steps = self._trajs_group["traj00000_q"].shape[0]
        self.num_trajs = self.num_trajectories
        small_model = qg_model_from_hdf5(file_path=file_path, model="small") # Source for recomputing other values from q
        self._state_recomputer = jax.jit(_make_small_model_recomputer(small_model))
        self._core_traj_data = _get_core_traj_data(file_path) # Source for t, tc, ablevel

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
        slicer = slice(start, end)
        slicer_dqhdt = slice(start, end + 2 if end is not None else None)
        t = self._core_traj_data.t[slicer]
        tc = self._core_traj_data.tc[slicer]
        ablevel = self._core_traj_data.ablevel[slicer]
        q = self._trajs_group[f"traj{traj:05d}_q"][slicer]
        dqhdt = self._trajs_group[f"traj{traj:05d}_dqhdt"][slicer_dqhdt]
        return self._state_recomputer(q, dqhdt, t, tc, ablevel)
