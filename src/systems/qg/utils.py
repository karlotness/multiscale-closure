import functools
import dataclasses
import jax
import jax.numpy as jnp
from .kernel import PseudoSpectralKernelState


def make_gen_traj(model, num_steps):

    def do_steps(carry_state, _y):
        next_state = model.step_forward(carry_state)
        return next_state, carry_state

    def gen_traj(rng):
        init = model.create_initial_state(rng)
        _last_step, steps = jax.lax.scan(do_steps, init, None, length=num_steps)
        return steps

    return gen_traj


def slice_kernel_state(state, slicer):
    data_fields = frozenset(f.name for f in dataclasses.fields(PseudoSpectralKernelState))
    return PseudoSpectralKernelState(**{k: getattr(state, k)[slicer] for k in data_fields})


def get_online_rollout(start_state, num_steps, apply_fn, params, small_model, memory_init_fn, batch_stats, param_type="uv", train=False, scan_fn=jax.lax.scan):
    match param_type:
        case "uv":
            memory = memory_init_fn(params, start_state.u, start_state.v)
        case "q":
            memory = memory_init_fn(params, start_state.q, start_state.u, start_state.v)
        case _:
            raise ValueError(f"invalid parameterization type {param_type}")

    def do_steps(carry_state, _):
        old_state, memory, stats = carry_state

        def apply_net_uv(state):
            nonlocal memory, stats
            net_param = {"params": params, "batch_stats": stats}
            if train:
                (sx, sy, memory), new_stats = apply_fn(net_param, state.u, state.v, memory, train, mutable=["batch_stats"])
                stats = new_stats["batch_stats"]
            else:
                sx, sy, memory = apply_fn(net_param, state.u, state.v, memory, train)
            return sx, sy

        def apply_net_q(state):
            nonlocal memory, stats
            net_param = {"params": params, "batch_stats": stats}
            if train:
                (dq, memory), new_stats = apply_fn(net_param, state.q, state.u, state.v, memory, train, mutable=["batch_stats"])
                stats = new_stats["batch_stats"]
            else:
                dq, memory = apply_fn(net_param, state.q, state.u, state.v, memory, train)
            return dq

        match param_type:
            case "uv":
                kw = {"uv_param_func": apply_net_uv}
            case "q":
                kw = {"q_param_func": apply_net_q}
            case _:
                raise ValueError(f"invalid parameterization type {param_type}")

        new_state = small_model.step_forward(old_state, **kw)
        return (new_state, memory, stats), new_state

    (_last_step, last_memory, last_batch_stats), new_states = scan_fn(
        do_steps,
        (start_state, memory, batch_stats),
        None,
        length=num_steps,
    )
    if train:
        return new_states, last_memory, last_batch_stats
    else:
        return new_states, last_memory


def get_online_batch_loss(real_traj, apply_fn, params, small_model, loss_fn, memory_init_fn, batch_stats, param_type="uv", train=False, scan_fn=jax.lax.scan):
    first_step = slice_kernel_state(real_traj, 0)

    match param_type:
        case "uv":
            memory = memory_init_fn(params, first_step.u, first_step.v)
        case "q":
            memory = memory_init_fn(params, first_step.q, first_step.u, first_step.v)
        case _:
            raise ValueError(f"invalid parameterization type {param_type}")

    def do_steps(carry_state, true_step):
        old_state, memory, stats = carry_state

        def apply_net_uv(state):
            nonlocal memory, stats
            net_param = {"params": params, "batch_stats": stats}
            if train:
                (sx, sy, memory), new_stats = apply_fn(net_param, state.u, state.v, memory, train, mutable=["batch_stats"])
                stats = new_stats["batch_stats"]
            else:
                sx, sy, memory = apply_fn(net_param, state.u, state.v, memory, train)
            return sx, sy

        def apply_net_q(state):
            nonlocal memory, stats
            net_param = {"params": params, "batch_stats": stats}
            if train:
                (dq, memory), new_stats = apply_fn(net_param, state.q, state.u, state.v, memory, train, mutable=["batch_stats"])
                stats = new_stats["batch_stats"]
            else:
                dq, memory = apply_fn(net_param, state.q, state.u, state.v, memory, train)
            return dq

        match param_type:
            case "uv":
                kw = {"uv_param_func": apply_net_uv}
            case "q":
                kw = {"q_param_func": apply_net_q}
            case _:
                raise ValueError(f"invalid parameterization type {param_type}")

        new_state = small_model.step_forward(old_state, **kw)
        loss = loss_fn(real_step=true_step.q, est_step=new_state.q)
        return (new_state, memory, stats), loss

    (_last_step, _last_memory, last_batch_stats), losses = scan_fn(
        do_steps,
        (first_step, memory, batch_stats),
        slice_kernel_state(real_traj, slice(1, None)),
    )
    if train:
        return losses, last_batch_stats
    else:
        return losses
