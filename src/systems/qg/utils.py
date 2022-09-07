import functools
import dataclasses
import jax
import jax.numpy as jnp
from .kernel import PseudoSpectralKernelState


def make_gen_traj(big_model, spectral_coarsener):

    def do_steps(carry_state, _in):
        big_state = carry_state
        new_big_state = big_model.step_forward(big_state)
        # We output the *previous* state so the initial conditions get stacked
        big_out = spectral_coarsener.downsample_state(big_state)
        return new_big_state, big_out

    @functools.partial(jax.jit, static_argnums=[1])
    def gen_traj(rng, num_steps):
        big_init = big_model.create_initial_state(rng)
        _last_step, big_steps = jax.lax.scan(do_steps, big_init, None, length=num_steps)
        return big_steps

    return gen_traj


class SpectralCoarsener:
    def __init__(self, big_model, small_model):
        self.big_model = big_model
        self.small_model = small_model
        self.big_size = big_model.nx
        self.small_size = small_model.nx
        self._filtr = jnp.exp(-small_model.wv**2 * (2 * small_model.dx)**2 / 24)
        self._keep = self.small_size // 2
        self._int_scale = self.big_size // self.small_size
        if self.small_size >= self.big_size:
            raise ValueError(
                f"big size ({self.big_size}) must be larger than small ({self.small_size})"
            )
        if self.big_size % self.small_size != 0:
            raise ValueError(
                f"big size ({self.big_size}) does not divide small ({self.small_size})"
            )
        if self.small_size % 2 != 0:
            raise ValueError(f"small size ({self.small_size}) must be divisible by 2")

    def spectral_filter_spectral(self, var_fft):
        assert var_fft.shape[-1] != self.big_size
        assert var_fft.shape[-2] == self.big_size
        assert var_fft.shape[-3] == 2
        return jnp.hstack(
            [
                var_fft[:, :self._keep, :self._keep + 1],
                var_fft[:, -self._keep:, :self._keep + 1],
            ]
        ) * self._filtr / (self._int_scale) ** 2

    def spectral_filter_spatial(self, var):
        assert var.shape == (2, self.big_size, self.big_size)
        var_fft = jnp.fft.rfftn(var, axes=(-2, -1))
        out_var = self.spectral_filter_spectral(var_fft)
        return jnp.fft.irfftn(out_var, axes=(-2, -1))

    def downsample_state(self, state, replace_into=None):
        if replace_into is None:
            replace_into = state
        new_vals = {}
        scale_down = ["q", "u", "v", "uq", "vq"]
        spectral_scale_down = ["dqhdt", "dqhdt_p", "dqhdt_pp"]
        for name in scale_down:
            in_arr = getattr(state, name)
            a = self.spectral_filter_spatial(in_arr)
            new_vals[name] = a
        for name in spectral_scale_down:
            in_arr = getattr(state, name)
            a = self.spectral_filter_spectral(in_arr)
            new_vals[name] = a
        partial_state = dataclasses.replace(replace_into, **new_vals)
        new_vals["ph"] = self.small_model._apply_a_ph(partial_state)
        return dataclasses.replace(replace_into, **new_vals)


def slice_kernel_state(state, slicer):
    data_fields = frozenset(f.name for f in dataclasses.fields(PseudoSpectralKernelState))
    return PseudoSpectralKernelState(**{k: getattr(state, k)[slicer] for k in data_fields})


def get_online_rollout(start_state, num_steps, apply_fn, params, small_model, memory_init_fn, batch_stats, param_type="uv", train=False):
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
                (sx, sy, memory), stats = apply_fn(net_param, state.u, state.v, memory, train, mutable=["batch_stats"])
            else:
                sx, sy, memory = apply_fn(net_param, state.u, state.v, memory, train)
            return sx, sy

        def apply_net_q(state):
            nonlocal memory, stats
            net_param = {"params": params, "batch_stats": stats}
            if train:
                (dq, memory), stats = apply_fn(net_param, state.q, state.u, state.v, memory, train, mutable=["batch_stats"])
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

    (_last_step, last_memory, last_batch_stats), new_states = jax.lax.scan(
        do_steps,
        (start_state, memory, batch_stats),
        None,
        length=num_steps,
    )
    if train:
        return new_states, last_memory, last_batch_stats
    else:
        return new_states, last_memory


def get_online_batch_loss(real_traj, apply_fn, params, small_model, loss_fn, memory_init_fn, batch_stats, param_type="uv", train=False):
    first_step = slice_kernel_state(real_traj, 0)

    match param_type:
        case "uv":
            memory = memory_init_fn(params, start_state.u, start_state.v)
        case "q":
            memory = memory_init_fn(params, start_state.q, start_state.u, start_state.v)
        case _:
            raise ValueError(f"invalid parameterization type {param_type}")

    def do_steps(carry_state, true_step):
        old_state, memory, stats = carry_state

        def apply_net_uv(state):
            nonlocal memory, stats
            net_param = {"params": params, "batch_stats": stats}
            if train:
                (sx, sy, memory), stats = apply_fn(net_param, state.u, state.v, memory, train, mutable=["batch_stats"])
            else:
                sx, sy, memory = apply_fn(net_param, state.u, state.v, memory, train)
            return sx, sy

        def apply_net_q(state):
            nonlocal memory, stats
            net_param = {"params": params, "batch_stats": stats}
            if train:
                (dq, memory), stats = apply_fn(net_param, state.q, state.u, state.v, memory, train, mutable=["batch_stats"])
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

    (_last_step, _last_memory, last_batch_stats), losses = jax.lax.scan(
        do_steps,
        (first_step, memory, batch_stats),
        slice_kernel_state(real_traj, slice(1, None))
    )
    if train:
        return losses, last_batch_stats
    else:
        return losses
