import jax
import jax.numpy as jnp
import dataclasses
import json


def attach_to_object(self):
    def decorator(func):
        setattr(self, func.__name__, func)
        return func
    return decorator


DTYPE_COMPLEX = jnp.complex64
DTYPE_REAL = jnp.float32


def register_dataclass_pytree(cls):
    fields = tuple(f.name for f in dataclasses.fields(cls))

    def flatten(obj):
        return [getattr(obj, name) for name in fields], None

    def unflatten(aux_data, flat_contents):
        args = {name: value for name, value in zip(fields, flat_contents)}
        return cls(**args)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls


@register_dataclass_pytree
@dataclasses.dataclass
class PseudoSpectralKernelState:
    # FFT inputs & outputs
    q: jnp.ndarray
    ph: jnp.ndarray
    u: jnp.ndarray
    v: jnp.ndarray
    uq: jnp.ndarray
    vq: jnp.ndarray
    # Time stuff
    t: float
    tc: int
    ablevel: int
    # The tendency
    dqhdt: jnp.ndarray
    dqhdt_p: jnp.ndarray
    dqhdt_pp: jnp.ndarray

    @property
    def qh(self):
        return fft_q_to_qh(self.q)

    @qh.setter
    def set_qh(self, qh):
        self.q = ifft_qh_to_q(qh)

    @property
    def uh(self):
        return fft_u_to_uh(self.u)

    @uh.setter
    def set_uh(self, uh):
        self.u = ifft_uh_to_u(uh)

    @property
    def vh(self):
        return fft_v_to_vh(self.v)

    @vh.setter
    def set_vh(self, vh):
        self.v = ifft_vh_to_v(vh)

    @property
    def uqh(self):
        return fft_uq_to_uqh(self.uq)

    @uqh.setter
    def set_uqh(self, uqh):
        self.uq = ifft_uqh_to_uq(uqh)

    @property
    def vqh(self):
        return fft_vq_to_vqh(self.vq)

    @vqh.setter
    def set_vqh(self, vqh):
        self.vq = ifft_vqh_to_vq(vqh)


def _update_state(old_state, **kwargs):
    for k, v in kwargs.items():
        old_val = getattr(old_state, k)
        if hasattr(old_val, "shape"):
            assert old_val.shape == v.shape, f"Shape mismatch on {k}: {old_val.shape} vs {v.shape}"
    return dataclasses.replace(old_state, **kwargs)


def _generic_rfftn(a):
    return jnp.fft.rfftn(a, axes=(-2, -1))

def _generic_irfftn(a):
    return jnp.fft.irfftn(a, axes=(-2, -1))


fft_q_to_qh = _generic_rfftn
fft_u_to_uh = _generic_rfftn
fft_v_to_vh = _generic_rfftn
ifft_qh_to_q = _generic_irfftn
ifft_uh_to_u = _generic_irfftn
ifft_vh_to_v = _generic_irfftn
fft_du_to_duh = _generic_rfftn
fft_dv_to_dvh = _generic_rfftn
fft_dq_to_dqh = _generic_rfftn
fft_uq_to_uqh = _generic_rfftn
ifft_uqh_to_uq = _generic_irfftn
fft_vq_to_vqh = _generic_rfftn
ifft_vqh_to_vq = _generic_irfftn


class PseudoSpectralKernel:

    def __init__(self, nz, ny, nx, dt, filtr, rek=0):
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.nl = ny
        self.nk = (nx // 2) + 1
        self.kk = jnp.zeros((self.nk), dtype=DTYPE_REAL)
        self._ik = jnp.zeros((self.nk), dtype=DTYPE_COMPLEX)
        self.ll = jnp.zeros((self.nl), dtype=DTYPE_REAL)
        self._il = jnp.zeros((self.nl), dtype=DTYPE_COMPLEX)
        self._k2l2 = jnp.zeros((self.nl, self.nk), dtype=DTYPE_REAL)
        assert nx == ny
        # Friction
        self.rek = rek
        self.dt = float(dt)
        self.filtr = filtr
        self.Ubg = jnp.zeros((self.nk,), dtype=DTYPE_REAL)

        @attach_to_object(self)
        def create_initial_state():
            return self._create_initial_state()

        @attach_to_object(self)
        def fft(v):
            # v - 3dim array
            assert v.ndim == 3
            assert isinstance(v.dtype, jnp.floating)
            return jnp.fft.rfftn(v, axes=(-2, -1))

        @attach_to_object(self)
        def ifft(v):
            # v - 3dim array
            assert v.ndim == 3
            assert isinstance(v.dtype, jnp.complexfloating)
            return jnp.fft.irfftn(v, axes=(-2, -1))

        @attach_to_object(self)
        def invert(state):
            return self._invert(state)

        @attach_to_object(self)
        def do_advection(state):
            return self._do_advection(state)

        @attach_to_object(self)
        def do_uv_subgrid_parameterization(state, uv_param_func=None):
            return self._do_uv_subgrid_parameterization(state, uv_param_func)

        @attach_to_object(self)
        def do_q_subgrid_parameterization(state, q_param_func=None):
            return self._do_q_subgrid_parameterization(state, q_param_func)

        @attach_to_object(self)
        def do_friction(state):
            return self._do_friction(state)

        @attach_to_object(self)
        def forward_timestep(state):
            return self._forward_timestep(state)

        @attach_to_object(self)
        def set_dt(state, new_dt):
            return _update_state(state, dt=new_dt, ablevel=0)

        @attach_to_object(self)
        def set_q(state, new_q):
            return _update_state(state, q=new_q)

        @attach_to_object(self)
        def set_qh(state, new_qh):
            q = ifft_qh_to_q(new_qh)
            return _update_state(state, q=q)

    def _create_initial_state(self):
        def _empty_real():
            return jnp.zeros((self.nz, self.ny, self.nx), dtype=DTYPE_REAL)
        def _empty_com():
            return jnp.zeros((self.nz, self.nl, self.nk), dtype=DTYPE_COMPLEX)

        new_state = PseudoSpectralKernelState(
            # FFT I/O
            q = _empty_real(),
            ph = _empty_com(),
            u = _empty_real(),
            v = _empty_real(),
            uq = _empty_real(),
            vq = _empty_real(),
            # Time stuff
            t = 0.0,
            tc = 0,
            ablevel = 0,
            # The tendency
            dqhdt = _empty_com(),
            dqhdt_p = _empty_com(),
            dqhdt_pp = _empty_com(),
        )
        return new_state

    def _apply_a_ph(self, state):
        a = jnp.zeros((self.nz, self.nz, self.nl, self.nk), dtype=DTYPE_COMPLEX)
        ph = jnp.sum(a * jnp.expand_dims(state.qh, 0), axis=1)
        return ph

    def _invert(self, state):
        # Set ph to zero (skip, recompute fresh from sum below)
        # invert qh to find ph
        ph = self._apply_a_ph(state)
        # calculate spectral velocities
        uh = (-1 * jnp.expand_dims(self._il, (0, -1))) * ph
        vh = jnp.expand_dims(self._ik, (0, 1)) * ph
        # transform to get u and v
        u = ifft_uh_to_u(uh)
        v = ifft_vh_to_v(vh)
        # Update state values
        return _update_state(state, ph=ph, u=u, v=v)

    def _do_advection(self, state):
        # multiply to get advective flux in space
        uq = (state.u + jnp.expand_dims(self.Ubg[:self.nz], (-1, -2))) * state.q
        vq = state.v  * state.q
        # transform to get spectral advective flux
        uqh = fft_uq_to_uqh(uq)
        vqh = fft_vq_to_vqh(vq)
        # spectral divergence
        dqhdt = -1 * (jnp.expand_dims(self._ik, (0, 1)) * uqh +
                      jnp.expand_dims(self._il, (0, -1)) * vqh +
                      jnp.expand_dims(self._ikQy[:self.nz], 1) * state.ph)
        return _update_state(state, uq=uq, vq=vq, dqhdt=dqhdt)

    def _do_uv_subgrid_parameterization(self, state, uv_param_func):
        if uv_param_func is None:
            return state
        # convert to spectral space
        du, dv = uv_param_func(state)
        duh = fft_du_to_duh(du)
        dvh = fft_dv_to_dvh(dv)
        dqhdt = (
            state.dqhdt +
            ((-1 * jnp.expand_dims(self._il, (0, -1))) * duh) +
            (jnp.expand_dims(self._ik, (0, 1)) * dvh)
        )
        return _update_state(state, dqhdt=dqhdt)

    def _do_q_subgrid_parameterization(self, state, q_param_func):
        if q_param_func is None:
            return state
        dq = q_param_func(state)
        dqh = fft_dq_to_dqh(dq)
        dqhdt = state.dqhdt + dqh
        return _update_state(state, dqhdt=dqhdt)

    def _do_friction(self, state):
        # Apply Beckman friction to lower layer tendency
        k = self.nz - 1
        if self.rek:
            dqhdt = state.dqhdt.at[k].set(
                state.dqhdt[k] + (self.rek * self._k2l2 * state.ph[k])
            )
            return _update_state(state, dqhdt=dqhdt)
        else:
            return state

    def _forward_timestep(self, state):

        ablevel, dt1, dt2, dt3 = jax.lax.switch(
            state.ablevel,
            [
                lambda: (1, self.dt, 0.0, 0.0),
                lambda: (2, 1.5 * self.dt, -0.5 * self.dt, 0.0),
                lambda: (2, (23 / 12) * self.dt, (-16 / 12) * self.dt, (5 / 12) * self.dt),
            ]
        )

        qh_new = jnp.expand_dims(self.filtr, 0) * (
            state.qh +
            dt1 * state.dqhdt +
            dt2 * state.dqhdt_p +
            dt3 * state.dqhdt_pp
        )
        qh = qh_new
        dqhdt_pp = state.dqhdt_p
        dqhdt_p = state.dqhdt

        # do FFT of new qh
        q = ifft_qh_to_q(qh)

        # Update time tracking parameters
        tc = state.tc + 1
        t = state.t + self.dt

        return _update_state(state, ablevel=ablevel, dqhdt_pp=dqhdt_pp, dqhdt_p=dqhdt_p, q=q, tc=tc, t=t)

    def param_json(self):
        return json.dumps(
            {
                "nz": self.nz,
                "ny": self.ny,
                "nx": self.nx,
                "dt": self.dt,
                "filtr": self.filtr.to_py().tolist() if self.filter is not None else None,
                "rek": self.rek,
            }
        )

    @classmethod
    def from_param_json(cls, param_str):
        params = json.loads(param_str)
        if params["filtr"] is not None:
            params["filtr"] = jnp.asarray(params["filtr"])
        return cls(**params)
