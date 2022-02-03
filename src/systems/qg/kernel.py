import jax
import jax.numpy as jnp
import dataclasses


def attach_to_object(self):
    def decorator(func):
        setattr(self, func.__name__, func)
        return func
    return decorator


DTYPE_COMPLEX = jnp.complex64
DTYPE_REAL = jnp.float32


def register_dataclass_pytree(cls):
    fields = tuple(f.name for f in dataclasses.fields(cls))

    def flatten(self, obj):
        return [getattr(obj, name) for name in fields], None

    def unflatten(self, aux_data, flat_contents):
        args = {name: value in zip(fields, flat_contents)}
        return cls(**args)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls


@register_dataclass_pytree
@dataclasses.dataclass
class PseudoSpectralKernelState:
    # Array state parameters
    a: jnp.ndarray
    kk: jnp.ndarray
    _ik: jnp.ndarray
    ll: jnp.ndarray
    _il: jnp.ndarray
    _k2l2: jnp.ndarray
    # FFT inputs & outputs
    q: jnp.ndarray
    qh: jnp.ndarray
    ph: jnp.ndarray
    u: jnp.ndarray
    uh: jnp.ndarray
    v: jnp.ndarray
    vh: jnp.ndarray
    uq: jnp.ndarray
    uqh: jnp.ndarray
    vq: jnp.ndarray
    vqh: jnp.ndarray
    # has_uv_param
    du: jnp.ndarray
    dv: jnp.ndarray
    duh: jnp.ndarray
    dvh: jnp.ndarray
    # has_q_param
    dq: jnp.ndarray
    dqh: jnp.ndarray
    # Time stuff
    t: float
    tc: int
    ablevel: int
    # The tendency
    dqhdt: jnp.ndarray
    dqhdt_p: jnp.ndarray
    dqdhdt_pp: jnp.ndarray
    # Other state vars
    Ubg: jnp.ndarray
    Qy: jnp.ndarray
    _ikQy: jnp.ndarray


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
ifft_qh_to_q = _generic_irfftn
ifft_uh_to_u = _generic_irfftn
ifft_vh_to_v = _generic_irfftn
fft_du_to_duh = _generic_rfftn
fft_dv_to_dvh = _generic_rfftn
fft_dq_to_dqh = _generic_rfftn
fft_uq_to_uqh = _generic_rfftn
fft_vq_to_vqh = _generic_rfftn


class PseudoSpectralKernel:

    def __init__(self, nz, ny, nx, dt, filtr, has_q_param=False, has_uv_param=False, rek=0):
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.nl = ny
        self.nk = (nx // 2) + 1
        assert nx == ny
        self.has_q_param = has_q_param
        self.has_uv_param = has_uv_param
        # Friction
        self.rek = rek
        self.dt = float(dt)
        self.filtr = filtr

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
        def do_uv_subgrid_parameterization(state, du, dv):
            return self._do_uv_subgrid_parameterization(state, du, dv)

        @attach_to_object(self)
        def do_q_subgrid_parameterization(state, dq):
            return self._do_q_subgrid_parameterization(state, dq)

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
        def set_kk(state, new_kk):
            kk = new_kk
            _ik = 1j * new_kk
            _k2l2 = (jnp.expand_dims(new_kk, 0)**2) + (jnp.expand_dims(state.ll, -1)**2)
            return _update_state(state, kk=kk, _ik=_ik, _k2l2=_k2l2)

        @attach_to_object(self)
        def set_ll(state, new_ll):
            ll = new_ll
            _il = 1j * new_ll
            _k2l2 = (jnp.expand_dims(new_kk, 0)**2) + (jnp.expand_dims(state.ll, -1)**2)
            return _update_state(state, ll=ll, _il=_il, _k2l2=_k2l2)

        @attach_to_object(self)
        def set_a(state, new_a):
            return _update_state(state, a=b.astype(DTYPE_COMPLEX))

        @attach_to_object(self)
        def set_Ubg(state, new_Ubg):
            return _update_state(state, Ubg=new_Ubg)

        @attach_to_object(self)
        def set_Qy(state, new_Qy):
            _ikQy = 1j * (jnp.expand_dims(state.kk, 0) * jnp.expand_dims(new_Qy, -1))
            return _update_state(state, Qy=new_Qy, _ikQy=_ikQy)

        @attach_to_object(self)
        def set_q(state, new_q):
            qh = fft_q_to_qh(new_q)
            return _update_state(state, q=new_q, qh=qh)

        @attach_to_object(self)
        def set_qh(state, new_qh):
            q = ifft_qh_to_q(new_qh)
            return _update_state(state, qh=new_qh, q=q)

    def _create_initial_state(self):
        def _empty_real():
            return jnp.zeros((self.nz, self.ny, self.nx), dtype=DTYPE_REAL)
        def _empty_com():
            return jnp.zeros((self.nz, self.nl, self.nk), dtype=DTYPE_COMPLEX)

        du = None
        dv = None
        duh = None
        dvh = None
        if self.has_uv_param:
            du = _empty_real()
            dv = _empty_real()
            duh = _empty_com()
            dvh = _empty_com()

        dq = None
        dqh = None
        if self.has_q_param:
            dq = _empty_real()
            dqh = _empty_com()

        Ubg = jnp.zeros((self.nk,), dtype=DTYPE_REAL)
        Qy = jnp.zeros((self.nk,), dtype=DTYPE_REAL)

        new_state = PseudoSpectralKernelState(
            # State arrays
            a=jnp.zeros((self.nz, self.nz, self.nl, self.nk), dtype=DTYPE_COMPLEX),
            kk=jnp.zeros((self.nk,), dtype=DTYPE_REAL),
            _ik = jnp.zeros((self.nk,), dtype=DTYPE_COMPLEX),
            ll = jnp.zeros((self.nl,), dtype=DTYPE_REAL),
            _il = jnp.zeros((self.nl,), dtype=DTYPE_COMPLEX),
            _k2l2 = jnp.zeros((self.nl, self.nk), dtype=DTYPE_REAL),
            # FFT I/O
            q = _empty_real(),
            qh = _empty_com(),
            ph = _empty_com(),
            u = _empty_real(),
            uh = _empty_com(),
            v = _empty_real(),
            vh = _empty_com(),
            uq = _empty_real(),
            uqh = _empty_com(),
            vq = _empty_real(),
            vqh = _empty_com(),
            # if has_uv_param
            du=du,
            dv=dv,
            duh=duh,
            dvh=dvh,
            # if has_q_param
            dq=dq,
            dqh=dqh,
            # Time stuff
            t = 0.0,
            tc = 0,
            ablevel = 0,
            # The tendency
            dqhdt = _empty_com(),
            dqhdt_p = _empty_com(),
            dqdhdt_pp = _empty_com(),
            # Other parameters
            Ubg=Ubg,
            Qy=Qy,
            _ikQy=None,
        )
        new_state = self.set_Ubg(new_state, Ubg)
        new_state = self.set_Qy(new_state, Qy)
        return new_state

    def _invert(self, state):
        # Set ph to zero (skip, recompute fresh from sum below)
        # invert qh to find ph
        ph = jnp.sum(state.a * jnp.expand_dims(state.qh, 0), axis=1)
        # calculate spectral velocities
        uh = (-1 * jnp.expand_dims(state._il, (0, -1))) * ph
        vh = jnp.expand_dims(state._ik, (0, 1)) * ph
        # transform to get u and v
        u = ifft_uh_to_u(uh)
        v = ifft_vh_to_v(vh)
        # Update state values
        return _update_state(state, ph=ph, uh=uh, vh=vh, u=u, v=v)

    def _do_advection(self, state):
        # multiply to get advective flux in space
        uq = (state.u + jnp.expand_dims(state.Ubg[:self.nz], (-1, -2))) * state.q
        vq = state.v  * state.q
        # transform to get spectral advective flux
        uqh = fft_uq_to_uqh(uq)
        vqh = fft_vq_to_vqh(vq)
        # spectral divergence
        dqhdt = -1 * (jnp.expand_dims(state._ik, (0, 1)) * uqh +
                      jnp.expand_dims(state._il, (0, -1)) * vqh +
                      jnp.expand_dims(state._ikQy[:self.nz], 1) * state.ph)
        return _update_state(state, uq=uq, vq=vq, uqh=uqh, vqh=vqh, dqhdt=dqhdt)

    def _do_uv_subgrid_parameterization(self, state, du, dv):
        # convert to spectral space
        duh = fft_du_to_duh(du)
        dvh = fft_dv_to_dvh(dv)
        qdhdt = (
            state.dqhdt +
            ((-1 * jnp.expand_dims(state._il, (0, -1))) * duh) +
            (jnp.expand_dims(state._ik, (0, 1)) * dvh)
        )
        return _update_state(state, du=du, dv=dv, duh=duh, dvh=dvh, dqhdt=dqhdt)

    def _do_q_subgrid_parameterization(self, state, dq):
        dqh = fft_dq_to_dqh(dq)
        dqhdt = state.dqhdt + dqh
        return _update_state(state, dq=dq, dqh=dqh, dqhdt=dqhdt)

    def _do_friction(self, state):
        # Apply Beckman friction to lower layer tendency
        k = self.nz - 1
        if self.rek:
            dqhdt = state.dqhdt.at[k].set(
                state.dqhdt[k] + (self.rek * state._k2l2 * state.ph[k])
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

        return _update_state(state, ablevel=ablevel, qh=qh, dqhdt_pp=dqhdt_pp, dqhdt_p=dqhdt_p, q=q, tc=tc, t=t)
