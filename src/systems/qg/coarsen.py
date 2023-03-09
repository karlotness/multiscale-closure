import jax
import jax.numpy as jnp
import pyqg_jax
from . import utils


def _generic_rfftn(a):
    return jnp.fft.rfftn(a, axes=(-2, -1))


def _generic_irfftn(a):
    return jnp.fft.irfftn(a, axes=(-2, -1))


def coarsen_model(big_model, small_nx):
    assert big_model.nx == big_model.ny
    assert small_nx < big_model.nx
    assert small_nx % 2 == 0
    model_args = utils.qg_model_to_args(big_model)
    model_args["nx"] = small_nx
    model_args["ny"] = small_ny
    return pyqg_jax.qg_model.QGModel(**model_args)


def compute_q_total_forcing(big_state, coarsener):
    small_state = coarsener.coarsen_state(big_state)
    big_full_state = coarsener.big_model.get_full_state(big_state)
    small_full_state = coarsener.small_model.get_full_state(small_state)
    big_dqhdt = coarsener.coarsen(big_full_state.dqhdt)
    small_dqhdt = small_full_state.dqhdt
    return _generic_irfftn(big_dqhdt) - _generic_irfftn(small_dqhdt)


def _divergence(fx, fy, model):
    ddx = _generic_irfftn(_generic_rfftn(fx) * model.ik)
    ddy = _generic_irfftn(_generic_rfftn(fy) * model.il)
    return ddx + ddy


def _advect(var, u, v, model):
    return _divergence(var * u, var * v, model)


def compute_q_subgrid_forcing(big_state, coarsener):
    small_state = coarsener.coarsen_state(big_state)
    big_full_state = coarsener.big_model.get_full_state(big_state)
    small_full_state = coarsener.small_model.get_full_state(small_state)
    term1 = _advect(small_full_state.q, small_full_state.u, small_full_state.v, coarsener.small_model)
    term2 = _advect(big_full_state.q, big_full_state.u, big_full_state.v, coarsener.big_model)
    term2 = coarsener.coarsen(term2)
    return term1 - term2


class Coarsener:
    def __init__(self, big_model, small_nx):
        self.big_model = big_model
        self.small_model = coarsen_model(self.big_model, small_nx)
        self.ratio = self.big_model.nx / self.small_model.nx

    def coarsen_state(self, big_state):
        small_qh = self.coarsen(big_state.qh)
        return pyqg_jax.state.PseudoSpectralState(qh=small_qh)

    def uncoarsen_state(self, small_state):
        big_qh = self.uncoarsen(small_state.qh)
        return pyqg_jax.state.PseudoSpectralState(qh=big_qh)

    def coarsen(self, var):
        raise NotImplementedError("implement in a subclass")

    def uncoarsen(self, var):
        raise NotImplementedError("implement in a subclass")

    def _to_spec(self, q):
        return _generic_rfftn(q)

    def _to_real(self, q):
        return _generic_irfftn(q)

    def _is_spectral(self, var):
        return var.dtype in {jnp.dtype(t) for t in (jnp.complex64, jnp.complex128)}


class NoOpCoarsener(Coarsener):
    def __init__(self, big_model, small_nx):
        # Unusual: don't call super() init, handle everything here
        assert big_model.nx == big_model.ny
        assert small_nx == big_model.nx
        assert small_nx % 2 == 0
        self.big_model = big_model
        self.small_model = big_model
        self.ratio = 1.0

    def coarsen(self, var):
        assert self._is_spectral(var) or var.shape[-2:] == (self.big_model.ny, self.big_model.nx)
        return var

    def uncoarsen(self, var):
        return self.coarsen(var)


class SpectralCoarsener(Coarsener):
    def coarsen(self, var):
        assert self._is_spectral(var) or var.shape[-2:] == (self.big_model.ny, self.big_model.nx)
        assert var.ndim == 3
        dummy_varh = jax.eval_shape(
            self._to_spec,
            jnp.zeros(
                (self.small_model.nz, self.small_model.ny, self.small_model.nx),
                dtype=jnp.float32,
            )
        )
        nk = dummy_varh.shape[1] // 2
        if not self._is_spectral(var):
            vh = self._to_spec(var)
        else:
            vh = var
        trunc = jnp.hstack((vh[:, :nk,:nk+1],
                            vh[:,-nk:,:nk+1]))
        filtered = trunc * self.spectral_filter() / self.ratio**2
        if not self._is_spectral(var):
            return self._to_real(filtered)
        else:
            return filtered

    def uncoarsen(self, var):
        # Steps:
        # 1. compute target size
        # 2. compute needed zeros
        # 3. concatenate resulting signal
        assert self._is_spectral(var) or var.shape[-2:] == (self.small_model.ny, self.small_model.nx)
        assert var.ndim == 3
        dummy_big_varh = jax.eval_shape(
            self._to_spec,
            jnp.zeros(
                (self.big_model.nz, self.big_model.ny, self.big_model.nx),
                dtype=jnp.float32
            ),
        )
        big_nk = dummy_big_varh.shape[1] // 2
        if not self._is_spectral(var):
            vh = self._to_spec(var)
        else:
            vh = var
        # Do our spectral scaling operation
        nk = vh.shape[-2] // 2
        row_pad_shape = vh.shape[:-2] + (dummy_big_varh.shape[-2] - vh.shape[-2], vh.shape[-1])
        col_pad_shape = dummy_big_varh.shape[:-1] + (dummy_big_varh.shape[-1] - vh.shape[-1], )
        # Unscale vh (at least by the ratio, not by self.spectral_filter)
        vh = vh * self.ratio**2
        # Pad back missing values
        untrunc = jnp.concatenate(
            [
                vh[..., :, :nk, :],
                jnp.zeros_like(vh, shape=row_pad_shape),
                vh[..., :, -nk:, :],
            ],
            axis=-2,
        )
        untrunc = jnp.concatenate([untrunc, jnp.zeros_like(vh, shape=col_pad_shape)], axis=-1)
        if not self._is_spectral(var):
            return self._to_real(untrunc)
        else:
            return untrunc

    def spectral_filter(self):
        raise NotImplementedError("implement in a subclass")


class Operator1(SpectralCoarsener):
    def spectral_filter(self):
        return self.small_model.filtr


class Operator2(SpectralCoarsener):
    def spectral_filter(self):
        return jnp.exp(-self.small_model.wv**2 * (2*self.small_model.dx)**2 / 24)


class BasicSpectralCoarsener(SpectralCoarsener):
    def spectral_filter(self):
        return 1


class LinearResizeCoarsener(Coarsener):
    def __init__(self, big_model, small_nx):
        super().__init__(big_model=big_model, small_nx=small_nx)

    def coarsen(self, var):
        if self._is_spectral(var):
            var_spatial = self._to_real(var)
        else:
            var_spatial = var
        assert var_spatial.ndim == 3
        assert var_spatial.shape[-2:] == (self.big_model.ny, self.big_model.nx)
        result_shape = (var_spatial.shape[-3],) + (self.small_model.ny, self.small_model.nx)
        resized = jax.image.resize(
            var_spatial,
            shape=result_shape,
            method="linear",
            antialias=False,
        )
        if self._is_spectral(var):
            return self._to_spec(resized)
        else:
            return resized

    def uncoarsen(self, var):
        if self._is_spectral(var):
            var_spatial = self._to_real(var)
        else:
            var_spatial = var
        assert var_spatial.ndim == 3
        assert var_spatial.shape[-2:] == (self.small_model.ny, self.small_model.nx)
        result_shape = (var_spatial.shape[-3],) + (self.big_model.ny, self.big_model.nx)
        resized = jax.image.resize(
            var_spatial,
            shape=result_shape,
            method="linear",
            antialias=False,
        )
        if self._is_spectral(var):
            return self._to_spec(resized)
        else:
            return resized


COARSEN_OPERATORS = {
    "op1": Operator1,
    "op2": Operator2,
    "basic_spectral": BasicSpectralCoarsener,
}
