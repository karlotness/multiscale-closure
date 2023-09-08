# Derived from code courtesy of Pavel Perezhogin and pyqg

import jax.numpy as jnp
import jax
import numpy as np
import functools
import pyqg_jax


def make_spectrum_computer(type='power', averaging=False, truncate=False):

    def fft2d(arr):
        M = arr.shape[-1] * arr.shape[-2]
        return jnp.fft.rfftn(arr, axes=(-2,-1)) / M

    def isotropize(af2, *x):
        # TODO: check computation of nx
        with jax.ensure_compile_time_eval():
            nx = x[0].shape[-1]
            m = pyqg_jax.qg_model.QGModel(nx=nx)

        if type == "cross_layer":
            raise ValueError("unsupported spectrum type 'cross_layer'")

        def _comp_ispec(af):
            _k, sp = calc_ispec(m, af, averaging=averaging, truncate=truncate)
            return sp
        sp = jax.vmap(_comp_ispec)(af2)
        # Package for return
        return sp

    def spectrum_computer(*x):
        if any(xx.ndim != 5 for xx in x):
            # Expect dimensions: [run, time, lev, Ny, Nx]
            raise ValueError(f"wrong number of dimensions on spectrum input need 5 for [run, time, lev, Ny, Nx]")

        match type:
            case "power":
                af2 = jnp.abs(fft2d(x[0]))**2
            case "energy":
                af2 = jnp.abs(fft2d(x[0]))**2 / 2
            case "cospectrum":
                af2 = jnp.real(jnp.conj(fft2d(x[0])) * fft2d(x[1]))
            case "cross_layer":
                raise ValueError("unsupported spectrum type 'cross_layer'")

        af2 = jnp.mean(af2, axis=(0, 1))

        sp = isotropize(af2, *x)

        return sp

    return spectrum_computer


def calc_ispec(model, var_dens, averaging=True, truncate=True, nd_wavenumber=False, nfactor=1):
    # account for complex conjugate
    var_dens = jnp.concatenate(
        [
            jnp.expand_dims(var_dens[..., 0] / 2, -1),
            var_dens[..., 1:-1],
            jnp.expand_dims(var_dens[..., -1] / 2, -1),
        ],
        axis=-1
    )

    with jax.ensure_compile_time_eval():
        ll_max = jnp.max(jnp.abs(model.ll))
        kk_max = jnp.max(jnp.abs(model.kk))

        if truncate:
            kmax = jnp.minimum(ll_max, kk_max)
        else:
            kmax = jnp.sqrt(ll_max**2 + kk_max**2)

        kmin = jnp.minimum(model.dk, model.dl)

        dkr = jnp.sqrt(model.dk**2 + model.dl**2) * nfactor
        kdiff = kmax - dkr
        # left border of bins
        kr = jnp.arange(kmin, kdiff, dkr)


    # Shape for nx=64 -> wv=(64, 33), kr=(31,), dkr=(), var_dens=(64, 33)
    # so fkr should have shape (31, 64, 33)

    if averaging:

        def _avg(kri):
            fkr = (model.wv >= kri) & (model.wv <= kri + dkr)
            return jax.lax.cond(
                jnp.any(fkr),
                (lambda: jnp.mean(var_dens, where=fkr) * (kri + dkr / 2) * jnp.pi / (model.dk * model.dl)),
                (lambda: jnp.zeros_like(var_dens, shape=())),
            )

        phr = jax.vmap(_avg)(kr)
    else:

        def _sum(kri):
            fkr = (model.wv >= kri) & (model.wv < kri + dkr)
            return jnp.sum(var_dens, where=fkr) / dkr

        phr = jax.vmap(_sum)(kr)

    phr = phr * 2

    # convert left border of the bin to center
    kr = kr + dkr/2

    # convert to non-dimensional wavenumber
    # preserving integral over spectrum
    if nd_wavenumber:
        kr = kr / kmin
        phr = phr * kmin

    return kr, phr
