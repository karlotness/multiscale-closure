# Derived from code courtesy of Pavel Perezhogin and pyqg

import jax.numpy as jnp
import jax
import functools
from .qg_model import QGModel


def make_spectrum_computer(type='power', averaging=False, truncate=False, include_k=False):

    def fft2d(arr):
        M = arr.shape[-1] * arr.shape[-2]
        return jnp.fft.rfftn(arr, axes=(-2,-1)) / M

    def isotropize(af2, *x):
        # TODO: check computation of nx
        nx = x[0].shape[-1]
        m = QGModel(nx=nx)

        if type == "cross_layer":
            raise ValueError("unsupported spectrum type 'cross_layer'")

        def _comp_ispec(af):
            k, sp = calc_ispec(m, af, averaging=averaging, truncate=truncate)
            return k, sp
        sp = jax.vmap(_comp_ispec)(af2)
        # Package for return
        return sp, k

    def spectrum_computer(*x):
        if any(xx.ndim != 5 for xx in x):
            # Expect dimensions: [run, time, lev, Ny, Nx]
            raise ValueError(f"wrong number of dimensions on spectrum input need 5 had {xx.ndim}")

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

        sp, k = isotropize(af2, *x)

        if include_k:
            return sp, k
        else:
            return sp

    return spectrum_computer


def calc_ispec(model, var_dens, averaging=True, truncate=True, nd_wavenumber=False, nfactor=1):
    # account for complex conjugate
    var_dens[...,0] /= 2
    var_dens[...,-1] /= 2

    ll_max = jnp.max(jnp.abs(model.ll))
    kk_max = jnp.max(jnp.abs(model.kk))

    if truncate:
        kmax = jnp.minimum(ll_max, kk_max)
    else:
        kmax = jnp.sqrt(ll_max**2 + kk_max**2)

    kmin = jnp.minimum(model.dk, model.dl)

    dkr = jnp.sqrt(model.dk**2 + model.dl**2) * nfactor

    # left border of bins
    kr = jnp.arange(kmin, kmax-dkr, dkr)

    if averaging:
        fkr = (model.wv >= kr) & (model.wv <= kr + dkr)
        phr = jnp.vmap(functools.partial(jnp.mean, where=fkr), var_dens) * (kr + dkr / 2) * jnp.pi / (model.dk * model.dl)
    else:
        fkr = (model.wv >= kr) & (model.wv < kr + dkr)
        phr = jnp.vmap(functools.partial(jnp.sum, where=fkr), var_dens) / dkr

    phr = phr * 2

    # convert left border of the bin to center
    kr = kr + dkr/2

    # convert to non-dimensional wavenumber
    # preserving integral over spectrum
    if nd_wavenumber:
        kr = kr / kmin
        phr = phr * kmin

    return kr, phr
