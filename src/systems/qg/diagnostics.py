# Derived from code courtesy of Pavel Perezhogin

import dataclasses
import jax.numpy as jnp
import jax
from .spectral import make_spectrum_computer
from .kernel import register_dataclass_pytree


@register_dataclass_pytree
@dataclasses.dataclass
class SubgridScoreResult:
    l2_mean: jnp.ndarray
    l2_total: jnp.ndarray
    l2_residual: jnp.ndarray
    var_ratio: jnp.ndarray


def _find_level_index(shape):
    if len(shape) == 2:
        candidate = -2
    else:
        candidate = -3
    candidate = len(shape) + candidate
    if shape[candidate] != 2:
        raise ValueError(f"failed to infer dimension 'lev', expected two layers")
    return candidate


def subgrid_scores(true, mean, gen):
    """
    Docs from original code:
    Compute scalar metrics for three components of subgrid forcing:
    - Mean subgrid forcing      ~ close to true forcing in MSE
    - Generated subgrid forcing ~ close to true forcing in spectrum
    - Genereted residual        ~ close to true residual in spectrum
    true - xarray with true forcing
    mean - mean prediction
    gen  - generated prediction

    Result is score, i.e. 1-mse/normalization

    Here we assume that dataset has dimensions run x time x lev x Ny x Nx
    """
    def l2(x, x_true):
        assert x.shape == x_true.shape
        lev_dim = _find_level_index(x.shape)
        dims = tuple(d for d in range(x.ndim) if d != lev_dim)
        # Temporarily use float64 for these (accuracy)
        x = x.astype(jnp.float64)
        x_true = x_true.astype(jnp.float64)
        res = jnp.mean(jnp.sqrt(jnp.mean((x - x_true)**2, axis=dims) / jnp.mean((x_true**2), axis=dims)))
        return res.astype(x.dtype)

    sp = make_spectrum_computer(type="power", averaging=False, truncate=False)

    l2_mean = l2(mean, true)
    sp_true = sp(true)
    sp_gen = sp(gen)
    l2_total = l2(sp_gen, sp_true)
    sp_true_res = sp(true - mean)
    sp_gen_res = sp(gen - mean)
    l2_residual = l2(sp_gen_res, sp_true_res)

    # Compute var_ratio
    gen_res = gen - mean
    true_res = true - mean
    lev_dim = _find_level_index(mean.shape)
    dims = tuple(d for d in range(mean.ndim) if d != lev_dim)
    var_ratio = jnp.array(jnp.mean(gen_res**2, axis=dims) / jnp.mean(true_res**2, axis=dims))

    # Return results
    return SubgridScoreResult(
        l2_mean=l2_mean,
        l2_total=l2_total,
        l2_residual=l2_residual,
        var_ratio=var_ratio,
    )
