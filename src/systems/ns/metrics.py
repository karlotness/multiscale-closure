import numpy as np
import jax.numpy as jnp
import jax_cfd.data
import warnings


def ke(u, v):
    return (u**2 + v**2) / 2


def vort2d(u, v, grid):
    dx, dy = grid.step
    dv_dx = (jnp.roll(v, -1, axis=-2) - v) / dx
    du_dy = (jnp.roll(u, -1, axis=-1) - u) / dy
    return dv_dx - du_dy


def normalize(v):
    norm = jnp.sqrt(jnp.sum(v**2, axis=(-1, -2)))
    return v / norm


def correlate(predicted, target):
    predicted = normalize(predicted)
    target = normalize(target)
    return jnp.mean(jnp.sum((predicted * target), axis=(-1, -2)))


# THIS FUNCTION IS NOT JAX!
def energy_spectrum(u_traj, v_traj, grid, time_traj):
    assert u_traj.ndim == 3
    assert v_traj.ndim == 3
    assert time_traj.ndim == 1
    assert time_traj.shape[0] == u_traj.shape[0] == v_traj.shape[0]
    assert u_traj.shape == v_traj.shape
    ds = jax_cfd.data.xarray_utils.velocity_trajectory_to_xarray(
        trajectory=(u_traj, v_traj),
        grid=grid,
        time=time_traj,
        samples=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng_spec = jax_cfd.data.xarray_utils.isotropic_energy_spectrum(
            ds, average_dims=(),
        ).rename("energy_spectrum")
    return np.moveaxis((eng_spec.k**5 * eng_spec).data, 0, -1), eng_spec.k.data
