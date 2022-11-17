import dataclasses
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import json
from .kernel import _generic_rfftn, _generic_irfftn, register_dataclass_pytree
from .qg_model import QGModel

# Spectral filters as used in Ross, Li, Perezhogin, et al (2022)

@register_dataclass_pytree
@dataclasses.dataclass
class PartialCoarsenedStep:
    q: Array
    dqhdt: Array
    t: Array
    tc: Array
    ablevel: Array
    q_total_forcing: Array


@register_dataclass_pytree
@dataclasses.dataclass
class PartialCoarsenedTraj:
    q: Array
    dqhdt: Array
    dqhdt_p: Array
    dqhdt_pp: Array
    t: Array
    tc: Array
    ablevel: Array
    q_total_forcing: Array

class Coarsener:
    def __init__(self, big_model, small_nx):
        assert big_model.nx == big_model.ny
        assert small_nx < big_model.nx
        assert small_nx % 2 == 0
        params = json.loads(big_model.param_json())
        params["nx"] = small_nx
        params["ny"] = small_nx
        self.big_model = big_model
        self.small_model = QGModel.from_param_json(json.dumps(params))
        self.ratio = self.big_model.nx / self.small_model.nx

    def _coarsen_step(self, big_state):
        assert big_state.q.ndim == 3
        assert big_state.q.shape == (self.big_model.nz, self.big_model.ny, self.big_model.nx)
        # Coarsen q, dqhdt
        q = self.coarsen(big_state.q)
        dqhdt = self.coarsen(big_state.dqhdt)
        # Copy other attributes
        t = big_state.t
        tc = big_state.tc
        ablevel = big_state.ablevel
        # Compute q total forcing
        small_state = self.small_model.create_initial_state(jax.random.PRNGKey(0))
        #  step 1: copy basic attributes
        small_state.t = big_state.t
        small_state.tc = big_state.tc
        small_state.ablevel = big_state.ablevel
        small_state.q = q
        #  step 2: run part of time-stepping calculations
        small_state = self.small_model.invert(small_state) # Recompute ph, u, v
        small_state = self.small_model.do_advection(small_state) # Recompute uq, vq, dqhdt
        small_state = self.small_model.do_friction(small_state) # Recompute dqhdt
        #  step 3: q forcing is the subtraction of dqhdt
        q_total_forcing = self._to_real(dqhdt) - self._to_real(small_state.dqhdt)
        # Package values and return
        return PartialCoarsenedStep(
            q=q,
            dqhdt=dqhdt,
            t=t,
            tc=tc,
            ablevel=ablevel,
            q_total_forcing=q_total_forcing,
        )

    def coarsen_traj(self, big_traj):
        assert big_traj.q.ndim == 4
        assert big_traj.q.shape[1:] == (self.big_model.nz, self.big_model.ny, self.big_model.nx)
        # Coarsen each state
        small_steps = jax.vmap(self._coarsen_step)(big_traj)
        # Patch up the dqhdt_p and dqhdt_pp values
        dqhdt_p = jnp.concatenate(
            [
                jnp.expand_dims(jnp.zeros_like(small_steps.dqhdt[0]), 0),
                small_steps.dqhdt[:-1],
            ]
        )
        dqhdt_pp = jnp.concatenate(
            [
                jnp.zeros_like(small_steps.dqhdt[:2]),
                small_steps.dqhdt[:-2],
            ]
        )
        small_traj = PartialCoarsenedTraj(
            q=small_steps.q,
            dqhdt=small_steps.dqhdt,
            dqhdt_p=dqhdt_p,
            dqhdt_pp=dqhdt_pp,
            t=small_steps.t,
            tc=small_steps.tc,
            ablevel=small_steps.ablevel,
            q_total_forcing=q_total_forcing,
        )
        # All done
        return small_traj

    def coarsen(self, var):
        raise NotImplementedError("implement in a subclass")

    def _to_spec(self, q):
        return _generic_rfftn(q)

    def _to_real(self, q):
        return _generic_irfftn(q)

    def _is_spectral(self, var):
        return var.dtype in {jnp.dtype(t) for t in (jnp.complex64, jnp.complex128)}


class SpectralCoarsener(Coarsener):
    def coarsen(self, var):
        assert var.ndim == 3
        dummy_small_var = jnp.zeros(
            (self.small_model.nz, self.small_model.ny, self.small_model.nx),
            dtype=jnp.float32
        )
        dummy_varh = self._to_spec(dummy_small_var)
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

    def spectral_filter(self):
        raise NotImplementedError("implement in a subclass")


class Operator1(SpectralCoarsener):
    def spectral_filter(self):
        return self.small_model.filtr


class Operator2(SpectralCoarsener):
    def spectral_filter(self):
        return jnp.exp(-self.small_model.wv**2 * (2*self.small_model.dx)**2 / 24)


COARSEN_OPERATORS = {
    "op1": Operator1,
    "op2": Operator2,
}
