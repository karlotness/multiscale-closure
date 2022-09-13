import jax
import jax.numpy as jnp
import json
from .kernel import _generic_rfftn, _generic_irfftn
from .qg_model import QGModel

# Spectral filters as used in Ross, Li, Perezhogin, et al (2022)

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

    def coarsen_traj(self, big_traj):
        assert big_traj.q.ndim == 4
        assert big_traj.q.shape[1:] == (self.big_model.nz, self.big_model.ny, self.big_model.nx)
        # Coarsen each state
        def _coarsen_state(_carry, big_state):
            big_state = self.big_model.invert(big_state)
            # Create an appropriate zeroed initial state
            small_state = self.small_model.create_initial_state(jax.random.PRNGKey(0))
            # Copy basic attributes
            small_state.t = big_state.t
            small_state.tc = big_state.tc
            small_state.ablevel = big_state.ablevel
            # Initialize new q value
            small_state.q = self.coarsen_q(big_state.q) # Recompute q
            small_state = self.small_model.invert(small_state) # Recompute ph, u, v
            small_state = self.small_model.do_advection(small_state) # Recompute uq, vq, dqhdt
            small_state = self.small_model.do_friction(small_state) # Recompute dqhdt
            # Now all that remains is to patch up the shifted dqhdt_p and dqhdt_pp values (later pass)
            return None, small_state
        _, small_traj = jax.lax.scan(_coarsen_state, None, big_traj)
        # Patch up the dqhdt_p and dqhdt_pp values
        small_traj.dqhdt_p = jnp.concatenate(
            [
                jnp.expand_dims(jnp.zeros_like(small_traj.dqhdt_p[0]), 0),
                small_traj.dqhdt[:-1]
            ]
        )
        small_traj.dqhdt_pp = jnp.concatenate(
            [
                jnp.zeros_like(small_traj.dqhdt_pp[:2])
                small_traj.dqhdt[:-2]
            ]
        )
        # All done
        return small_traj

    def coarsen_q(self, q):
        raise NotImplementedError("implement in a subclass")

    def _to_spec(self, q):
        return _generic_rfftn(q)

    def _to_real(self, q):
        return _generic_irfftn(q)


class SpectralCoarsener(Coarsener):
    def coarsen_q(self, q):
        assert q.ndim == 3
        vh = self._to_spec(q)
        dummy_small_q = jnp.zeros(
            (self.small_model.nz, self.small_model.ny, self.small_model.nx),
            dtype=jnp.float32
        )
        dummy_qh = self._to_spec(dummy_small_q)
        nk = dummy_qh.shape[1] // 2
        trunc = np.hstack((vh[:, :nk,:nk+1],
                           vh[:,-nk:,:nk+1]))
        filtered = trunc * self.spectral_filter() / self.ratio**2
        return self._to_real(filtered)

    def spectral_filter(self):
        raise NotImplementedError("implement in a subclass")


class Operator1(SpectralCoarsener):
    def spectral_filter(self):
        return self.small_model.filtr


class Operator2(SpectralCoarsener):
    def spectral_filter(self):
        return jnp.exp(-self.small_model.wv**2 * (2*self.small_model.dx)**2 / 24)
