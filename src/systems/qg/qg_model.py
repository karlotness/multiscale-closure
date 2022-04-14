import dataclasses
import jax
import jax.numpy as jnp
import jax.random
from . import model
from .kernel import DTYPE_REAL, DTYPE_COMPLEX, attach_to_object, PseudoSpectralKernelState, register_dataclass_pytree

class QGModel(model.Model):
    def __init__(
            self,
            beta=1.5e-11,
            rd=15000.0,
            delta=0.25,
            H1 = 500,
            U1=0.025,
            U2=0.0,
            **kwargs,
    ):
        super().__init__(nz=2, **kwargs)
        self.beta = beta
        self.rd = rd
        self.delta = delta
        self.Hi = jnp.array([H1, H1/delta])
        self.U1 = U1
        self.U2 = U2

        # initialize background, inversion matrix, forcing

        # INITIALIZE BACKGROUND
        self.H = self.Hi.sum()
        self.Ubg = jnp.array([self.U1, self.U2])
        self.U = self.U1 - self.U2
        # The F parameters
        self.F1 = self.rd**-2 / (1 + self.delta)
        self.F2 = self.delta * self.F1
        # The meridional PV gradients in each layer
        self.Qy1 = self.beta + self.F1 * (self.U1 - self.U2)
        self.Qy2 = self.beta - self.F2 * (self.U1 - self.U2)
        self.Qy = jnp.array([self.Qy1, self.Qy2])
        self._ikQy = 1j * (jnp.expand_dims(self.kk, 0) * jnp.expand_dims(self.Qy, -1))
        # complex versions, multiplied by k, speeds up computations to precompute
        self.ikQy1 = self.Qy1 * 1j * self.k
        self.ikQy2 = self.Qy2 * 1j * self.k
        # vector version
        self.ikQy = jnp.vstack([jnp.expand_dims(self.ikQy1, axis=0), jnp.expand_dims(self.ikQy2, axis=0)])
        self.ilQx = 0
        #layer spacing
        self.del1 = self.delta / (self.delta + 1)
        self.del2 = (self.delta + 1) ** -1

        # INITIALIZE INVERSION MATRIX
        a = jnp.zeros((self.nz, self.nz, self.nl, self.nk), dtype=DTYPE_REAL)
        # inverse determinant
        det_inv = self.wv2 * (self.wv2 + self.F1 + self.F2)
        det_inv_mask = (det_inv != 0)
        det_inv = jnp.where(det_inv_mask, det_inv**-1, 0)
        a = a.at[0, 0].set(-(self.wv2 + self.F2) * det_inv)
        a = a.at[0, 1].set(-self.F1 * det_inv)
        a = a.at[1, 0].set(-self.F2 * det_inv)
        a = a.at[1, 1].set(-(self.wv2 + self.F1) * det_inv)
        self.a = jnp.where(jnp.isfinite(a), a, 0).astype(DTYPE_COMPLEX)

        # INITIALIZE FORCING
        pass # nothing to do

        # calc cfl
        @attach_to_object(self)
        def _calc_cfl(state):
            return jnp.abs(
                jnp.hstack([state.u + jnp.expand_dims(self.Ubg, axis=(1, 2)), state.v])
            ).max() * self.dt / self.dx

        # calc ke
        @attach_to_object(self)
        def _calc_ke(state):
            ke1 = 0.5 * self.Hi[0] * self.spec_var(self.wv * state.ph[0])
            ke2 = 0.5 * self.Hi[1] * self.spec_var(self.wv * state.ph[1])
            return (ke1.sum() + ke2.sum()) / self.H

        @attach_to_object(self)
        def _calc_eddy_time(state):
            ens = 0.5 * self.Hi[0] * self.spec_var(self.wv2 * self.ph1) + 0.5 * self.Hi[1] * self.spec_var(self.wv2 * self.ph2)
            return 2 * jnp.pi * jnp.sqrt(self.H / ens) / 86400

        @attach_to_object(self)
        def _set_q1q2(state, q1, q2):
            return self.set_q(state, jnp.vstack([jnp.expand_dims(q1, axis=0), jnp.expand_dims(q2, axis=0)]))

        @attach_to_object(self)
        def create_initial_state(rng):
            return self._create_initial_state(rng)

    def _create_initial_state(self, rng):
        state = super()._create_initial_state()
        # initial conditions (pv anomalies)
        rng_a, rng_b = jax.random.split(rng, num=2)
        q1 = 1e-7 * jax.random.uniform(rng_a, shape=(self.ny, self.nx)) + 1e-6 * (jnp.ones((self.ny, 1)) * jax.random.uniform(rng_b, shape=(1, self.nx)))
        q2 = jnp.zeros_like(self.x)
        state = self._set_q1q2(state, q1, q2)
        return state
