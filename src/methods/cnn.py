from collections.abc import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
from ._defs import ACTIVATIONS

class CNN(nn.Module):
    features_kernels: Sequence[tuple[int, int]]
    activation: str = "relu"
    rank: int = 1

    @nn.compact
    def __call__(self, x):
        for i, (features, kernels) in enumerate(self.features_kernels):
            kern_size = (kernels, ) * self.rank
            x = nn.Conv(features=features, kernel_size=kern_size, strides=1, padding="SAME")(x)
            if i < len(self.features_kernels) - 1:
                x = ACTIVATIONS[self.activation](x)
        return x


class ClosureCnnV1(nn.Module):
    nz: int = 2
    # Constants from QG validation set
    u_mean: float = -1.05890814e-13
    u_std:  float = 0.009262884
    v_mean: float = 1.0549827e-13
    v_std:  float = 0.009245564
    q_mean: float = 2.716908e-07
    q_std:  float = 1.7318747e-06

    def net_description(self):
        return {
            "architecture": "closure-cnn-v1",
            "params": {
                "nz": self.nz,
                "u_mean": self.u_mean,
                "u_std": self.u_std,
                "v_mean": self.v_mean,
                "v_std": self.v_std,
                "q_mean": self.q_mean,
                "q_std": self.q_std,
            },
        }

    @nn.compact
    def __call__(self, u, v):
        u = (u - self.u_mean) / self.u_std
        v = (v - self.v_mean) / self.v_std
        x = jnp.concatenate([u, v], axis=0)
        x = jnp.moveaxis(x, 0, -1)
        # Step 1
        x = nn.Conv(features=256, kernel_size=(5, 5))(x)
        x = jax.nn.selu(x)
        # Step 2
        x = nn.Conv(features=256, kernel_size=(5, 5))(x)
        x = jax.nn.selu(x)
        # Step 3
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = jax.nn.selu(x)
        # Step 4
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        # No fixed convs
        x = nn.Conv(features=2 * self.nz, kernel_size=(3, 3))(x)
        x = jnp.moveaxis(x, -1, 0)
        # Scale output by std (not mean since it's a derivative)
        # Split and return
        sx, sy = jnp.split(x, 2, axis=0)
        sx = sx * self.u_std
        sy = sy * self.v_std
        return sx, sy
