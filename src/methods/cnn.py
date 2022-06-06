from collections.abc import Sequence
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
    @nn.compact
    def __call__(self, u, v):
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
        x = nn.Conv(features=2 * small_model.nz, kernel_size=(3, 3))(x)
        x = jnp.moveaxis(x, -1, 0)
        # Split and return
        sx, sy = jnp.split(x, 2, axis=0)
        return sx, sy
