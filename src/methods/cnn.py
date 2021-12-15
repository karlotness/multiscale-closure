from collections.abc import Sequence
import jax.numpy as jnp
import flax.linen as nn
from ._defs import ACTIVATIONS

class CNN(nn.Module):
    features_kernels: Sequence[tuple[int, int]]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        for i, (features, kernels) in enumerate(self.features_kernels):
            x = nn.Conv(features=features, kernel_size=(kernels, ), strides=1, padding="SAME")(x)
            if i < len(self.features_kernels) - 1:
                x = ACTIVATIONS[self.activation](x)
        return x
