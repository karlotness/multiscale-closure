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
