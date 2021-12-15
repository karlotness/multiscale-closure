from collections.abc import Sequence
import jax.numpy as jnp
import flax.linen as nn
from ._defs import ACTIVATIONS

class MLP(nn.Module):
    output_dims: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        for i, dim in enumerate(self.output_dims):
            x = nn.Dense(dim)(x)
            if i < len(self.output_dims) - 1:
                x = ACTIVATIONS[self.activation](x)
        return x
