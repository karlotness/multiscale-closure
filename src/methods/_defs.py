import jax
import jax.numpy as jnp
import flax.linen as nn

ACTIVATIONS = {
    "relu": nn.relu,
    "hard_sigmoid": jax.nn.hard_sigmoid,
    "tanh": jnp.tanh,
    "sigmoid": nn.sigmoid,
    "selu": jax.nn.selu,
}


class UVParameterization(nn.Module):

    def net_description(self):
        raise NotImplementedError("implement net_description in a subclass")

    def init_memory(self, u, v):
        return None

    def parameterization(self, u, v, memory):
        raise NotImplementedError("implement parameterization in a subclass")

    def __call__(self, u, v):
        return self.parameterization(u, v, self.init_memory(u, v))
