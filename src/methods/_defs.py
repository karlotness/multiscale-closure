import jax
import jax.numpy as jnp
import flax.linen as flax_nn

ACTIVATIONS = {
    "relu": jax.nn.relu,
    "hard_sigmoid": jax.nn.hard_sigmoid,
    "tanh": jnp.tanh,
    "sigmoid": jax.nn.sigmoid,
    "selu": jax.nn.selu,
}


class UVParameterization(flax_nn.Module):
    param_type = "uv"

    def net_description(self):
        raise NotImplementedError("implement net_description in a subclass")

    def init_memory(self, u, v):
        return None

    def parameterization(self, u, v, memory, train):
        raise NotImplementedError("implement parameterization in a subclass")

    def __call__(self, u, v, train):
        return self.parameterization(u, v, self.init_memory(u, v), train)


class QParameterization(flax_nn.Module):
    param_type = "q"

    def net_description(self):
        raise NotImplementedError("implement net_description in a subclass")

    def init_memory(self, q, u, v):
        return None

    def parameterization(self, q, u, v, memory, train):
        raise NotImplementedError("implement parameterization in a subclass")

    def __call__(self, q, u, v, train):
        return self.parameterization(q, u, v, self.init_memory(q, u, v), train)
