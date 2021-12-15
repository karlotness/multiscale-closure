import jax
import jax.numpy as jnp
import flax.linen as nn

ACTIVATIONS = {
    "relu": nn.relu,
    "hard_sigmoid": jax.nn.hard_sigmoid,
    "tanh": jnp.tanh,
    "sigmoid": nn.sigmoid,
}
