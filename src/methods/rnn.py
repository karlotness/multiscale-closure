from collections.abc import Sequence
import jax.numpy as jnp
import flax.linen as nn
from ._defs import ACTIVATIONS
from .cnn import CNN


class RNN(nn.Module):
    cells: Sequence[nn.Module]

    def initial_state(self, x):
        # Stack initial states for modules
        states = [c.initial_state(x) for c in self.cells]
        return states

    @nn.compact
    def __call__(self, x, hidden):
        new_hiddens = []
        for cell, h in zip(self.cells, hidden):
            x, new_h = cell(x, h)
            new_hiddens.append(new_h)
        return x, new_hiddens


class ConvLSTMCell(nn.Module):
    # Implementation after shi15
    filters: int
    kernel_size: int
    activation: str = "tanh"
    recurrent_activation: str = "hard_sigmoid"
    bias: bool = True

    def initial_state(self, x):
        # Return (hidden, carry)
        hidden_shape = list(x.shape)
        hidden_shape[-1] = self.filters
        hidden_shape = tuple(hidden_shape)
        return (jnp.zeros_like(x, shape=hidden_shape), jnp.zeros_like(x, shape=hidden_shape))

    @nn.compact
    def __call__(self, x, hidden):
        hidden, carry = hidden
        rank = x.ndim - 1
        kernel_size = (self.kernel_size, ) * rank
        # Input conv
        x_i = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=self.bias)(x)
        x_f = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=self.bias)(x)
        x_c = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=self.bias)(x)
        x_o = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=self.bias)(x)
        # Recurrent conv
        h_i = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=False)(hidden)
        h_f = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=False)(hidden)
        h_c = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=False)(hidden)
        h_o = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=False)(hidden)
        # Combine results
        i = ACTIVATIONS[self.recurrent_activation](x_i + h_i)
        f = ACTIVATIONS[self.recurrent_activation](x_f + h_f)
        c = f * carry + i * ACTIVATIONS[self.activation](x_c + h_c)
        o = ACTIVATIONS[self.recurrent_activation](x_o + h_o)
        h = o * ACTIVATIONS[self.activation](c)
        # Return y, new_hidden, new_carry
        return h, (h, c)
