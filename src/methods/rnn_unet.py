from collections.abc import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
from ._defs import ACTIVATIONS, UVParameterization


class TrainableConstLike(nn.Module):
    @nn.compact
    def __call__(self, x):
        param = self.param("const", jax.nn.initializers.constant(0), x.shape, x.dtype)
        return param


class RNN(nn.Module):
    cells: Sequence[nn.Module]

    def initial_state(self, x):
        # Stack initial states for modules
        states = [c.initial_state(x) for c in self.cells]
        return states

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

    def setup(self):
        rank = 2
        kernel_size = (self.kernel_size, ) * rank
        # Two learnable initial states
        self.init_hidden = TrainableConstLike()
        self.init_carry = TrainableConstLike()
        # The required convolutions
        self.x_to_xi = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=self.bias)
        self.x_to_xf = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=self.bias)
        self.x_to_xc = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=self.bias)
        self.x_to_xo = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=self.bias)
        self.hidden_to_hi = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=False)
        self.hidden_to_hf = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=False)
        self.hidden_to_hc = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=False)
        self.hidden_to_ho = nn.Conv(features=self.filters, kernel_size=kernel_size, strides=1, padding="SAME", use_bias=False)

    def initial_state(self, x):
        # Return (hidden, carry)
        return self.init_hidden(x), self.init_carry(x)

    def __call__(self, x, hidden):
        hidden, carry = hidden
        # Input conv
        x_i = self.x_to_xi(x)
        x_f = self.x_to_xf(x)
        x_c = self.x_to_xc(x)
        x_o = self.x_to_xo(x)
        # Recurrent conv
        h_i = self.hidden_to_hi(hidden)
        h_f = self.hidden_to_hf(hidden)
        h_c = self.hidden_to_hc(hidden)
        h_o = self.hidden_to_ho(hidden)
        # Combine results
        i = ACTIVATIONS[self.recurrent_activation](x_i + h_i)
        f = ACTIVATIONS[self.recurrent_activation](x_f + h_f)
        c = f * carry + i * ACTIVATIONS[self.activation](x_c + h_c)
        o = ACTIVATIONS[self.recurrent_activation](x_o + h_o)
        h = o * ACTIVATIONS[self.activation](c)
        # Return y, new_hidden, new_carry
        return h, (h, c)


class RNNUNet(nn.Module):
    cells_chans_kernel_levels: Sequence[tuple[int, int, int]]
    activation: str = "relu"
    out_size: int = 4

    def setup(self):
        rnns = []
        for cells, chans, kernel in self.cells_chans_kernel_levels:
            # Define core RNNs
            rnns.append(
                RNN(
                    cells=[
                        ConvLSTMCell(
                            filters=chans,
                            kernel_size=kernel,
                        )
                        for _ in range(cells)
                    ]
                )
            )
        self.rnns = rnns
        # Define the u-net up/down convolutions
        conv_down = []
        conv_up = []
        out_size = self.out_size
        for cells, chans, kernel in self.cells_chans_kernel_levels:
            conv_down.append(
                nn.Conv(features=chans, kernel_size=(3, 3), padding="SAME")
            )
            conv_up.append(
                nn.ConvTranspose(features=out_size, kernel_size=(3, 3), padding="SAME")
            )
            out_size *= 2
        self.conv_down = conv_down
        self.conv_up = conv_up[::-1]
        self.mini_conv = nn.Conv(features=chans, kernel_size=(1, 1), padding="SAME")

    @nn.nowrap
    def _do_pool_down(self, x):
        return nn.avg_pool(x, (2, 2), strides=(2, 2))

    @nn.nowrap
    def _do_pool_up(self, x):
        return jnp.repeat(jnp.repeat(x, 2, axis=-2), 2, axis=-3)

    def initial_state(self, x):
        act = ACTIVATIONS[self.activation]
        initial_states = []
        for c_down, rnn in zip(self.conv_down, self.rnns):
            x = c_down(x)
            initial_states.append(rnn.initial_state(act(x)))
            x = self._do_pool_down(x)
        return initial_states

    def __call__(self, x, hidden):
        act = ACTIVATIONS[self.activation]
        levels = []
        new_hidden = []
        # Downward pass
        for c_down, rnn, h in zip(self.conv_down, self.rnns, hidden, strict=True):
            x = c_down(x)
            y, new_h = rnn(act(x), h)
            levels.append(y)
            new_hidden.append(new_h)
            x = self._do_pool_down(x)
        x = self.mini_conv(x)
        # Upward pass
        for c_up in self.conv_up:
            partner = levels.pop()
            x = c_up(jnp.concatenate([partner, act(self._do_pool_up(x))], axis=-1))
        return x, new_hidden


class RNNUNetUV(UVParameterization):
    # Constants from QG validation set
    u_mean: float = -1.05890814e-13
    u_std:  float = 0.009262884
    v_mean: float = 1.0549827e-13
    v_std:  float = 0.009245564
    activation: str = "relu"
    cells_chans_kernel_levels: Sequence[tuple[int, int, int]] = ((7, 64, 3), (5, 128, 3), (2, 256, 2))

    def setup(self):
        self.unet = RNNUNet(
            cells_chans_kernel_levels=self.cells_chans_kernel_levels,
            activation=self.activation,
            out_size=4,
        )

    def net_description(self):
        return {
            "architecture": "rnn-unet",
            "params": {
                "activation": self.activation,
                "cells_chans_kernel_levels": self.cells_chans_kernel_levels,
                "u_mean": self.u_mean,
                "u_std": self.u_std,
                "v_mean": self.v_mean,
                "v_std": self.v_std,
            },
        }

    @nn.nowrap
    def _init_process_states(self, u, v):
        u = (u - self.u_mean) / self.u_std
        v = (v - self.v_mean) / self.v_std
        x = jnp.moveaxis(jnp.concatenate([u, v], axis=0), 0, -1)
        return x

    def init_memory(self, u, v):
        return self.unet.initial_state(self._init_process_states(u, v))

    def parameterization(self, u, v, memory):
        x = self._init_process_states(u, v)
        y, memory = self.unet(x, memory)
        y = jnp.moveaxis(y, -1, 0)
        sx, sy = jnp.split(y, 2, axis=0)
        sx = sx * self.u_std
        sy = sy * self.v_std
        return sx, sy, memory
