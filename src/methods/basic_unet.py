import collections.abc
import operator
import itertools
import jax
import jax.numpy as jnp
import equinox as eqx
from .eqx_modules import EasyPadConv


def key_split_or_none(key, splits):
    if key is None:
        return [None] * splits
    else:
        return jax.random.split(key, splits)


def unpool_up(x):
    assert x.ndim == 3
    return jnp.repeat(jnp.repeat(x, 2, axis=-1), 2, axis=-2)


class BasicUNetV1(eqx.Module):
    img_size: int = eqx.static_field()
    n_layers_in: int = eqx.static_field()
    n_layers_out: int = eqx.static_field()
    layers_down: collections.abc.Sequence[eqx.Module]
    layers_up: collections.abc.Sequence[eqx.Module]
    pool_down: eqx.Module
    unpool_up: eqx.Module

    def __init__(
        self,
        img_size: int,
        n_layers_in: int,
        n_layers_out: int,
        padding: str = "circular",
        *,
        key: jax.Array
    ):
        self.img_size = img_size
        self.n_layers_in = n_layers_in
        self.n_layers_out = n_layers_out

        if self.img_size // 4 < 1:
            raise ValueError(f"image size to small {img_size}")
        if self.img_size % 4 != 0:
            raise ValueError("image size must be divisible by 4")

        conv_kernel_channels = [
            (
                # Top layer down
                [
                    (5, 128),
                    (5, 64),
                ],
                # Top layer up
                [
                    (3, 64),
                    (3, "out"),
                ],
            ),
            (
                # Middle layer down
                [
                    (3, 128),
                    (3, 128),
                ],
                # Middle layer up
                [
                    (3, 128),
                    (3, 64),
                ],
            ),
            (
                # Bottom layer down
                [
                    (3, 256),
                    (3, 256),
                ],
                # Bottom layer up
                [
                    (3, 256),
                    (3, 128),
                ],
            ),
        ]

        down_key, up_key = jax.random.split(key, 2)
        # Construct down layers
        layers_down = []
        down_keys = jax.random.split(down_key, len(conv_kernel_channels))
        in_channels = self.n_layers_in
        for layer_key, kern_chans_down in zip(down_keys, map(operator.itemgetter(0), conv_kernel_channels), strict=True):
            layer_keys = jax.random.split(layer_key, len(kern_chans_down))
            layers = []
            for key, kern_chan in zip(layer_keys, kern_chans_down, strict=True):
                kernel_size, channels_out = kern_chan
                if channels_out == "in":
                    channels_out = self.n_layers_in
                elif channels_out == "out":
                    channels_out = self.n_layers_out
                layers.extend(
                    [
                        EasyPadConv(
                            num_spatial_dims=2,
                            in_channels=in_channels,
                            out_channels=channels_out,
                            kernel_size=kernel_size,
                            use_bias=True,
                            padding="circular",
                            key=key,
                        ),
                        eqx.nn.Lambda(jax.nn.relu),
                    ]
                )
                in_channels = channels_out
            layers_down.append(eqx.nn.Sequential(layers))
        self.layers_down = layers_down
        # Construct up layers
        in_channels = 0
        layers_up = []
        up_keys = jax.random.split(up_key, len(conv_kernel_channels))
        for i, (layer_key, kern_chans_up, down_chans) in enumerate(
            zip(
                up_keys,
                map(operator.itemgetter(1), reversed(conv_kernel_channels)),
                map(lambda l: l[0][-1][-1], reversed(conv_kernel_channels)),
                strict=True,
            )
        ):
            layer_keys = jax.random.split(layer_key, len(kern_chans_up))
            layers = []
            in_channels += down_chans
            for key, kern_chan in zip(layer_keys, kern_chans_up, strict=True):
                kernel_size, channels_out = kern_chan
                if channels_out == "in":
                    channels_out = self.n_layers_in
                elif channels_out == "out":
                    channels_out = self.n_layers_out
                layers.extend(
                    [
                        EasyPadConv(
                            num_spatial_dims=2,
                            in_channels=in_channels,
                            out_channels=channels_out,
                            kernel_size=kernel_size,
                            use_bias=True,
                            padding="circular",
                            key=key,
                        ),
                        eqx.nn.Lambda(jax.nn.relu),
                    ]
                )
                in_channels = channels_out
            if i == len(conv_kernel_channels) - 1:
                layers.pop()
            layers_up.append(eqx.nn.Sequential(layers))
        self.layers_up = layers_up
        self.pool_down = eqx.nn.AvgPool2d(
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.unpool_up = eqx.nn.Lambda(unpool_up)

    def __call__(self, x: jax.Array, *, key: jax.Array|None = None):
        assert x.shape == (self.n_layers_in, self.img_size, self.img_size)
        key_down, key_up = key_split_or_none(key, 2)
        ret = x
        # Downward process
        down_keys = key_split_or_none(key, len(self.layers_down))
        skip_connections = []
        for i, (layer, key) in enumerate(zip(self.layers_down, down_keys, strict=True)):
            ret = layer(ret, key=key)
            skip_connections.append(ret)
            if i < len(self.layers_down) - 1:
                ret = self.pool_down(ret, key=None)
        ret = None
        # Upward process
        up_keys = key_split_or_none(key, len(self.layers_down))
        for layer, key in zip(self.layers_up, up_keys, strict=True):
            skip_value = skip_connections.pop()
            if ret is not None:
                ret = jnp.concatenate(
                    [
                        skip_value,
                        self.unpool_up(ret, key=None),
                    ],
                    axis = 0
                )
            else:
                ret = skip_value
            ret = layer(ret, key=key)
        assert ret.shape == (self.n_layers_out, self.img_size, self.img_size)
        return ret
