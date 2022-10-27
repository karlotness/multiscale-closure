# FCNN parameterization like Guillaumin and Zanna (2021) and based on
# implementation for Ross, Li, Perezhogin, et al (2022)

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import equinox as eqx
from .eqx_modules import EasyPadConv


class GZFCNN(eqx.Module):
    conv_seq: eqx.nn.Sequential
    img_size: int = eqx.static_field()
    n_layers: int = eqx.static_field()

    def __init__(self, img_size: int, n_layers: int, padding: str = "circular", *, key: Array):
        self.img_size = img_size
        self.n_layers = n_layers

        features_kernels = [
            (128, (5, 5)),
            (64, (5, 5)),
            (32, (3, 3)),
            (32, (3, 3)),
            (32, (3, 3)),
            (32, (3, 3)),
            (32, (3, 3)),
            (n_layers, (3, 3)),
        ]
        ops = []
        prev_chans = n_layers + 1
        keys = jax.random.split(key, len(features_kernels))
        for (feature, kern_size), conv_key in zip(features_kernels, keys, strict=True):
            conv = EasyPadConv(
                num_spatial_dims=2,
                in_channels=prev_chans,
                out_channels=feature,
                kernel_size=kern_size,
                padding=padding,
                use_bias=True,
                key=conv_key,
            )
            ops.extend([conv, eqx.nn.Lambda(jax.nn.relu)])
            prev_chans = feature
        # Remove final activation
        ops.pop()
        self.conv_seq = eqx.nn.Sequential(ops)

    def __call__(self, x: Array, t: Float, *, key: Array|None = None):
        assert x.ndim == 3
        assert x.shape[-2:] == (self.img_size, self.img_size)
        assert x.shape[-3] == self.n_layers
        # Place time after x dimension
        x = jnp.concatenate(
            [
                x,
                jnp.expand_dims(jnp.full_like(x, t, shape=x.shape[1:]), 0),
            ]
        )
        return self.conv_seq(x, key=key)
