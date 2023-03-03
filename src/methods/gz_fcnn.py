# FCNN parameterization like Guillaumin and Zanna (2021) and based on
# implementation for Ross, Li, Perezhogin, et al (2022)

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import equinox as eqx
from .eqx_modules import EasyPadConv


GZFCNN_FEATURES_KERNELS = {
    "small": [
        (128, (5, 5)),
        (64, (5, 5)),
        (32, (3, 3)),
        (32, (3, 3)),
        (32, (3, 3)),
        (32, (3, 3)),
        (32, (3, 3)),
        ("layers_out", (3, 3)),
    ],
    "large": [
        (128, (11, 11)),
        (64, (11, 11)),
        (32, (7, 7)),
        (32, (7, 7)),
        (32, (7, 7)),
        (32, (7, 7)),
        (32, (7, 7)),
        ("layers_out", (7, 7)),
    ],
    "medium": [
        (128, (9, 9)),
        (64, (9, 9)),
        (32, (5, 5)),
        (32, (5, 5)),
        (32, (5, 5)),
        (32, (5, 5)),
        (32, (5, 5)),
        ("layers_out", (5, 5)),
    ],
}


class BaseGZFCNN(eqx.Module):
    conv_seq: eqx.nn.Sequential
    img_size: int = eqx.static_field()
    n_layers_in: int = eqx.static_field()
    n_layers_out: int = eqx.static_field()

    def __init__(
        self,
        img_size: int,
        n_layers_in: int,
        n_layers_out: int,
        arch_size: str,
        padding: str = "circular",
        normalization: str | None = None,
        *,
        key: Array
    ):
        self.img_size = img_size
        self.n_layers_in = n_layers_in
        self.n_layers_out = n_layers_out

        features_kernels = GZFCNN_FEATURES_KERNELS[arch_size]
        ops = []
        prev_chans = self.n_layers_in
        keys = jax.random.split(key, len(features_kernels))
        for (feature, kern_size), conv_key in zip(features_kernels, keys, strict=True):
            if feature == "layers_out":
                feature = self.n_layers_out
            ops.append(
                EasyPadConv(
                    num_spatial_dims=2,
                    in_channels=prev_chans,
                    out_channels=feature,
                    kernel_size=kern_size,
                    padding=padding,
                    use_bias=True,
                    key=conv_key,
                )
            )
            match normalization:
                case None | "none":
                    pass
                case "layer":
                    ops.append(
                        eqx.nn.LayerNorm(
                            shape=(feature, self.img_size, self.img_size),
                            elementwise_affine=True,
                        )
                    )
                case _:
                    raise ValueError(f"invalid normalization choise {self.normalization}")
            ops.append(eqx.nn.Lambda(jax.nn.relu))
            prev_chans = feature
        # Remove final activation
        ops.pop()
        self.conv_seq = eqx.nn.Sequential(ops)

    def __call__(self, x: Array, *, key: Array|None = None):
        assert x.ndim == 3
        assert x.shape[-2:] == (self.img_size, self.img_size)
        assert x.shape[-3] == self.n_layers_in
        return self.conv_seq(x, key=key)


def GZFCNN(img_size, n_layers_in, n_layers_out, padding="circular", normalization=None, *, key):
    return BaseGZFCNN(
        img_size=img_size,
        n_layers_in=n_layers_in,
        n_layers_out=n_layers_out,
        padding=padding,
        arch_size="small",
        normalization=normalization,
        key=key,
    )


def LargeGZFCNN(img_size, n_layers_in, n_layers_out, padding="circular", normalization=None, *, key):
    return BaseGZFCNN(
        img_size=img_size,
        n_layers_in=n_layers_in,
        n_layers_out=n_layers_out,
        padding=padding,
        arch_size="large",
        normalization=normalization,
        key=key,
    )


def MediumGZFCNN(img_size, n_layers_in, n_layers_out, padding="circular", normalization=None, *, key):
    return BaseGZFCNN(
        img_size=img_size,
        n_layers_in=n_layers_in,
        n_layers_out=n_layers_out,
        padding=padding,
        arch_size="medium",
        normalization=normalization,
        key=key,
    )
