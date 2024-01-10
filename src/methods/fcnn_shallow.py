# Shallower versions of the gz-fcnn architectures

import itertools
import collections
import typing
import jax
import jax.numpy as jnp
import equinox as eqx
from .eqx_modules import EasyPadConv, TrainableWeightBias


KERNEL_SIZES = {
    "puresmall": (3, 3),
    "small": (5, 3),
    "medium": (9, 5),
    "puremedium": (5, 5),
    "large": (11, 7),
}


class CustomFCNN(eqx.Module):
    conv_seq: eqx.nn.Sequential
    img_size: int = eqx.field(static=True)
    n_layers_in: int = eqx.field(static=True)
    n_layers_out: int = eqx.field(static=True)
    zero_mean: bool = eqx.field(static=True)

    def __init__(
        self,
        img_size: int,
        n_layers_in: int,
        n_layers_out: int,
        padding: str = "circular",
        normalization: str | None = None,
        zero_mean: bool = False,
        *,
        features_kernels: collections.abc.Sequence[tuple[(int | typing.Literal["layers_out"]), tuple[int, int]]],
        key: jax.Array,
    ):
        self.img_size = img_size
        self.n_layers_in = n_layers_in
        self.n_layers_out = n_layers_out
        self.zero_mean = zero_mean

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
                    ops.append(
                        eqx.nn.Identity()
                    )
                case "layer":
                    ops.append(
                        eqx.nn.Sequential(
                            [
                                eqx.nn.LayerNorm(
                                    shape=(feature, self.img_size, self.img_size),
                                    elementwise_affine=False,
                                ),
                                TrainableWeightBias(
                                    num_spatial_dims=2,
                                    num_layers=feature,
                                ),
                            ]
                        )
                    )
                case _:
                    raise ValueError(f"invalid normalization choice {self.normalization}")
            ops.append(eqx.nn.Lambda(jax.nn.relu))
            prev_chans = feature
        # Remove final activation and normalization
        self.conv_seq = eqx.nn.Sequential(ops[:-2])

    def __call__(self, x: jax.Array, *, key: jax.Array|None = None):
        assert x.ndim == 3
        assert x.shape[-2:] == (self.img_size, self.img_size)
        assert x.shape[-3] == self.n_layers_in
        res = self.conv_seq(x, key=key)
        if self.zero_mean:
            res = res - jnp.mean(res)
        return res


def make_shallow_fcnn(
    img_size,
    n_layers_in,
    n_layers_out,
    padding="circular",
    normalization=None,
    zero_mean=False,
    *,
    key,
    arch_version,
    arch_size,
    arch_layers,
):
    match arch_version:
        case 1:
            k_big, k_small = KERNEL_SIZES[arch_size]
            if arch_layers < 1:
                raise ValueError(f"must have at least 1 layer, requested {arch_layers}")
            features_kernels = [
                (c, (k, k)) for c, k in zip(
                    itertools.chain(
                        itertools.islice([128, 64], arch_layers - 1),
                        itertools.repeat(32, max(arch_layers - 3, 0)),
                        ["layers_out"],
                    ),
                    itertools.chain(itertools.repeat(k_big, 2), itertools.repeat(k_small)),
                )
            ]
            assert len(features_kernels) == arch_layers
            assert features_kernels[-1][0] == "layers_out"
            return CustomFCNN(
                img_size=img_size,
                n_layers_in=n_layers_in,
                n_layers_out=n_layers_out,
                padding=padding,
                normalization=normalization,
                zero_mean=zero_mean,
                features_kernels=features_kernels,
                key=key,
            )
        case _:
            raise ValueError(f"invalid version {arch_version}")
