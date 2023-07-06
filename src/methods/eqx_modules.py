from collections.abc import Sequence
import typing
import types
import functools
from jaxtyping import Array
import jax.numpy as jnp
import equinox as eqx


class TrainableWeightBias(eqx.Module):
    num_spatial_dims: int = eqx.static_field()
    num_layers: int = eqx.static_field()
    weight: Array
    bias: Array

    def __init__(self, num_spatial_dims, num_layers, weight=True, bias=True):
        self.num_spatial_dims = num_spatial_dims
        self.num_layers = num_layers
        shape = (num_layers, ) + (1, ) * num_spatial_dims
        self.weight = jnp.ones(shape) if weight else None
        self.bias = jnp.zeros(shape) if bias else None

    def __call__(self, x: Array, *, key: typing.Optional["jax.random.PRNGKey"] = None):
        if self.weight is not None:
            x = self.weight * x
        if self.bias is not None:
            x = x + self.bias
        return x


def _do_pad_input(x, pad_type, strides, filter_sizes, n_spatial_dims, dilations):
    if pad_type not in {"valid", "circular", "same"}:
        raise ValueError(f"invalid padding type {pad_type}")
    if pad_type == "valid":
        return x
    # Compute padding size for same/circular
    spatial_dims = x.shape[-n_spatial_dims:]
    if isinstance(strides, int):
        strides = (strides, ) * n_spatial_dims
    if isinstance(filter_sizes, int):
        filter_sizes = (filter_sizes, ) * n_spatial_dims
    if isinstance(dilations, int):
        dilations = (dilations, ) * n_spatial_dims
    pad_amts = [(0, 0)] * (x.ndim - n_spatial_dims)
    for stride, filt, size, dilate in zip(strides, filter_sizes, spatial_dims, dilations, strict=True):
        filt = filt * dilate
        if size % stride == 0:
            pad_along_dim = max(filt - stride, 0)
        else:
            pad_along_dim = max(filt - (size % stride), 0)
        pad_before = pad_along_dim // 2
        pad_after = pad_along_dim - pad_before
        pad_amts.append((pad_before, pad_after))
    # Do the actual padding
    return jnp.pad(
        x,
        pad_amts,
        mode="wrap" if pad_type == "circular" else "constant",
    )


@functools.cache
def make_circular_pooling(pool_cls):

    def cls_init(self, *args, **kwargs):
        return pool_cls.__init__(self, *args, **kwargs, padding=0)

    def cls_call(self, x, *, key=None):
        padded = _do_pad_input(
            x,
            pad_type="circular",
            strides=self.stride,
            filter_sizes=self.kernel_size,
            n_spatial_dims=self.num_spatial_dims,
            dilations=1,
        )
        return pool_cls.__call__(self, padded, key=key)

    def update_namespace(ns):
        ns["__init__"] = cls_init
        ns["__call__"] = cls_call

    return types.new_class(
        f"CircularPadded{pool_cls.__name__}",
        bases=(pool_cls, ),
        exec_body=update_namespace,
    )


class EasyPadConv(eqx.Module):
    conv_op: eqx.nn.Conv
    padding_type: str = eqx.static_field()

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int|Sequence[int],
        stride: int = 1,
        padding: typing.Literal["same", "valid", "circular"] = "valid",
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        *,
        key = "jax.random.PRNGKey",
    ):
        self.padding_type = padding
        self.conv_op = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            key=key,
        )

    def __call__(self, x: Array, *, key: Array|None = None):
        # Do pre-padding if needed
        padded = _do_pad_input(
            x,
            pad_type=self.padding_type,
            strides=self.conv_op.stride,
            filter_sizes=self.conv_op.kernel_size,
            n_spatial_dims=self.conv_op.num_spatial_dims,
            dilations=self.conv_op.dilation,
        )
        return self.conv_op(padded)
