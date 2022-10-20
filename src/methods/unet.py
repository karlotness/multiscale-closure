from collections.abc import Sequence
import typing
import math
from jaxtyping import Array, Float
import jax
import jax.numpy as jnp
from ._defs import ACTIVATIONS
import equinox as eqx
from .eqx_modules import EasyPadConv


class LearnedTimeConv(eqx.Module):
    kernel_bias_mlp: eqx.nn.MLP
    kernel_bias_basis: eqx.nn.Linear
    in_channels: int = eqx.static_field()
    out_channels: int = eqx.static_field()
    kernel_size: int = eqx.static_field()
    use_bias: bool = eqx.static_field()
    basis_size: int = eqx.static_field()

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_bias: bool = True,
        basis_size: int = 10,
        width: int = 64,
        depth: int = 3,
        *,
        key: Array,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.basis_size = basis_size

        rng_1, rng_2 = jax.random.split(key)

        # Set up network for weights
        self.kernel_bias_mlp = eqx.nn.MLP(
            in_size=1,
            out_size=self.basis_size,
            width_size=width,
            depth=depth,
            activation=jax.nn.relu,
            key=rng_1,
        )
        # Set up network for outputs
        dummy_conv = self._get_dummy_conv()
        weight_size = math.prod(dummy_conv.conv_op.weight.shape)
        bias_size = math.prod(dummy_conv.conv_op.bias.shape) if self.use_bias else 0
        total_size = weight_size + bias_size
        self.kernel_bias_basis = eqx.nn.Linear(
            in_features=basis_size,
            out_features=total_size,
            use_bias=True,
            key=rng_2,
        )

    def _get_dummy_conv(self):
        return EasyPadConv(
            num_spatial_dims=2,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding="circular",
            dilation=1,
            groups=1,
            use_bias=self.use_bias,
            key=jax.random.PRNGKey(0),
        )

    def __call__(self, x: Array, t: Float, *, key: Array|None = None):
        kernel_bias = self.kernel_bias_basis(self.kernel_bias_mlp(t))
        # Create dummy convolution object
        dummy_conv = self._get_dummy_conv()
        weight_size = math.prod(dummy_conv.conv_op.weight.shape)
        # Split kernel from bias and reshape
        weight, bias = jnp.split(kernel_bias, [weight_size])
        # Update weight and bias of dummy conv
        if self.use_bias:
            assert dummy_conv.conv_op.bias is not None
            dummy_conv = eqx.tree_at(
                lambda mod: [mod.conv_op.weight, mod.conv_op.bias],
                dummy_conv,
                replace=[
                    weight.reshape(dummy_conv.conv_op.weight.shape),
                    bias.reshape(dummy_conv.conv_op.bias.shape)
                ],
            )
        else:
            dummy_conv = eqx.tree_at(
                lambda mod: mod.conv_op.weight,
                dummy_conv,
                replace=weight.reshape(dummy_conv.conv_op.weight.shape),
            )
        # Apply updated dummy conv
        return dummy_conv(x)


class PartiallyLearnedTimeConv(eqx.Module):
    static_conv: eqx.nn.Conv
    time_conv: LearnedTimeConv

    def __init__(
        self,
        in_channels: int,
        out_static_channels: int,
        out_time_channels: int,
        kernel_size: int,
        use_bias: bool = True,
        basis_size: int = 10,
        width: int = 64,
        depth: int = 3,
        *,
        key: Array,
    ):
        rng_1, rng_2 = jax.random.split(key)
        self.static_conv = EasyPadConv(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_static_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="circular",
            dilation=1,
            groups=1,
            use_bias=use_bias,
            key=rng_1,
        )
        self.time_conv = LearnedTimeConv(
            in_channels=in_channels,
            out_channels=out_time_channels,
            kernel_size=kernel_size,
            use_bias=use_bias,
            basis_size=basis_size,
            width=width,
            depth=depth,
            key=rng_2,
        )

    def __call__(self, x: Array, t: Float, *, key: Array|None = None):
        static_out = self.static_conv(x)
        time_out = self.time_conv(x, t)
        return jnp.concatenate([static_out, time_out], axis=-self.static_conv.conv_op.num_spatial_dims - 1)


class XTSequential(eqx.Module):
    modules: Sequence[eqx.Module]

    def __init__(self, modules, *, key: Array = None):
        self.modules = modules

    def __len__(self):
        return len(modules)

    def __getitem__(self, idx):
        return modules[idx]

    def __call__(self, x: Array, t: Float, *, key: Array = None):
        if key is None:
            rngs = [None] * len(self.modules)
        else:
            rngs = jax.random.split(key, len(self.modules))
        for module, k in zip(self.modules, rngs):
            if isinstance(module, (PartiallyLearnedTimeConv, LearnedTimeConv)):
                x = module(x, t, key=k)
            else:
                x = module(x, key=k)
        return x


class UNet(eqx.Module):
    blocks_downward: Sequence[eqx.Module]
    blocks_upward: Sequence[eqx.Module]
    sample_downward: Sequence[eqx.Module]
    sample_upward: Sequence[eqx.Module]
    across_convs: eqx.Module

    def __init__(self, *, key: "jax.random.PRNGKey"):
        self.blocks_downward = []
        self.blocks_upward = []
        self.sample_downward = []
        self.sample_upward = []
        in_chans = 2
        out_chans = 64
        rng_ctr = key
        for level in range(3):
            rng_ctr, *rngs = jax.random.split(rng_ctr, 6)
            # "Across" operations
            self.blocks_downward.append(
                XTSequential(
                    [
                        PartiallyLearnedTimeConv(
                            in_channels=in_chans,
                            out_static_channels=out_chans - 8,
                            out_time_channels=8,
                            kernel_size=3,
                            use_bias=True,
                            basis_size=10,
                            width=64,
                            depth=3,
                            key=rngs[0]
                        ),
                        eqx.nn.Lambda(jax.nn.relu),
                        EasyPadConv(
                            num_spatial_dims=2,
                            in_channels=out_chans,
                            out_channels=out_chans,
                            kernel_size=3,
                            padding="circular",
                            use_bias=True,
                            key=rngs[1],
                        ),
                    ]
                )
            )
            self.blocks_upward.append(
                XTSequential(
                    [
                        EasyPadConv(
                            num_spatial_dims=2,
                            in_channels=2*out_chans,
                            out_channels=out_chans,
                            kernel_size=3,
                            padding="circular",
                            use_bias=True,
                            key=rngs[2],
                        ),
                        eqx.nn.Lambda(jax.nn.relu),
                        PartiallyLearnedTimeConv(
                            in_channels=out_chans,
                            out_static_channels=(out_chans if level == 0 else out_chans//2) - 8,
                            out_time_channels=8,
                            kernel_size=3,
                            use_bias=True,
                            basis_size=10,
                            width=64,
                            depth=3,
                            key=rngs[3]
                        ),
                        (
                            eqx.nn.Identity() if level != 0 else
                            EasyPadConv(
                                num_spatial_dims=2,
                                in_channels=(out_chans if level == 0 else out_chans//2),
                                out_channels=in_chans,
                                kernel_size=1,
                                padding="circular",
                                use_bias=True,
                                key=rngs[4],
                            )
                        ),
                    ]
                )
            )
            # Pooling and upsampling operations
            self.sample_downward.append(
                eqx.nn.AvgPool2d(
                    kernel_size=2,
                    stride=2,
                )
            )
            self.sample_upward.append(
                eqx.nn.Lambda(
                    lambda x: jnp.repeat(jnp.repeat(x, 2, axis=-1), 2, axis=-2)
                )
            )
            in_chans = out_chans
            out_chans = out_chans * 2
        rng_ctr, rng_1, rng_2 = jax.random.split(rng_ctr, 3)
        self.across_convs = XTSequential(
            [
                EasyPadConv(
                    num_spatial_dims=2,
                    in_channels=in_chans,
                    out_channels=out_chans,
                    kernel_size=3,
                    padding="circular",
                    use_bias=True,
                    key=rng_1,
                ),
                eqx.nn.Lambda(jax.nn.relu),
                EasyPadConv(
                    num_spatial_dims=2,
                    in_channels=out_chans,
                    out_channels=in_chans,
                    kernel_size=3,
                    padding="circular",
                    use_bias=True,
                    key=rng_2,
                ),
            ]
        )
        self.sample_downward = self.sample_downward
        self.sample_upward = self.sample_upward

    def __call__(self, q: Array, t: Float, *, key: Array|None = None):
        # Check that we don't have a native batch (require vmap)
        assert q.ndim == 3
        assert q.shape[-1] == q.shape[-2]
        # Do downward pass
        skip_connections = []
        for down_op, resample in zip(self.blocks_downward, self.sample_downward):
            q = down_op(q, t)
            skip_connections.append(q)
            q = resample(q)
        # Do across convs
        q = self.across_convs(q, t)
        # Do upward ops
        for up_op, resample in zip(reversed(self.blocks_upward), reversed(self.sample_upward)):
            q = resample(q)
            q = jnp.concatenate([q, skip_connections.pop()], axis=-3)
            q = up_op(q, t)
        return q
