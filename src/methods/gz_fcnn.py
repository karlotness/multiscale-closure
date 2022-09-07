# FCNN parameterization like Guillaumin and Zanna (2021) and based on
# implementation for Ross, Li, Perezhogin, et al (2022)

import jax
import jax.numpy as jnp
import flax.linen as nn
from ._defs import ACTIVATIONS, QParameterization


class _FCNN(nn.Module):
    features_kernels: tuple = (
        (128, (5, 5)),
        (64, (5, 5)),
        (32, (3, 3)),
        (32, (3, 3)),
        (32, (3, 3)),
        (32, (3, 3)),
        (32, (3, 3)),
        (1, (3, 3)),
    )
    zero_mean: bool = True

    @nn.compact
    def __call__(self, x, train):
        seq_steps = []
        for features, kernel in self.features_kernels[:-1]:
            seq_steps.extend(
                [
                    nn.Conv(features=features, kernel_size=kernel, strides=1, padding="CIRCULAR"),
                    nn.relu,
                    nn.BatchNorm(epsilon=1e-05, momentum=0.1, use_running_average=not train, axis_name="batch"),
                ]
            )
        seq_steps.append(
            nn.Conv(features=self.features_kernels[-1][0], kernel_size=self.features_kernels[-1][1], strides=1, padding="CIRCULAR")
        )
        x = nn.Sequential(seq_steps)(x)
        if self.zero_mean:
            x = x - x.mean()
        return x


class GZFCNNV1(QParameterization):
    zero_mean: bool = True

    def net_description(self):
        return {
            "architecture": "gz-fcnn-v1",
            "params": {
                "zero_mean": self.zero_mean,
            },
        }

    @nn.compact
    def parameterization(self, q, u, v, memory, train):
        assert memory is None

        # Count the number of layers nz
        n_layers = q.shape[-3] # extract nz from (nz, ny, nx)
        x = jnp.concatenate([q, u, v], axis=0)
        x = jnp.moveaxis(x, 0, -1)
        # For each model layer nz, apply a separate _FCNN network
        dq = jnp.concatenate(
            [
                _FCNN(zero_mean=self.zero_mean)(x, train) for _ in range(n_layers)
            ],
            axis=-1,
        )
        return jnp.moveaxis(dq, -1, 0), None
