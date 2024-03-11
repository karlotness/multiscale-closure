import collections
import jax
import jax.numpy as jnp
import equinox as eqx
from . import get_net_constructor


class StackedNets(eqx.Module):
    seq: collections.abc.Sequence[eqx.Module]
    img_size: int = eqx.field(static=True)
    n_layers_in: int = eqx.field(static=True)
    n_layers_out: int = eqx.field(static=True)

    def __init__(
        self,
        layers,
        img_size: int,
        n_layers_in: int,
        n_layers_out: int,
        zero_mean: bool = False,
        *,
        key: jax.Array | None = None,
    ):
        if zero_mean:
            raise ValueError("zero_mean not supported")
        self.seq = layers
        self.img_size = img_size
        self.n_layers_in = n_layers_in
        self.n_layers_out = n_layers_out

    def __call__(self, x: jax.Array, *, key: jax.Array|None = None):
        inputs = [x]
        result = None
        for net in self.seq:
            result = jax.checkpoint(lambda ips: net(jnp.concatenate(ips, axis=0)))(inputs) + (0 if result is None else result)
            inputs.append(result)
        return result


def make_stacked_noscale_net(
    *,
    key,
    arch_version,
    arch_str,
    **arch_args,
):
    if arch_version != 1:
        raise ValueError(f"unsupported version {arch_version}")
    # Continue processing architecture string
    sub_arch_strings = arch_str.split(":")
    keys = jax.random.split(key, len(sub_arch_strings))
    nets = []
    layers_in = arch_args["n_layers_in"]
    for sub_arch, sub_key in zip(sub_arch_strings, keys, strict=True):
        sub_args = arch_args.copy()
        sub_args["n_layers_in"] = layers_in
        net_cls = get_net_constructor(sub_arch)
        net = net_cls(
            **sub_args,
            key=sub_key
        )
        nets.append(net)
        layers_in += sub_args["n_layers_out"]
    if not nets:
        raise ValueError("no sub-networks specified")
    final_net = StackedNets(
        layers=nets,
        img_size=arch_args["img_size"],
        n_layers_in=arch_args["n_layers_in"],
        n_layers_out=arch_args["n_layers_out"],
        zero_mean=arch_args.get("zero_mean", False),
        key=None,
    )
    return final_net
