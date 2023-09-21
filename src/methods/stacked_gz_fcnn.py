import collections.abc
import jax
import jax.numpy as jnp
import equinox as eqx
from .gz_fcnn import GZFCNN, MediumGZFCNN, BaseGZFCNN


class BaseStackedGZFCNN(eqx.Module):
    seq: collections.abc.Sequence[eqx.Module]
    img_size: int = eqx.field(static=True)
    n_layers_in: int = eqx.field(static=True)
    n_layers_out: int = eqx.field(static=True)

    def __init__(
        self,
        img_size: int,
        n_layers_in: int,
        n_layers_out: int,
        depth: int,
        padding: str = "circular",
        normalization: str | None = None,
        zero_mean: bool = False,
        fcnn_class: type[BaseGZFCNN] = GZFCNN,
        *,
        key: jax.Array,
    ):
        self.seq = []
        self.img_size = img_size
        self.n_layers_in = n_layers_in
        self.n_layers_out = n_layers_out
        prev_layers = n_layers_in
        keys = jax.random.split(key, depth)
        for net_key in keys:
            self.seq.append(
                fcnn_class(
                    img_size=img_size,
                    n_layers_in=prev_layers,
                    n_layers_out=n_layers_out,
                    padding=padding,
                    normalization=normalization,
                    zero_mean=zero_mean,
                    key=net_key,
                )
            )
            prev_layers += self.seq[-1].n_layers_out

    def __call__(self, x: jax.Array, *, key: jax.Array|None = None):
        inputs = [x]
        for net in self.seq:
            result = jax.checkpoint(lambda ips: net(jnp.concatenate(ips, axis=0)))(inputs)
            inputs.append(result)
        return result


def StackedGZFCNN(img_size, n_layers_in, n_layers_out, depth, padding="circular", normalization=None, zero_mean=False, *, key):
    return BaseStackedGZFCNN(
        img_size=img_size,
        n_layers_in=n_layers_in,
        n_layers_out=n_layers_out,
        depth=depth,
        padding=padding,
        fcnn_class=GZFCNN,
        normalization=normalization,
        zero_mean=zero_mean,
        key=key,
    )

def MediumStackedGZFCNN(img_size, n_layers_in, n_layers_out, depth, padding="circular", normalization=None, zero_mean=False, *, key):
    return BaseStackedGZFCNN(
        img_size=img_size,
        n_layers_in=n_layers_in,
        n_layers_out=n_layers_out,
        depth=depth,
        padding=padding,
        fcnn_class=MediumGZFCNN,
        normalization=normalization,
        zero_mean=zero_mean,
        key=key,
    )
