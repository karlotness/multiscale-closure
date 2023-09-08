import math
import functools
import operator
import dataclasses
import jax
import jax.numpy as jnp
import equinox as eqx
import powerpax as ppx


@jax.tree_util.register_pytree_node_class
class Scaler:
    def __init__(self, mean, var):
        self.mean = jnp.expand_dims(jnp.asarray(mean), (-1, -2))
        self.var = jnp.expand_dims(jnp.asarray(var), (-1, -2))
        self.std = jnp.sqrt(self.var)

    def scale_to_standard(self, a):
        if a.dtype == jnp.dtype(jnp.float64):
            dest_type = jnp.float64
        else:
            dest_type = jnp.float32
        return (a - self.mean.astype(dest_type)) / self.std.astype(dest_type)

    def scale(self, a):
        return self.scale_to_standard(a)

    def scale_from_standard(self, a):
        if a.dtype == jnp.dtype(jnp.float64):
            dest_type = jnp.float64
        else:
            dest_type = jnp.float32
        return (a * self.std.astype(dest_type)) + self.mean.astype(dest_type)

    def unscale(self, a):
        return self.scale_from_standard(a)

    def tree_flatten(self):
        return (self.mean, self.var, self.std), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        mean, var, std = children
        obj = cls.__new__(cls)
        obj.mean = mean
        obj.var = var
        obj.std = std
        return obj


def register_pytree_dataclass(cls):
    fields = tuple(f.name for f in dataclasses.fields(cls))

    def flatten(obj):
        return [getattr(obj, name) for name in fields], None

    def unflatten(aux_data, flat_contents):
        return cls(**dict(zip(fields, flat_contents, strict=True)))

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls


def strided_scan(f, init, xs, length=None, reverse=False, unroll=1, stride=1):
    if reverse:
        raise ValueError("reversed scan is not supported")
    stride = operator.index(stride)
    if stride < 1:
        raise ValueError(f"illegal stride value {stride}")
    return ppx.sliced_scan(
        f,
        init,
        xs,
        length=length,
        reverse=reverse,
        unroll=unroll,
        start=stride - 1,
        stop=None,
        step=stride,
    )


def chunked_vmap(fun, chunk_size):
    return ppx.chunked_vmap(fun, chunk_size)


@jax.tree_util.register_pytree_node_class
class EquinoxTrainState:
    # Fields:
    # - net
    # - optim
    # - optim_state
    # - _param_filter

    def __init__(self, net, optim, *, param_filter=eqx.is_inexact_array):
        self.net = net
        self.optim = optim
        self._param_filter = param_filter
        self.optim_state = self.optim.init(eqx.filter(net, self._param_filter))

    def apply_updates(self, grads):
        updates, new_opt_state = self.optim.update(grads, self.optim_state, eqx.filter(self.net, self._param_filter))
        new_net = eqx.apply_updates(self.net, updates)
        # New object
        cls = type(self)
        obj = cls.__new__(cls)
        obj.net = new_net
        obj.optim_state = new_opt_state
        obj.optim = self.optim
        obj._param_filter = self._param_filter
        return obj

    def tree_flatten(self):
        children = (self.net, self.optim_state)
        aux_data = (self.optim, self._param_filter)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        net, optim_state = children
        optim, _param_filter = aux_data
        obj.net = net
        obj.optim_state = optim_state
        obj.optim = optim
        obj._param_filter = _param_filter
        return obj


def filter_scan(f, init, xs, length=None, reverse=False, unroll=1):
    # NOTE: Filtering applies only to the carry/init values

    init_var, carry_static = eqx.partition(init, filter_spec=eqx.is_array_like)

    @functools.wraps(f)
    def filtered_f(carry, x):
        carry = eqx.combine(carry, carry_static)
        new_carry, new_y = f(carry, x)
        new_carry = eqx.filter(new_carry, filter_spec=eqx.is_array_like)
        return new_carry, new_y

    final_carry, ys = jax.lax.scan(
        filtered_f,
        init_var,
        xs,
        length=length,
        reverse=reverse,
        unroll=unroll,
    )
    final_carry = eqx.combine(final_carry, carry_static)

    return final_carry, ys


def make_json_serializable(pytree):
    return jax.tree_util.tree_map(lambda leaf: leaf.item() if leaf.size == 1 else leaf.tolist(), pytree)



def hvp(f, x, v):
    # Return hess(f)(x) * v
    primals, tangents = jax.jvp(jax.grad(f), x, v)
    return tangents


def trace_jac(f, x):
    # Return trace(jac(f)(x))
    assert x.ndim == 1
    return jnp.trace(jax.jacrev(f)(x))


def trace_jac_hutch(f, x, rng, num_samples=10):
    # Estimate trace(jac(f)(x))
    assert x.ndim == 1
    g_mat = jax.random.rademacher(rng, shape=(num_samples, x.shape[0]), dtype=x.dtype)
    _primals_out, vjp_f = jax.vjp(f, x)
    hutch_est = jnp.einsum("nd,nd->n", jax.vmap(vjp_f)(g_mat)[0], g_mat)
    return jnp.mean(hutch_est)


def checkpoint_chunked_scan(f, init, xs, length=None, chunk_size=5):
    return ppx.checkpoint_chunked_scan(
        f, init, xs, length=length, chunk_size=chunk_size
    )
