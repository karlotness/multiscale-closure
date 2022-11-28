import math
import functools
import operator
import jax
import jax.numpy as jnp
import equinox as eqx


def strided_scan(f, init, xs, length=None, reverse=False, unroll=1, stride=1):
    if reverse:
        raise ValueError(f"reversed scan is not supported")
    stride = operator.index(stride)
    if stride < 1:
        raise ValueError(f"illegal stride value {stride}")
    elif stride == 1:
        # Trivial case: stride=1 -> do normal scan
        return jax.lax.scan(f, init, xs, length=length, reverse=reverse, unroll=unroll)
    # Continue with full processing
    target_length = _get_target_length(xs, length)
    chunks, remainder = divmod(target_length, stride)
    array_head = chunks * stride
    # Process xs into chunks and remainder
    if xs is None:
        chunked_xs = None
        remainder_xs = None
    else:
        chunked_xs = xs[:array_head].reshape((chunks, stride) + xs.shape[1:])
        remainder_xs = xs[-remainder:]
    # Do main chunked scan
    def inner_scan(carry, x):
        f_carry, _y = carry
        new_carry, y = f(f_carry, x)
        return (new_carry, y), None

    def outer_scan(carry, x):
        dummy_carry, dummy_y = f(carry, x[0] if x is not None else None)
        (last_carry, y), _ = jax.lax.scan(inner_scan, (carry, dummy_y), x, length=stride, reverse=reverse, unroll=unroll)
        return last_carry, y

    carry, ys = jax.lax.scan(outer_scan, init, chunked_xs, length=chunks, reverse=reverse, unroll=unroll)
    # Do remainder scan
    def remainder_scan(carry, x):
        new_carry, _y = f(carry, x)
        return new_carry, None

    carry, _remys = jax.lax.scan(remainder_scan, carry, remainder_xs, length=remainder, reverse=reverse, unroll=unroll)
    # Return stacks from the chunk, and final state from the remainder
    return carry, ys


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


def ilog2(n):
    n = operator.index(n)
    if n < 0:
        raise ValueError(f"invalid integer log2 for negative value {n}")
    elif n == 0:
        raise ValueError(f"invalid integer log2 for zero value")
    return n.bit_length() - 1


def icbrt(n):
    n = operator.index(n)
    if n == 0:
        return 0
    if n < 0:
        return icbrt(-n)
    r = 1
    while r**3 < n:
        r += 1
    return r - (1 if r**3 > n else 0)


def _get_target_length(xs, length):
    if xs is not None:
        leaf_lengths = set(operator.index(x.shape[0]) for x in jax.tree_util.tree_leaves(xs))
        if len(leaf_lengths) != 1:
            raise ValueError(f"inconsistent lengths for tree input: {set(leaf_lengths)}")
        leaf_lengths = next(iter(leaf_lengths))
    else:
        leaf_lengths = None
    if length is not None:
        length = operator.index(length)
    match (leaf_lengths, length):
        case (None, None):
            raise ValueError("invalid target length for None inputs")
        case (l, None) | (None, l):
            return l
        case (la, lb) if la == lb:
            return la
        case _:
            raise ValueError(f"ambiguous lengths for scan {leaf_lengths} vs {length}")


def checkpoint_chunked_scan(f, init, xs, length=None, chunk_size=5):
    chunk_size = operator.index(chunk_size)
    target_length = _get_target_length(xs, length)
    num_chunks = target_length // chunk_size
    if target_length <= chunk_size:
        # Trivial, just do a normal scan
        return jax.lax.scan(f, init, xs, length=length)
    return easy_nested_scan(
        f,
        init,
        xs,
        length=length,
        nested_lengths=[num_chunks, chunk_size],
        continue_scan_fn=jax.lax.scan,
    )


def checkpoint_log2_scan(f, init, xs, length=None):
    target_length = _get_target_length(xs, length)
    log_depth = ilog2(target_length)
    if log_depth < 2:
        return jax.lax.scan(f, init, xs, length=length)
    return easy_nested_scan(
        f,
        init,
        xs,
        length=length,
        nested_lengths=[2] * log_depth,
        continue_scan_fn=checkpoint_log2_scan,
    )


def checkpoint_sqrt_scan(f, init, xs, length=None):
    target_length = _get_target_length(xs, length)
    sqrt_length = math.isqrt(target_length)
    if sqrt_length < 2:
        return jax.lax.scan(f, init, xs, length=length)
    return easy_nested_scan(
        f,
        init,
        xs,
        length=length,
        nested_lengths=[sqrt_length] * 2,
        continue_scan_fn=jax.lax.scan,
    )


def checkpoint_cbrt_scan(f, init, xs, length=None):
    target_length = _get_target_length(xs, length)
    cbrt_length = icbrt(target_length)
    if cbrt_length < 2:
        return checkpoint_sqrt_scan(f, init, xs, length=length)
    return easy_nested_scan(
        f,
        init,
        xs,
        length=length,
        nested_lengths=[cbrt_length] * 3,
        continue_scan_fn=checkpoint_sqrt_scan,
    )


def easy_nested_scan(f, init, xs, length, *, nested_lengths, continue_scan_fn=jax.lax.scan):
    target_length = _get_target_length(xs, length)
    if target_length <= 1:
        return jax.lax.scan(f, init, xs, length=length)
    pre_length = math.prod(nested_lengths)
    remainder = target_length - pre_length
    pre_xs = jax.tree_map(operator.itemgetter(slice(None, pre_length)), xs)
    post_xs = jax.tree_map(operator.itemgetter(slice(pre_length, None)), xs)
    pre_carry, pre_out = nested_checkpoint_scan(
        f,
        init,
        pre_xs,
        pre_length,
        nested_lengths=nested_lengths,
        scan_fn=jax.lax.scan,
        checkpoint_fn=functools.partial(jax.checkpoint, prevent_cse=False),
    )
    if remainder > 0:
        post_carry, post_out = continue_scan_fn(
            f,
            pre_carry,
            post_xs,
            length=remainder,
        )
        result = jax.tree_map(lambda a, b: jnp.concatenate([a, b]), pre_out, post_out)
    else:
        post_carry = pre_carry
        result = pre_out
    return post_carry, result


def nested_checkpoint_scan(f, init, xs, length, *, nested_lengths, scan_fn=jax.lax.scan, checkpoint_fn=jax.checkpoint):
  """A version of lax.scan that supports recursive gradient checkpointing.

  The interface of `nested_checkpoint_scan` exactly matches lax.scan, except for
  the required `nested_lengths` argument.

  The key feature of `nested_checkpoint_scan` is that gradient calculations
  require O(max(nested_lengths)) memory, vs O(prod(nested_lengths)) for unnested
  scans, which it achieves by re-evaluating the forward pass
  `len(nested_lengths) - 1` times.

  `nested_checkpoint_scan` reduces to `lax.scan` when `nested_lengths` has a
  single element.

  Args:
    f: function to scan over.
    init: initial value.
    xs: scanned over values.
    length: leading length of all dimensions
    nested_lengths: required list of lengths to scan over for each level of
      checkpointing. The product of nested_lengths must match length (if
      provided) and the size of the leading axis for all arrays in ``xs``.
    scan_fn: function matching the API of lax.scan
    checkpoint_fn: function matching the API of jax.checkpoint.

  Returns:
    Carry and output values.
  """
  # Copyright 2022 Google LLC.
  # SPDX-License-Identifier: Apache-2.0
  # THIS CODE BY shoyer on GITHUB: https://github.com/google/jax/issues/2139
  if length is not None and length != math.prod(nested_lengths):
    raise ValueError(f'inconsistent {length=} and {nested_lengths=}')

  def nested_reshape(x):
    x = jnp.asarray(x)
    new_shape = tuple(nested_lengths) + x.shape[1:]
    return x.reshape(new_shape)

  sub_xs = jax.tree_map(nested_reshape, xs)
  return _inner_nested_scan(f, init, sub_xs, nested_lengths, scan_fn,
                            checkpoint_fn)


def _inner_nested_scan(f, init, xs, lengths, scan_fn, checkpoint_fn):
  """Recursively applied scan function."""
  # Copyright 2022 Google LLC.
  # SPDX-License-Identifier: Apache-2.0
  # THIS CODE BY shoyer on GITHUB: https://github.com/google/jax/issues/2139
  if len(lengths) == 1:
    return scan_fn(f, init, xs, lengths[0])

  @checkpoint_fn
  def sub_scans(carry, xs):
    return _inner_nested_scan(f, carry, xs, lengths[1:], scan_fn, checkpoint_fn)

  carry, out = scan_fn(sub_scans, init, xs, lengths[0])
  stacked_out = jax.tree_map(jnp.concatenate, out)
  return carry, stacked_out


CHECKPOINT_SCANNERS = {
    "scan": jax.lax.scan,
    "log2": checkpoint_log2_scan,
    "sqrt": checkpoint_sqrt_scan,
    "cbrt": checkpoint_cbrt_scan,
}
