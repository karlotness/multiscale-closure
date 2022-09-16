import math
import functools
import operator
import jax
import jax.numpy as jnp


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
    # This code by me
    if length is not None:
        return operator.index(length)
    else:
        # Ensure lengths are unique
        leaf_lengths = set(operator.index(x.shape[0]) for x in jax.tree_util.tree_leaves(xs))
        if len(leaf_lengths) != 1:
            raise ValueError(f"inconsistent lengths for tree input: {leaf_lengths}")
        return next(iter(leaf_lengths))


def checkpoint_binomial_scan(f, init, xs, length=None):
    target_length = _get_target_length(xs, length)
    if target_length <= 1:
        return jax.lax.scan(f, init, xs, length=None)


def checkpoint_log2_scan(f, init, xs, length=None):
    target_length = _get_target_length(xs, length)
    if target_length <= 1:
        return jax.lax.scan(f, init, xs, length=None)
    # Compute log2 levels
    log_depth = ilog2(target_length)
    pre_length = 2**log_depth
    remainder = target_length - pre_length
    pre_slicer = slice(None, pre_length)
    post_slicer = slice(pre_length, None)
    if xs is not None:
        pre_xs = jax.tree_map(operator.itemgetter(pre_slicer), xs)
        post_xs = jax.tree_map(operator.itemgetter(post_slicer), xs)
    else:
        pre_xs = None
        post_xs = None
    pre_carry, pre_out = nested_checkpoint_scan(
        f,
        init,
        pre_xs,
        pre_length,
        nested_lengths=[2]*log_depth,
    )
    if remainder > 0:
        post_carry, post_out = checkpoint_log2_scan(
            f,
            pre_carry,
            post_xs,
            remainder,
        )
        result = jnp.concatenate([pre_out, post_out])
    else:
        post_carry = pre_carry
        result = pre_out
    return post_carry, result


def checkpoint_sqrt_scan(f, init, xs, length=None):
    target_length = _get_target_length(xs, length)
    if target_length <= 1:
        return jax.lax.scan(f, init, xs, length=None)
    sqrt_length = math.isqrt(target_length)
    remainder = target_length - (sqrt_length**2)
    pre_slicer = slice(None, sqrt_length**2)
    post_slicer = slice(sqrt_length**2, None)
    if xs is not None:
        pre_xs = jax.tree_map(operator.itemgetter(pre_slicer), xs)
        post_xs = jax.tree_map(operator.itemgetter(post_slicer), xs)
    else:
        pre_xs = None
        post_xs = None
    pre_carry, pre_out = nested_checkpoint_scan(
        f,
        init,
        pre_xs,
        sqrt_length**2,
        nested_lengths=[sqrt_length, sqrt_length],
    )
    if remainder > 0:
        post_carry, post_out = nested_checkpoint_scan(
            f,
            pre_carry,
            post_xs,
            remainder,
            nested_lengths=[remainder],
        )
        result = jnp.concatenate([pre_out, post_out])
    else:
        post_carry = pre_carry
        result = pre_out
    return post_carry, result


def checkpoint_cbrt_scan(f, init, xs, length=None):
    target_length = _get_target_length(xs, length)
    cbrt_length = icbrt(target_length)
    if target_length <= 1 or cbrt_length < 2:
        return jax.lax.scan(f, init, xs, length=None)
    pre_length = cbrt_length**3
    remainder = target_length - (pre_length)
    pre_slicer = slice(None, pre_length)
    post_slicer = slice(pre_length, None)
    if xs is not None:
        pre_xs = jax.tree_map(operator.itemgetter(pre_slicer), xs)
        post_xs = jax.tree_map(operator.itemgetter(post_slicer), xs)
    else:
        pre_xs = None
        post_xs = None
    pre_carry, pre_out = nested_checkpoint_scan(
        f,
        init,
        pre_xs,
        pre_length,
        nested_lengths=[cbrt_length, cbrt_length, cbrt_length],
    )
    if remainder > 0:
        post_carry, post_out = checkpoint_sqrt_scan(
            f,
            pre_carry,
            post_xs,
            remainder,
        )
        result = jnp.concatenate([pre_out, post_out])
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

  @functools.partial(checkpoint_fn, prevent_cse=False)
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
