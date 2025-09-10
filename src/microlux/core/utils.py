"""
Utility functions used by the adaptive contour integrator.

Includes array insert/delete kernels with static shapes, a gradient‑stopping
wrapper for JAX transforms, and a thin warning helper used when capacity is
exceeded.
"""

import warnings

import jax
import jax.numpy as jnp
from jax import lax


def insert_body(carry, k):
    """
    Insert a block into a fixed‑size array at index `idx[k]`.

    Parameters:
        carry (tuple): `(array, add_array, idx, add_number)` working state.
        k (int): Position index into `idx` and `add_number`.

    Returns:
        tuple: Updated carry and loop index for `lax.scan`.
    """
    array, add_array, idx, add_number = carry
    ite = jnp.arange(array.shape[0])
    mask = ite < idx[k]
    array = jnp.where(mask[:, None], array, jnp.roll(array, add_number[k], axis=0))
    mask2 = (ite >= idx[k]) & (ite < idx[k] + add_number[k])
    add_array = jnp.roll(add_array, idx[k], axis=0)
    array = jnp.where(mask2[:, None], add_array, array)
    add_array = jnp.roll(add_array, -1 * add_number[k] - idx[k], axis=0)
    idx += add_number[k]
    return (array, add_array, idx, add_number), k


def custom_insert(array, idx, add_array):
    """
    Insert rows into `array` without changing its final shape.

    Extra capacity is trimmed from the tail after a standard `jnp.insert`.

    Parameters:
        array (jax.Array): Target array with fixed leading dimension.
        idx (jax.Array): Insert indices per row (−1 entries are ignored upstream).
        add_array (jax.Array): Rows to insert.

    Returns:
        jax.Array: New array with rows inserted and size restored.
    """
    final_array = jnp.insert(array, idx, add_array, axis=0)
    final_array = final_array[: array.shape[0]]
    return final_array


def delete_body(carry, k):
    """
    Delete a single row by rolling the tail forward within fixed shape.
    """
    array, ite2, delidx = carry
    mask = ite2 < delidx[k]
    array = jnp.where(mask[:, None], array, jnp.roll(array, -1, axis=0))
    delidx -= (~mask).any()
    return (array, ite2, delidx), k


def custom_delete(array, delidx):
    """
    Delete selected rows from `array` while keeping static shape.

    Parameters:
        array (jax.Array): Target array with fixed leading dimension.
        delidx (jax.Array): Indices to delete (values beyond size are ignored).

    Returns:
        jax.Array: Array with deletions compacted and padded at the tail.
    """
    fill_value = array[-1]
    ite = jnp.arange(array.shape[0])
    carry, _ = lax.scan(delete_body, (array, ite, delidx), jnp.arange(delidx.shape[0]))
    array, _, _ = carry
    array = jnp.where(
        (ite < ite.size - (delidx < array.shape[0]).sum())[:, None], array, fill_value
    )
    return array


def stop_grad_wrapper(func):
    """
    Wrap a function so inputs/outputs have gradients stopped.

    Useful to keep reverse‑mode graphs compact when running while/scan loops.
    """
    def wrapper(*args, **kwargs):
        args = jax.lax.stop_gradient(args)
        kwargs = jax.lax.stop_gradient(kwargs)
        return jax.lax.stop_gradient(func(*args, **kwargs))

    return wrapper


def warn_length_not_enough(required_length, Max_length):
    """
    Emit a warning when sampling capacity is exceeded.
    """
    warnings.warn(
        "No enough space to insert new samplings, which may cause the error larger than the tolerance. Current length vs max length: {} vs {}. Consider incresing default_strategy parameters.".format(
            required_length, Max_length - 2
        )
    )
