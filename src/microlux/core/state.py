"""
Typed state containers used by the adaptive contour integration.

`Iterative_State` holds per‑row sampling angles, roots, parity, and assorted
flags. `Error_State` tracks magnification, error histogram, tolerance targets,
and termination bookkeeping. `get_default_state` constructs placeholder states
with pre‑shaped arrays for static‑shape execution.
"""

import warnings
from typing import NamedTuple, Union

import jax
import jax.numpy as jnp


MAX_CAUSTIC_INTERSECT_NUM = 15


class Iterative_State(NamedTuple):
    """
    State of the iterative sampling/solving process.

    Attributes:
        sample_num (int): Number of valid sampling rows.
        theta (jax.Array): Angles per row.
        roots (jax.Array): Complex roots (images) per row.
        parity (jax.Array): Image parity per row.
        ghost_roots_distant (jax.Array): Distance proxy to flag buried images.
        sort_flag (Union[bool, jax.Array]): Whether rows are already stably sorted.
        Is_create (jax.Array): Encoded creation/destruction events.
    """

    sample_num: int
    theta: jax.Array
    roots: jax.Array
    parity: jax.Array
    ghost_roots_distant: jax.Array
    sort_flag: Union[bool, jax.Array]
    Is_create: jax.Array = jnp.zeros((4, MAX_CAUSTIC_INTERSECT_NUM), dtype=int)


class Error_State(NamedTuple):
    """
    Aggregates magnification and error tracking for adaptive sampling.

    Attributes:
        mag (jax.Array): Current magnification (pre‑normalized and normalized forms).
        mag_no_diff (int): Counter of consecutive negligible changes in `mag`.
        outloop (int): Total deletions due to invalid parity across iterations.
        error_hist (jax.Array): Per‑row error estimates for adaptive control.
        epsilon (float): Absolute tolerance target.
        epsilon_rel (float): Relative tolerance target.
        exceed_flag (bool): True if sampling length exceeded buffer capacity.
    """

    mag: jax.Array
    mag_no_diff: int
    outloop: int
    error_hist: jax.Array
    epsilon: float
    epsilon_rel: float
    exceed_flag: bool = False


def get_default_state(total_length: int) -> tuple[Iterative_State, Error_State]:
    """
    Construct pre‑shaped placeholder states for static‑shape execution.

    Parameters:
        total_length (int): Total buffer length to allocate for rows.

    Returns:
        tuple: `(Iterative_State, Error_State)` initialized with sentinels.
    """

    pad_value = [jnp.nan, 0.0, jnp.nan + 1j * jnp.nan, jnp.nan, jnp.nan, 0.0, True]
    shape = [1, 1, 5, 5, 1, 1, 1]
    init_fun = lambda x, y: jnp.full((total_length, y), x)
    theta, error_hist, roots, parity, ghost_roots_dis, buried_error, sort_flag = (
        jax.tree.map(init_fun, pad_value, shape)
    )
    sample_n = 0
    roots_state = Iterative_State(
        sample_n, theta, roots, parity, ghost_roots_dis, sort_flag
    )
    error_state = Error_State(jnp.array([0.0]), 0, 0, error_hist, 1e-3, 1e-3)

    return roots_state, error_state
