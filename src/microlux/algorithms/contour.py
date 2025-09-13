"""
Adaptive contour integration for binary microlensing (JAX).

This module implements a hierarchical, adaptive contour integration to compute
finite‑source magnifications for binary lenses while remaining JIT‑friendly.
Because JAX requires static shapes, arrays are grown in staged increments
(`default_strategy`) rather than allocating a single large buffer. To support
reverse‑mode differentiation through `lax.while_loop`, an analytic wrapper is
provided that stops gradients where needed and recomputes required quantities
with a compact computation graph.

Main functions:
- `contour_integral(...)`: Public entry point. Runs the adaptive loop, grows
  buffers as needed, and returns `(mag, result_state)`.
- `contour_init(...)`: Builds initial sampling, solves for roots/images, and
  returns the loop carry state.
- `cond_fun(...)`: Stopping condition combining relative/absolute error,
  minimum spacing, stability counters, and buffer capacity.
- `while_body_fun(...)`: Adds samples at high‑error intervals, resolves new
  images, and updates magnification/error histograms.
- `update_mag(...)`: Recomputes magnification and bookkeeping after insertions.

Notes:
- Uses `Iterative_State` and `Error_State` from `microlux.utils` to track the
  sampling/roots and error state, respectively.
- Relies on polynomial root operations and error estimators from sibling
  modules under `microlux`.
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax

from ..core.lens_equation import (
    get_zeta_l,
    refine_gradient,
)
from .error_estimator import error_sum
from .solution import (
    add_points,
    find_create_points,
    get_buried_error,
    get_poly_coff,
    get_real_roots,
    get_sorted_roots,
)
from ..core.state import (
    Error_State,
    Iterative_State,
)
from ..core.utils import (
    insert_body,
    stop_grad_wrapper,
    warn_length_not_enough,
)


# jax config is set in package __init__


def analytic_wrapper(trajectory_l, rho, s, q, roots_state, mag_state):
    """
    Wrapper for hierarchical contour integration with analytic chain rule.

    This wrapper keeps the reverse‑mode autodiff graph compact and stable
    across `lax.while_loop` by stopping gradients where appropriate and then
    recomputing the necessary quantities analytically.

    Parameters:
        trajectory_l (jax.Array): Source trajectory in the low‑mass coordinate system.
        rho (float): Source radius.
        s (float): Binary separation (Einstein radius units).
        q (float): Mass ratio (m2/m1).
        roots_state (Iterative_State): Current sampling/roots state.
        mag_state (Error_State): Current magnification/error state.

    Returns:
        tuple: `(trajectory_l, rho, s, q, roots_state, mag_state)` with
        `mag_state.mag` updated using the analytic correction terms.
    """

    sample_num, theta, roots, parity, ghost_roots_distant, sort_flag, is_create = (
        roots_state
    )
    mask = ~jnp.isnan(roots)
    roots_nan_filled = jnp.where(mask, roots, 100.0)
    parity = jnp.where(mask, parity, 0.0)
    theta = jnp.where(mask, theta, 0.0)

    # stop gradient to avoid nan in reverse mode
    zeta_l = get_zeta_l(rho, trajectory_l, theta)
    roots_nan_filled = refine_gradient(zeta_l, q, s, roots_nan_filled)

    adjacent_valid_mask = mask[1:] & mask[:-1]
    roots_state_refine_grad = Iterative_State(
        sample_num,
        theta,
        roots_nan_filled,
        parity,
        ghost_roots_distant,
        sort_flag,
        is_create,
    )
    mag_ndarray = (
        (roots_nan_filled.imag[0:-1] + roots_nan_filled.imag[1:])
        * (roots_nan_filled.real[0:-1] - roots_nan_filled.real[1:])
        * parity[0:-1]
    )
    mag = 1 / 2 * jnp.sum(jnp.where(adjacent_valid_mask, mag_ndarray, 0.0).sum(axis=1))

    _, magc, parab = error_sum(roots_state_refine_grad, rho, q, s, mask)
    # parab = jax.lax.stop_gradient(parab)
    mag = (mag + magc + parab) / (jnp.pi * rho**2)

    mag_state = mag_state._replace(mag=mag)
    return (trajectory_l, rho, s, q, roots_state, mag_state)


@partial(jax.jit, static_argnames=["default_strategy", "analytic"])
def contour_integral(
    trajectory_l, tol, retol, rho, s, q, default_strategy=(60, 80, 150), analytic=True
) -> Tuple[jnp.ndarray, Tuple]:
    """
    Adaptive contour integration using a gradually grown, pre‑shaped array.

    For JIT compilation, array shapes must be static. Optimal sampling length is
    unknown a priori, so using a single large buffer wastes memory/time, while a
    small one may terminate early and violate tolerances. This function grows
    the working arrays in stages (hierarchical strategy) to balance performance
    and accuracy, and optionally applies an analytic chain rule to keep the AD
    graph compact across loops.

    Parameters:
        trajectory_l (jax.Array): Source trajectory in the low‑mass frame.
        tol (float): Absolute tolerance on magnification error.
        retol (float): Relative tolerance on magnification error.
        rho (float): Source radius.
        s (float): Binary separation (Einstein radius units).
        q (float): Mass ratio (m2/m1).
        default_strategy (tuple[int, int, int]): Array growth plan per layer.
            Example `(60, 80, 150)` yields maximum length `60 -> 140 -> 290`.
        analytic (bool): If True, use analytic chain rule to enable reverse‑mode
            differentiation across `while_loop` and reduce graph size. This may
            be slower when gradients are not required. Default True.

    Returns:
        tuple: `(mag, result)` where `mag` is the scalar magnification
        at the first trajectory point and `result` is the final state tuple
        `(trajectory_l, rho, s, q, roots_state, mag_state)`.
    """

    # JIT requires static shapes. We therefore extend the buffer length in
    # stages instead of allocating an excessively large array up front.
    # Current default cumulative length: 60 + 80 + 150 = 290.
    @partial(jax.jit, static_argnums=(-1,))
    def reshape_fun(carry, arraylength):
        """
        Extend state arrays by `arraylength`, padding new slots with sentinels.

        Parameters:
            carry (tuple): Current `(trajectory_l, rho, s, q, roots_state, mag_state)`.
            arraylength (int): Number of rows to append to state buffers.

        Returns:
            tuple: Updated carry with resized arrays; previous values preserved.
        """
        (trajectory_l, rho, s, q, roots_state, mag_state) = carry

        sample_num, theta, roots, parity, ghost_roots_distant, sort_flag, is_created = (
            roots_state
        )

        error_hist = mag_state.error_hist
        # Reshape arrays and fill appended regions with default sentinels.
        pad_list = [theta, error_hist, roots, parity, ghost_roots_distant, sort_flag]
        pad_value = [jnp.nan, 0.0, jnp.nan, jnp.nan, jnp.nan, True]
        pad_fun = lambda x, y: jnp.pad(
            x, ((0, arraylength), (0, 0)), "constant", constant_values=y
        )
        if analytic:
            pad_fun = stop_grad_wrapper(pad_fun)
            padded_list = jax.tree.map(pad_fun, pad_list, pad_value)
            padded_list = jax.lax.stop_gradient(padded_list)
        else:
            padded_list = jax.tree.map(pad_fun, pad_list, pad_value)

        theta, error_hist, roots, parity, ghost_roots_distant, sort_flag = padded_list
        carry = (
            trajectory_l,
            rho,
            s,
            q,
            Iterative_State(
                sample_num, theta, roots, parity, ghost_roots_distant, sort_flag, is_created
            ),
            Error_State(
                mag_state.mag,
                mag_state.mag_no_diff,
                mag_state.outloop,
                error_hist,
                mag_state.epsilon,
                mag_state.epsilon_rel,
            ),
        )
        return carry

    def secondary_contour(carry):
        """
        Run a refinement pass using the extended array capacity.

        Parameters:
            carry (tuple): `(result, result_last, add_length, max_array_length)`.

        Returns:
            tuple: Updated `(result, result_last, max_array_length)` after
            executing the inner while‑loop once more with the larger buffers.
        """
        result, result_last, add_length, max_array_length = carry

        # Choose implementation for adding points (while_loop or scan). In the
        # current JAX version, reverse‑mode differentiation through while_loop
        # needs special handling; the analytic wrapper provides that.

        # while loop

        # resultnew,resultlast=lax.while_loop(cond_fun,while_body_fun,(resultlast,resultlast))
        if analytic:
            stop_grad_loop = stop_grad_wrapper(
                lambda x: lax.while_loop(cond_fun, while_body_fun, x)
            )
            result_new, result_last = stop_grad_loop((result_last, result_last))
            result_new = analytic_wrapper(
                trajectory_l, rho, s, q, result_new[-2], result_new[-1]
            )
        else:
            result_new, result_last = lax.while_loop(
                cond_fun, while_body_fun, (result_last, result_last)
            )

        max_array_length += add_length
        return result_new, result_last, max_array_length

    # Initial pass (using the first stage of default_strategy)

    if analytic:
        carry = stop_grad_wrapper(contour_init)(
            rho,
            s,
            q,
            trajectory_l,
            tol,
            epsilon_rel=retol,
            inite=default_strategy[0] - 3,
            n_ite=default_strategy[0],
        )
        stop_grad_loop = lambda x: lax.while_loop(cond_fun, while_body_fun, x)
        result_no_grad, result_last = stop_grad_wrapper(stop_grad_loop)((carry, carry))
        result = analytic_wrapper(
            trajectory_l, rho, s, q, result_no_grad[-2], result_no_grad[-1]
        )

    else:
        carry = contour_init(
            rho,
            s,
            q,
            trajectory_l,
            tol,
            epsilon_rel=retol,
            inite=default_strategy[0] - 3,
            n_ite=default_strategy[0],
        )
        result, result_last = lax.while_loop(cond_fun, while_body_fun, (carry, carry))

    max_array_length = default_strategy[0]
    for i in range(len(default_strategy) - 1):
        add_length = default_strategy[i + 1]

        result_last = reshape_fun(result_last, add_length)
        result = reshape_fun(result, add_length)

        result, result_last, max_array_length = lax.cond(
            (result[-2].sample_num < max_array_length - 2),
            lambda x: (x[0], x[1], x[-1]),
            secondary_contour,
            (result, result_last, add_length, max_array_length),
        )

    (trajectory_l, rho, s, q, roots_state, mag_state) = result

    condition = roots_state.sample_num < max_array_length - 2

    def update_result_fun(carry):
        # Mark that the sampling capacity was exceeded to warn the caller.
        result_last_local = carry[1]
        mag_state_local = result_last_local[-1]
        jax.debug.callback(
            warn_length_not_enough, carry[0][-2].sample_num, max_array_length
        )
        mag_state_new = mag_state_local._replace(exceed_flag=True)
        if analytic:
            result_last_local, mag_state_new = (
                jax.lax.stop_gradient(result_last_local),
                jax.lax.stop_gradient(mag_state_new),
            )
            result_last_update = analytic_wrapper(
                trajectory_l, rho, s, q, result_last_local[-2], mag_state_new
            )
        else:
            result_last_update = (
                trajectory_l,
                rho,
                s,
                q,
                result_last_local[-2],
                mag_state_new,
            )
        return result_last_update

    result = lax.cond(
        condition, lambda x: x[0], update_result_fun, (result, result_last)
    )
    return (result[-1].mag[0], result)


@partial(jax.jit, static_argnames=("inite", "n_ite"))
def contour_init(rho, s, q, trajectory_l, epsilon, epsilon_rel=0, inite=30, n_ite=60):
    """
    Initialize contour integration with a fixed working length.

    Builds the initial sampling grid, solves for images/roots, computes the
    first magnification estimate and error histogram, and returns the carry
    tuple for the main loop.

    Parameters:
        rho (float): Source radius.
        s (float): Binary separation (Einstein radius units).
        q (float): Mass ratio (m2/m1).
        trajectory_l (jax.Array): Source trajectory in the low‑mass frame.
        epsilon (float): Absolute tolerance.
        epsilon_rel (float, optional): Relative tolerance. Default 0.
        inite (int, optional): Initial number of samples. Default 30.
        n_ite (int, optional): Buffer length for this stage. Default 60.

    Returns:
        tuple: `(trajectory_l, rho, s, q, roots_state, mag_state)` for use as
        the loop carry in `while_body_fun`.
    """
    m1 = 1 / (1 + q)
    m2 = q / (1 + q)
    sample_num = inite
    theta = jnp.where(
        jnp.arange(n_ite) < inite,
        jnp.resize(jnp.linspace(0, 2 * jnp.pi, inite), n_ite),
        jnp.nan,
    )[:, None]
    error_hist = jnp.ones(n_ite)
    zeta_l = get_zeta_l(rho, trajectory_l, theta)
    coeff = get_poly_coff(zeta_l, s, q / (1 + q))
    roots, parity, ghost_roots_distant, outloop, coeff, zeta_l, theta, _ = get_real_roots(
        coeff, zeta_l, theta, s, m1, m2, jnp.arange(n_ite)
    )

    buried_error = get_buried_error(ghost_roots_distant, sample_num) / jnp.pi / rho**2

    # Whether roots in each row still need sorting/matching.
    sort_flag = jnp.where(jnp.arange(n_ite) < inite, False, True)[:, None]
    sort_flag = sort_flag.at[0].set(True)  # First row is already aligned.

    indices_update, sort_flag = get_sorted_roots(roots, parity, sort_flag, n_ite)
    roots = roots[jnp.arange(n_ite)[:, None], indices_update]
    parity = parity[jnp.arange(n_ite)[:, None], indices_update]

    is_created = find_create_points(roots, parity, sample_num)
    roots_state = Iterative_State(
        sample_num, theta, roots, parity, ghost_roots_distant, sort_flag, is_created
    )

    # Compute the first magnification and error estimate.
    mag_no_diff_num = 0
    mag = (
        1
        / 2
        * jnp.nansum(
            jnp.nansum(
                (roots.imag[0:-1] + roots.imag[1:])
                * (roots.real[0:-1] - roots.real[1:])
                * parity[0:-1],
                axis=0,
            )
        )
    )
    error_hist, magc, parab = error_sum(roots_state, rho, q, s)
    mag = (mag + magc + parab) / (jnp.pi * rho**2)
    error_hist += buried_error
    mag_state = Error_State(
        mag, mag_no_diff_num, outloop, error_hist, epsilon, epsilon_rel
    )

    carry = (trajectory_l, rho, s, q, roots_state, mag_state)

    return carry


def cond_fun(carry):
    """
    Stopping condition for the adaptive sampling loop.

    Uses a combination of absolute/relative error thresholds, minimum angle
    spacing, consecutive non‑change counts, and buffer capacity to decide
    whether to continue adding samples.

    Parameters:
        carry (tuple): `(current_carry, last_carry)`.

    Returns:
        bool: True to continue iterating; False to stop.
    """
    carry, carrylast = carry
    # Decide whether to continue the loop using relative/absolute error.
    (trajectory_l, rho, s, q, roots_state, mag_state) = carry
    theta = roots_state.theta
    sample_num = roots_state.sample_num
    error_hist = mag_state.error_hist
    epsilon = mag_state.epsilon
    epsilon_rel = mag_state.epsilon_rel
    mag = mag_state.mag
    mag_no_diff_num = mag_state.mag_no_diff
    outloop = mag_state.outloop
    K = 2
    max_array_length = jnp.shape(theta)[0]
    mini_interval = jnp.nanmin(jnp.abs(jnp.diff(theta, axis=0)))
    abs_mag_cond = jnp.nansum(error_hist) > epsilon

    # Absolute‑only stopping (unused but kept for reference):
    # abs_mag_cond2 = (error_hist > epsilon / jnp.sqrt(sample_n / 2)).any()
    rel_mag_cond = (
        (error_hist / jnp.abs(mag)) > (epsilon_rel / jnp.sqrt(sample_num / K))
    ).any()  # Larger K makes the criterion stricter.

    # rel_mag_cond=(jnp.nansum(error_hist)>epsilon_rel*mag)[0]
    # relmag_diff_cond=(jnp.abs((mag-maglast)/maglast)>1/2*epsilon_rel)[0]
    # mag_diff_cond=(jnp.abs(mag-maglast)>1/2*epsilon)[0]

    # Switchable criteria: absolute vs relative error. If you change this,
    # also adjust the add‑points policy in `while_body_fun` accordingly.
    # `outloop` counts iterations where added points had ambiguous parity/roots.
    # When it exceeds a small threshold, the loop terminates.

    loop = (
        rel_mag_cond
        & (mini_interval > 1e-14)
        & (outloop <= 2)
        & abs_mag_cond
        & (mag_no_diff_num < 4)
        & (sample_num < max_array_length - 2)
    )
    # Alternative variants kept for experimentation/debugging:
    # jax.debug.print('{}', mag)
    # jax.debug.breakpoint()
    return loop


def while_body_fun(carry):
    """
    Body of the adaptive loop: add samples where error is high.

    This function selects intervals with the largest estimated errors, inserts
    additional sampling angles, solves for the new images, and updates both the
    error histogram and magnification. It respects the current maximum buffer
    length; if adding would exceed capacity, it only updates the sample count.

    Parameters:
        carry (tuple): `(current_carry, last_carry)`.

    Returns:
        tuple: Updated `(current_carry, last_carry)`.
    """
    carry, carrylast = carry
    carrylast = carry
    # Add points, then recalculate error and magnification.
    (trajectory_l, rho, s, q, roots_state, mag_state) = carry
    theta = roots_state.theta
    epsilon_rel = mag_state.epsilon_rel
    error_hist = mag_state.error_hist
    mag = mag_state.mag
    sample_num = roots_state.sample_num

    max_add = 4
    max_array_length = jnp.shape(theta)[0]
    max_index_length = max_array_length // 5
    max_total_num = max_array_length // 2
    K = 2
    # Add samples to multiple intervals in one pass.

    # Absolute‑error selection mode (reference):

    # idx = jnp.where(error_hist > epsilon_rel / jnp.sqrt(sample_n),
    #                 size=int(Max_array_length/5), fill_value=0)[0]
    # add_number = jnp.ceil((error_hist[idx] / epsilon_rel * jnp.sqrt(sample_n))
    #                       ** 0.2).astype(int)  # Insert at least one point.

    # Relative‑error selection mode (active):
    # error_hist_sorted = jnp.sort(error_hist,axis=0)[::-1]
    # sort_idx = jnp.argsort(error_hist,axis=0)[::-1]

    # idx=jnp.where(error_hist_sorted/jnp.abs(mag)>epsilon_rel/jnp.sqrt(sample_n),size=int(Max_array_length),fill_value=-1)[0]
    # #print('idx', idx_2)

    # idx = sort_idx[idx].reshape(-1)
    # idx = jnp.sort(idx)
    # zerot_counts = jnp.sum(idx==0)
    # idx = jnp.roll(idx,-zerot_counts)

    idx = jnp.where(
        (error_hist / jnp.abs(mag)) > (epsilon_rel / jnp.sqrt(sample_num / K)),
        size=max_index_length,
        fill_value=-1,
    )[0]

    add_number = jnp.ceil(
        (error_hist[idx] / jnp.abs(mag) / epsilon_rel * jnp.sqrt(sample_num / K))
        ** 0.2
    ).astype(int)  # Insert at least one point (excluding duplicates).

    add_number = jnp.where((idx == -1)[:, None], 0, add_number)
    add_number = jnp.where(add_number > max_add, max_add, add_number)
    add_number = jax.lax.cond(
        add_number.sum() > max_total_num,
        lambda x: (x * (max_total_num / x.sum())).astype(int),
        lambda x: x,
        add_number,
    )
    # Debug helper to inspect allocation budget and selections:
    # jax.debug.print('add_number {}/{}  idx length {}/{}  sample_n: {}',
    #                 add_number.sum(), Max_total_num,
    #                 (idx != 0).sum(), Max_index_length, sample_n[0])

    def encode_theta_inserts(carry, k):
        (theta, idx, add_number, add_theta_encode) = carry

        theta_diff = (theta[idx[k]] - theta[idx[k] - 1]) / (add_number[k] + 1)
        add_theta = (
            jnp.arange(1, max_total_num + 1)[:, None] * theta_diff + theta[idx[k] - 1]
        )
        add_theta = jnp.where(
            (jnp.arange(max_total_num) < add_number[k])[:, None], add_theta, jnp.nan
        )
        carry2, _ = insert_body(
            (
                add_theta_encode,
                add_theta,
                jnp.where(jnp.isnan(add_theta_encode), size=1)[0],
                add_number[k][None],
            ),
            0,
        )
        add_theta_encode = carry2[0]
        return (theta, idx, add_number, add_theta_encode), k

    def update_carry(carrylast):
        carry, _ = lax.scan(
            encode_theta_inserts,
            (theta, idx, add_number, jnp.full((max_total_num, 1), jnp.nan)),
            jnp.arange(idx.shape[0]),
        )
        add_theta = carry[-1]
        jax_repeat = jax.jit(
            jnp.repeat, static_argnames=["axis", "total_repeat_length"]
        )
        idx_all = jax_repeat(idx, add_number, total_repeat_length=max_total_num)
        idx_all = jnp.where(
            jnp.arange(idx_all.shape[0]) < add_number.sum(), idx_all, -1
        )
        # Compute new source positions and solve for additional images.
        add_zeta_l = get_zeta_l(rho, trajectory_l, add_theta)
        roots_state_new, buried_error, add_outloop = add_points(
            idx_all, add_zeta_l, add_theta, roots_state, s, 1 / (1 + q), q / (1 + q)
        )
        buried_error = buried_error / jnp.pi / rho**2
        mag_state_new = update_mag(
            roots_state_new, mag_state, rho, q, s, buried_error, add_outloop
        )
        carry = (trajectory_l, rho, s, q, roots_state_new, mag_state_new)
        return carry

    def no_update_carry(carrylast):
        trajectory_l, rho, s, q, roots_state, mag_state = carrylast
        roots_state_new = roots_state._replace(sample_num=(sample_num + add_number.sum()))
        return (trajectory_l, rho, s, q, roots_state_new, mag_state)

    carry = lax.cond(
        (sample_num + add_number.sum()) < (max_array_length - 2),
        update_carry,
        no_update_carry,
        carrylast,
    )
    return (carry, carrylast)


def update_mag(roots_state, mag_state_last, rho, q, s, buried_error, add_outloop):
    """
    Recompute magnification and error histogram after adding samples.

    Parameters:
        roots_state (Iterative_State): Updated roots/sampling state.
        mag_state_last (Error_State): Previous magnification/error state.
        rho (float): Source radius.
        q (float): Mass ratio (m2/m1).
        s (float): Binary separation.
        buried_error (jax.Array | float): Error contribution from buried images.
        add_outloop (int): Count of ambiguous insertions in this iteration.

    Returns:
        Error_State: Updated error state with new magnification and histogram.
    """
    maglast = mag_state_last.mag
    epsilon = mag_state_last.epsilon
    epsilon_rel = mag_state_last.epsilon_rel

    mag = (
        1
        / 2
        * jnp.nansum(
            jnp.nansum(
                (roots_state.roots.imag[0:-1] + roots_state.roots.imag[1:])
                * (roots_state.roots.real[0:-1] - roots_state.roots.real[1:])
                * roots_state.parity[0:-1],
                axis=0,
            )
        )
    )
    error_hist, magc, parab = error_sum(roots_state, rho, q, s)
    mag = (mag + magc + parab) / (jnp.pi * rho**2)
    add_mag_no_diff_num = (jnp.abs(mag - maglast) < 1 / 2 * epsilon).sum()

    # Track consecutive iterations with negligible change. If this counter grows
    # too large, `cond_fun` will stop the loop.
    mag_no_diff = jnp.where(add_mag_no_diff_num > 0, mag_state_last.mag_no_diff + 1, 0)

    error_hist += buried_error
    mag_state = Error_State(
        mag,
        mag_no_diff,
        add_outloop + mag_state_last.outloop,
        error_hist,
        epsilon,
        epsilon_rel,
    )
    return mag_state
