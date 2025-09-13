"""
Adaptive contour-integration helpers for microlensing root management.

Provides utilities to insert new sampling angles, solve and sort polynomial
roots across rows, correct parity inconsistencies, detect creation/destruction
events at caustics, and update working arrays while keeping JAX-friendly
static shapes.

Key functions:
- add_points(...): insert samples and reorder roots consistently
- get_sorted_roots(...): stable matching of roots across rows
- get_real_roots(...): solve, filter physical images, fix parity
- find_create_points(...): locate caustic creation/destruction events
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from ..core.lens_equation import get_parity, get_parity_error, get_poly_coff, verify
from ..numerics.assignment import find_nearest
from ..numerics.polynomial import get_roots
from ..core.utils import (
    custom_delete,
    custom_insert,
)
from ..core.state import (
    Iterative_State,
    MAX_CAUSTIC_INTERSECT_NUM,
)


 # jax config is set in package __init__


def add_points(add_idx, add_zeta, add_theta, roots_state, s, m1, m2):
    """
    Insert new sampling rows and integrate them into the working state.

    Parameters:
        add_idx (jax.Array): Target insert indices per new row (−1 to skip).
        add_zeta (jax.Array): Source positions at new angles.
        add_theta (jax.Array): Angles to insert.
        roots_state (Iterative_State): Current roots/sampling state.
        s (float): Binary separation.
        m1 (float): Primary mass fraction.
        m2 (float): Secondary mass fraction.

    Returns:
        tuple:
            - `Iterative_State`: Updated state with new rows inserted and roots
              sorted across all rows.
            - `buried_error` (jax.Array): Error accumulator for buried images.
            - `outloop` (int): Count of deletions due to inconsistent parity.
    """
    sample_n, theta, roots, parity, ghost_roots_dis, sort_flag, Is_create = roots_state
    add_coeff = get_poly_coff(add_zeta, s, m2)
    (
        add_roots,
        add_parity,
        add_ghost_roots,
        outloop,
        add_coeff,
        add_zeta,
        add_theta,
        add_idx,
    ) = get_real_roots(add_coeff, add_zeta, add_theta, s, m1, m2, add_idx)

    sample_n += (add_idx != -1).sum()

    ## insert the new samplings
    insert_fun = lambda x, y: custom_insert(x, add_idx, y)
    original_list = [theta, roots, parity, ghost_roots_dis, sort_flag]
    add_list = [
        add_theta,
        add_roots,
        add_parity,
        add_ghost_roots,
        jnp.full([add_roots.shape[0], 1], False),
    ]
    theta, unsorted_roots, unsorted_parity, ghost_roots_dis, sort_flag = jax.tree.map(
        insert_fun, original_list, add_list
    )

    buried_error = get_buried_error(ghost_roots_dis, sample_n)

    # Reorder all roots and parity for consistent identity across rows
    indices_update, sort_flag = get_sorted_roots(
        unsorted_roots, unsorted_parity, sort_flag, add_theta.shape[0]
    )

    roots = unsorted_roots[jnp.arange(unsorted_roots.shape[0])[:, None], indices_update]
    parity = unsorted_parity[
        jnp.arange(unsorted_parity.shape[0])[:, None], indices_update
    ]

    Is_create = find_create_points(roots, parity, sample_n)
    return (
        Iterative_State(
            sample_n, theta, roots, parity, ghost_roots_dis, sort_flag, Is_create
        ),
        buried_error,
        outloop,
    )


def get_buried_error(ghost_roots_dis, sample_n):
    """
    Estimate error to guard against buried images near cusps.

    Based on Bozza (2010); we use a conservative criterion to reduce the risk
    of missing solutions when an image becomes temporarily hidden.

    Parameters:
        ghost_roots_dis (jax.Array): Distance metric between candidate ghost
            roots used to flag potential buried images.
        sample_n (int): Current number of valid sampling rows.

    Returns:
        jax.Array: Per‑row buried‑image error contribution.
    """
    n_ite = ghost_roots_dis.shape[0]
    error_buried = jnp.zeros((n_ite, 1))

    idx_j = jnp.arange(1, n_ite)
    idx_i = jnp.roll(idx_j, shift=1)
    idx_k = jnp.roll(idx_j, shift=-1)

    idx_k = jnp.where(
        idx_k == sample_n, 1, idx_k
    )  # because the last point is the same as the first point 0=2pi
    idx_i = jnp.where(idx_i == n_ite - 1, sample_n - 1, idx_i)

    Ghost_i = ghost_roots_dis[idx_i]
    Ghost_j = ghost_roots_dis[idx_j]
    Ghost_k = ghost_roots_dis[idx_k]
    KInCaustic = jnp.isnan(Ghost_k)
    IInCaustic = jnp.isnan(Ghost_i)

    # only add points in one side same as the VBBL
    add_item = jnp.where(
        (
            # ( (Ghost_i>Ghost_j)&(Ghost_k>Ghost_j)
            # ((((Ghost_i-Ghost_j)/(theta[idx_i]-theta[idx_j])*(theta[idx_j]-theta[idx_k]))>Ghost_j)
            (
                (Ghost_i > 2 * Ghost_j)
                | (
                    (Ghost_i > 1.5 * Ghost_j) & (Ghost_k > Ghost_j)
                )  # supplementary condition to avoid the burried images, 1.5 is a tunable parameter
            )
            & (~KInCaustic)
        ),
        (Ghost_i - Ghost_j) ** 2,
        0,
    )

    error_buried = error_buried.at[idx_k].add(add_item)

    add_item = jnp.where(
        (
            # ( (Ghost_i>Ghost_j)&(Ghost_k>Ghost_j)
            # ((Ghost_j<((Ghost_k-Ghost_j)/(theta[idx_k]-theta[idx_j])*(theta[idx_j]-theta[idx_i])))
            (
                (2 * Ghost_j < Ghost_k)
                | ((1.5 * Ghost_j < Ghost_k) & (Ghost_i > Ghost_j))  # same as above
            )
            & (~IInCaustic)
        ),
        (Ghost_k - Ghost_j) ** 2,
        0,
    )
    error_buried = error_buried.at[idx_j].add(add_item)

    return error_buried


@partial(jax.jit, static_argnames=("max_unsorted_num",))
def get_sorted_roots(roots, parity, sort_flag, max_unsorted_num):
    """
    Stably match and reorder roots and parity across rows.

    The cost function penalizes geometric distance and parity disagreement to
    achieve consistent identity assignment for images across adjacent rows.

    Parameters:
        roots (jax.Array): Complex roots per row.
        parity (jax.Array): Parity per root.
        sort_flag (jax.Array | bool): Flags for rows that already have stable order.
        max_unsorted_num (int): Upper bound on number of rows to resort.

    Returns:
        tuple:
            - `indices_update` (jax.Array): Index map used for reordering.
            - `sort_flag` (jax.Array): Updated flags (all True after resort).
    """

    indices = jnp.tile(jnp.arange(roots.shape[1]), (roots.shape[0], 1))

    def sort_body1(
        indices, k
    ):  # sort the roots and parity for adjacent points olde-new and new-new pairs
        def False_fun_sort1(indices):
            sort_indices = find_nearest(
                roots[k - 1, indices[k - 1]],
                parity[k - 1, indices[k - 1]],
                roots[k, :],
                parity[k, :],
            )
            indices = indices.at[k].set(sort_indices)
            return indices

        indices = lax.cond(k == -1, lambda x: x, False_fun_sort1, indices)
        return indices, k

    false_i = jnp.where(~sort_flag, size=max_unsorted_num, fill_value=-1)[0]
    indices_update, _ = lax.scan(sort_body1, indices, false_i)

    def sort_body2(indices_temp, i):  # sort the roots and parity for new-old pairs
        def False_fun(indices_temp):
            sort_indices = find_nearest(
                roots[i, indices_temp[i]],
                parity[i, indices_temp[i]],
                roots[i + 1, indices_temp[i + 1]],
                parity[i + 1, indices_temp[i + 1]],
            )

            cond = jnp.arange(roots.shape[0])[:, None] < i + 1
            indices_temp = jnp.where(cond, indices_temp, indices_temp[:, sort_indices])

            return indices_temp

        indices_temp = lax.cond(i == -2, lambda x: x, False_fun, indices_temp)

        return indices_temp, i

    resort_i = jnp.where(
        (~sort_flag[0:-1]) & (sort_flag[1:]), size=max_unsorted_num, fill_value=-2
    )[0]
    indices_update2, _ = lax.scan(sort_body2, indices_update, resort_i)

    roots = roots[jnp.arange(roots.shape[0])[:, None], indices_update2]
    parity = parity[jnp.arange(parity.shape[0])[:, None], indices_update2]

    sort_flag = sort_flag.at[:].set(True)
    return indices_update2, sort_flag


def get_real_roots(coeff, zeta_l, theta, s, m1, m2, add_idx):
    """
    Solve, validate, and filter physical roots for inserted rows.

    Steps:
        1. Solve polynomial roots for each row.
        2. Compute parity of all roots.
        3. Verify roots by direct substitution into the lens equation.
        4. Select physical images using a relative error criterion (VBBL‑like).
        5. Fix wrong parity when detected.
        6. Delete remaining inconsistent roots/parity.

    Parameters:
        coeff (jax.Array): Polynomial coefficients per row.
        zeta_l (jax.Array): Source positions per row.
        theta (jax.Array): Angles per row.
        s (float): Binary separation.
        m1 (float): Primary mass fraction.
        m2 (float): Secondary mass fraction.
        add_idx (jax.Array): Insert indices used for newly added rows.

    Returns:
        tuple: Ordered as in the caller expectations:
            (`real_roots`, `real_parity`, `ghost_roots_dis`, `outloop`,
             `coeff`, `zeta_l`, `theta`, `add_idx`).
    """

    n_ite = zeta_l.shape[0]
    sample_n = (~jnp.isnan(zeta_l)).any(axis=1).sum()
    mask = jnp.arange(n_ite) < sample_n
    roots = get_roots(n_ite, jnp.where(mask[:, None], coeff, 0.0))
    roots = jnp.where(mask[:, None], roots, jnp.nan)
    parity = get_parity(roots, s, m1, m2)
    error = verify(zeta_l, roots, s, m1, m2)

    iterator = jnp.arange(n_ite)  # Criterion to select the roots (VBBL-like)
    dlmin = 1.0e-4
    dlmax = 1.0e-3
    sort_idx = jnp.argsort(error, axis=1)
    third_error = error[iterator, sort_idx[:, 2]]
    forth_error = error[iterator, sort_idx[:, 3]]
    three_roots_cond = (forth_error * dlmin) > (
        third_error + 1e-12
    )  # three roots criterion
    bad_roots_cond = (~three_roots_cond) & (
        (forth_error * dlmax) > (third_error + 1e-12)
    )  # bad roots criterion
    cond = jnp.zeros_like(roots, dtype=bool)
    full_value = jnp.where(three_roots_cond, True, False)
    cond = cond.at[iterator[:, None], sort_idx[:, 3:]].set(full_value[:, None])

    ghost_roots_dis = jnp.abs(
        roots[iterator, sort_idx[:, 3]] - roots[iterator, sort_idx[:, 4]]
    )
    ghost_roots_dis = jnp.where(three_roots_cond, ghost_roots_dis, jnp.nan)[:, None]

    # find the wrong parity and fix it
    nan_num = cond.sum(axis=1)
    real_roots = jnp.where(cond, jnp.nan + jnp.nan * 1j, roots)
    real_parity = jnp.where(cond, jnp.nan, parity)
    parity_sum = jnp.nansum(real_parity, axis=1)
    idx_parity_wrong = jnp.where((parity_sum != -1) & mask, size=n_ite, fill_value=-1)[
        0
    ]  # indices for rows with inconsistent parity
    real_parity = lax.cond(
        (idx_parity_wrong != -1).any(),
        update_parity,
        lambda x: x[-1],
        (
            zeta_l,
            real_roots,
            nan_num,
            sample_n,
            idx_parity_wrong,
            cond,
            s,
            m1,
            m2,
            real_parity,
        ),
    )
    parity_sum = jnp.nansum(real_parity, axis=1)
    bad_parities_cond = parity_sum != -1
    outloop = 0

    # delete the remaining wrong roots/parity
    carry = lax.cond(
        (bad_parities_cond & bad_roots_cond & mask).any(),
        theta_remove_fun,
        lambda x: x,
        (
            sample_n,
            theta,
            real_parity,
            real_roots,
            ghost_roots_dis,
            outloop,
            parity_sum,
            mask,
            add_idx,
        ),
    )
    (
        sample_n,
        theta,
        real_parity,
        real_roots,
        ghost_roots_dis,
        outloop,
        parity_sum,
        _,
        add_idx,
    ) = carry

    return (
        real_roots,
        real_parity,
        ghost_roots_dis,
        outloop,
        coeff,
        zeta_l,
        theta,
        add_idx,
    )


def update_parity(carry):
    """
    Refine parity assignment for rows flagged as inconsistent.

    Parameters:
        carry (tuple): Packed state for parity update; contains `zeta_l`,
            `real_roots`, `nan_num`, `sample_n`, `idx_parity_wrong`, `cond`,
            `s`, `m1`, `m2`, `real_parity`.

    Returns:
        jax.Array: Updated `real_parity` array.
    """
    (
        zeta_l,
        real_roots,
        nan_num,
        sample_n,
        idx_parity_wrong,
        cond,
        s,
        m1,
        m2,
        real_parity,
    ) = carry

    def loop_parity_body(carry, i):
        zeta_l, real_roots, real_parity, nan_num, sample_n, cond, s, m1, m2 = carry
        temp = real_roots[i]
        parity_process_fun = lambda x: lax.cond(
            (nan_num[i] == 0) & (i < sample_n),
            parity_5_roots_fun,
            parity_3_roots_fun,
            x,
        )
        real_parity = lax.cond(
            i < sample_n,
            parity_process_fun,
            lambda x: x[2],
            (temp, zeta_l, real_parity, i, cond, nan_num, s, m1, m2),
        )
        # Alternative parity correction branches kept for reference
        return (zeta_l, real_roots, real_parity, nan_num, sample_n, cond, s, m1, m2), i

    carry, _ = lax.scan(
        loop_parity_body,
        (zeta_l, real_roots, real_parity, nan_num, sample_n, cond, s, m1, m2),
        idx_parity_wrong,
    )
    zeta_l, real_roots, real_parity, nan_num, sample_n, cond, s, m1, m2 = carry
    real_parity = real_parity.at[-1].set(jnp.nan)
    return real_parity


def parity_5_roots_fun(carry):
    """
    Determine parity for ambiguous 5‑image configurations.

    Strategy: select principal and fifth images (+1 and −1 parity respectively),
    then sort the remaining three by x and assign (+1) to the middle image,
    (−1) to the two side images.
    """
    temp, zeta_l, real_parity, i, cond, nan_num, s, m1, m2 = carry
    prin_idx = jnp.where(
        jnp.sign(temp.imag) == jnp.sign(zeta_l.imag[i]), size=1, fill_value=0
    )[0]
    prin_root = temp[prin_idx][jnp.newaxis][0]
    prin_root = jnp.concatenate(
        [prin_root, temp[jnp.argmax(get_parity_error(temp, s, m1, m2))][jnp.newaxis]]
    )
    other = jnp.setdiff1d(temp, prin_root, size=3)
    x_sort = jnp.argsort(other.real)
    real_parity = real_parity.at[
        i,
        jnp.where((temp == other[x_sort[0]]) | (temp == other[x_sort[-1]]), size=2)[0],
    ].set(-1)
    real_parity = real_parity.at[
        i, jnp.where((temp == other[x_sort[1]]), size=1)[0]
    ].set(1)
    return real_parity


def parity_3_roots_fun(carry):
    """
    Determine parity for ambiguous 3‑image configurations.

    Strategy: set principal image to positive parity; assign negative parity to
    the other two. This is unreliable very near the x‑axis where the principal
    criterion breaks down.
    """
    temp, zeta_l, real_parity, i, cond, nan_num, s, m1, m2 = carry

    def parity_true_fun(carry):
        real_parity = carry
        real_parity = real_parity.at[i, jnp.where(~cond[i], size=3)].set(-1)
        real_parity = real_parity.at[
            i, jnp.where(jnp.sign(temp.imag) == jnp.sign(zeta_l.imag[i]), size=1)[0]
        ].set(1)
        return real_parity

    real_parity = lax.cond(
        (nan_num[i] != 0) & ((jnp.abs(zeta_l.imag[i]) > 1e-5)[0]),
        parity_true_fun,
        lambda x: x,
        real_parity,
    )
    return real_parity


def find_create_points(roots, parity, sample_n):
    """
    Locate creation/destruction events and encode their indices.

    The returned structure is used to choose the correct trapezoid orientation
    and to place error contributions at the proper rows for adaptive control.

    Parameters:
        roots (jax.Array): Image positions per row.
        parity (jax.Array): Parity assignments per row.
        sample_n (int): Effective number of valid rows.

    Returns:
        jax.Array: Encoded indices and flags with shape (4, MAX_CAUSTIC_INTERSECT_NUM).
    """
    cond = jnp.isnan(roots)
    Num_change_cond = jnp.diff(
        cond, axis=0
    )  # the roots at i is nan but the roots at i+1 is not nan/ the roots at i is not nan but the roots at i+1 is nan
    idx_x, idx_y = jnp.where(
        Num_change_cond & (jnp.arange(roots.shape[0] - 1) < (sample_n - 1))[:, None],
        size=MAX_CAUSTIC_INTERSECT_NUM * 2,
        fill_value=-2,
    )  ## the index i can't be the last index
    shift = jnp.where(cond[idx_x, idx_y], 1, 0)
    idx_x_create = idx_x + shift
    Create_Destroy = jnp.where(cond[idx_x, idx_y], 1, -1)
    Create_Destroy = jnp.where(idx_x < 0, 0, Create_Destroy)
    critical_idx = idx_x_create[0::2]
    critical_idy1 = idx_y[0::2]
    critical_idy2 = idx_y[1::2]

    critical_idx = jnp.where(critical_idx < 0, 0, critical_idx)
    critical_idy1 = jnp.where(critical_idy1 < 0, 0, critical_idy1)
    critical_idy2 = jnp.where(critical_idy2 < 0, 0, critical_idy2)

    critical_pos_idy = jnp.where(
        Create_Destroy[0::2] == -1 * parity[critical_idx, critical_idy1],
        critical_idy1,
        critical_idy2,
    )
    critical_neg_idy = jnp.where(
        Create_Destroy[0::2] == 1 * parity[critical_idx, critical_idy1],
        critical_idy1,
        critical_idy2,
    )

    return jnp.stack(
        [critical_idx, critical_pos_idy, critical_neg_idy, Create_Destroy[0::2]], axis=0
    )


def theta_remove_fun(carry):
    """
    Remove rows with parity still inconsistent after correction.

    Parameters:
        carry (tuple): Packed state containing sampling arrays and counters.

    Returns:
        tuple: Updated packed state after deletions.
    """
    (
        sample_n,
        theta,
        real_parity,
        real_roots,
        ghost_roots_dis,
        outloop,
        parity_sum,
        mask,
        add_idx,
    ) = carry
    n_ite = theta.shape[0]
    cond = (parity_sum != -1) & mask
    delidx = jnp.where(cond, size=n_ite, fill_value=10000)[0]

    sample_n -= cond.sum()

    delete_tree = [theta, real_parity, real_roots, ghost_roots_dis, add_idx[:, None]]
    theta, real_parity, real_roots, ghost_roots_dis, add_idx = jax.tree.map(
        lambda x: custom_delete(x, delidx), delete_tree
    )
    add_idx = add_idx[:, 0]

    outloop += cond.sum()
    # if the parity is still wrong, then delete the point
    return (
        sample_n,
        theta,
        real_parity,
        real_roots,
        ghost_roots_dis,
        outloop,
        parity_sum,
        mask,
        add_idx,
    )
