"""
Error estimation for adaptive contour integration.

Provides the error terms used to control sampling density in the adaptive
contour integration loop. Includes parabolic correction terms for ordinary
adjacent intervals and special handling near critical points where images are
created or destroyed.

Main functions:
- `error_sum(...)`: Per‑interval error histogram, critical‑event correction,
  and total parabolic correction (pre‑normalization).
- `error_ordinary(...)`: Error and parabolic correction between adjacent
  samples.
- `error_critical(...)`: Error contribution near creation/destruction events
  and associated parabolic correction.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..core.lens_equation import basic_partial, dot_product


def error_ordinary(deXProde2X, de_z, delta_theta, z, parity, de_deXPro_de2X):
    """
    Estimate error and parabolic correction on ordinary segments.

    Parameters:
        deXProde2X (jax.Array): Discrete approximation of x'·x'' along the contour.
        de_z (jax.Array): Derivative of image position with respect to source angle.
        delta_theta (jax.Array): Angle differences `theta[i+1] - theta[i]`.
        z (jax.Array): Image positions for each angle sample.
        parity (jax.Array): Parity (+1 or −1) of each image.
        de_deXPro_de2X (jax.Array): Derivative of `x'·x''` w.r.t. theta
            (only nonzero near caustic crossings).

    Returns:
        tuple:
            - `e_tot` (jax.Array): Estimated error per interval (shape [N-1, ...]).
            - `dAp` (jax.Array): Parabolic correction contribution per interval.
    """
    dAp_1 = 1 / 24 * (deXProde2X[0:-1] + deXProde2X[1:]) * delta_theta
    delta_theta_wave = jnp.abs(z[0:-1] - z[1:]) ** 2 / jnp.abs(
        dot_product(de_z[0:-1], de_z[1:])
    )

    dAp_v1 = (
        dAp_1 * delta_theta**2 * parity[0:-1]
    )  # old version of parabolic correction term
    dAp_v2 = (
        1
        / 12
        * (
            (z.real[1:] - z.real[0:-1]) * (de_z.imag[1:] - de_z.imag[0:-1])
            - (z.imag[1:] - z.imag[0:-1]) * (de_z.real[1:] - de_z.real[0:-1])
        )
        * delta_theta
        * parity[0:-1]
    )  # new version of parabolic correction term
    dAp = 0.5 * (dAp_v1 + dAp_v2)

    # e1=jnp.abs(1/48*jnp.abs(jnp.abs(deXProde2X[0:-1]-jnp.abs(deXProde2X[1:])))*delta_theta**3) # old version
    e1 = jnp.abs(dAp_v1 - dAp_v2) * 0.5
    e2 = 3 / 2 * jnp.abs(dAp_1 * (delta_theta_wave - delta_theta**2))
    e3 = 1 / 10 * jnp.abs(dAp) * delta_theta**2

    de_dAp = (
        1
        / 24
        * (de_deXPro_de2X[0:-1] - de_deXPro_de2X[1:])
        * delta_theta**3
        * parity[0:-1]
    )  # gradient of the parabolic correction term
    e4 = 1 / 10 * jnp.abs(de_dAp)  # gradient error of the parabolic correction
    e_tot = e1 + e2 + e3 + e4
    return e_tot, dAp


def error_critical(pos_idx, neg_idx, i, create, parity, deXProde2X, z, de_z):
    """
    Error estimation and parabolic correction near a critical point.

    Parameters:
        pos_idx (int | jax.Array): Column index of the positive‑parity branch.
        neg_idx (int | jax.Array): Column index of the negative‑parity branch.
        i (int | jax.Array): Row index of the critical sample.
        create (int | jax.Array): +1 for creation, −1 for destruction.
        parity (jax.Array): Parity matrix for all samples/images.
        deXProde2X (jax.Array): Discrete `x'·x''` term.
        z (jax.Array): Image positions.
        de_z (jax.Array): Derivative of image position w.r.t. theta.

    Returns:
        tuple:
            - `ce_tot` (jax.Array): Error contribution at the critical point.
            - `dAcP` (jax.Array): Parabolic correction at the critical point.
            - `magc` (jax.Array): Trapezoidal contribution to magnification at the
              critical point, before normalization.
    """
    # pos_idx=jnp.where(((Is_create[i]==create)|(Is_create[i]==10))&(parity[i]==-1*create),size=1)[0]
    # neg_idx=jnp.where(((Is_create[i]==create)|(Is_create[i]==10))&(parity[i]==1*create),size=1)[0]
    z_pos = z[i, pos_idx]
    z_neg = z[i, neg_idx]

    theta_wave = jnp.abs(z_pos - z_neg) / jnp.sqrt(
        jnp.abs(dot_product(de_z[i, pos_idx], de_z[i, neg_idx]))
    )

    dAcP_v1 = (
        parity[i, pos_idx]
        * 1
        / 24
        * (deXProde2X[i, pos_idx] - deXProde2X[i, neg_idx])
        * theta_wave**3
    )  # old version of the parabolic correction term at the critical point
    dAcP_v2 = (
        -1
        / 12
        * (
            (z[i, neg_idx].real - z[i, pos_idx].real)
            * (de_z[i, pos_idx].imag + de_z[i, neg_idx].imag)
            - (z[i, neg_idx].imag - z[i, pos_idx].imag)
            * (de_z[i, pos_idx].real + de_z[i, neg_idx].real)
        )
        * theta_wave
        * parity[i, pos_idx]
    )  # new version of the parabolic correction term at the critical point
    dAcP = 0.5 * (dAcP_v1 + dAcP_v2)

    # ce1=1/48*jnp.abs(deXProde2X[i,pos_idx]+deXProde2X[i,neg_idx])*theta_wave**3 # old version
    ce1 = 0.5 * jnp.abs(dAcP - dAcP_v2)  # new version of error term 1
    ce2 = (
        3
        / 2
        * jnp.abs(
            dot_product(z_pos - z_neg, de_z[i, pos_idx] - de_z[i, neg_idx])
            - create
            * 2
            * jnp.abs(z_pos - z_neg)
            * jnp.sqrt(jnp.abs(dot_product(de_z[i, pos_idx], de_z[i, neg_idx])))
        )
        * theta_wave
    )
    ce3 = 1 / 10 * jnp.abs(dAcP) * theta_wave**2
    ce_tot = ce1 + ce2 + ce3

    return (
        ce_tot,
        jnp.sum(dAcP),
        1
        / 2
        * (z[i, pos_idx].imag + z[i, neg_idx].imag)
        * (z[i, pos_idx].real - z[i, neg_idx].real),
    )


# Backward‑compatible alias (original misspelling kept for safety)
error_critial = error_critical


def error_sum(roots_state, rho, q, s, mask=None):
    """
    Compute error histogram and parabolic corrections for a full contour.

    Parameters:
        roots_state (Iterative_State): Current state containing theta, roots,
            parity, and creation/destruction markers.
        rho (float): Source radius.
        q (float): Mass ratio (m2/m1).
        s (float): Binary separation (Einstein units).
        mask (jax.Array | None): Optional boolean mask for valid samples/images.

    Returns:
        tuple:
            - `error_hist` (jax.Array): Per‑row error estimates for adaptive sampling.
            - `mag_c` (float): Critical‑point magnification contribution (pre‑norm).
            - `parab` (float): Sum of parabolic correction terms (pre‑norm).
    """
    Is_create = roots_state.Is_create
    z = roots_state.roots
    parity = roots_state.parity
    theta = roots_state.theta
    if mask is None:
        mask = ~jnp.isnan(z)
    caustic_crossing = (Is_create[3, :] != 0).any()
    deXProde2X, de_z, de_deXPro_de2X = basic_partial(
        z, theta, rho, q, s, caustic_crossing
    )
    deXProde2X = jnp.where(mask, deXProde2X, 0.0)
    de_z = jnp.where(mask, de_z, 0.0)
    de_deXPro_de2X = jnp.where(mask, de_deXPro_de2X, 0.0)

    error_hist = jnp.zeros_like(theta)
    delta_theta = jnp.diff(theta, axis=0)

    mag = jnp.array([0.0])
    e_ord, parab = error_ordinary(
        deXProde2X, de_z, delta_theta, z, parity, de_deXPro_de2X
    )

    diff_mask = mask[1:] & mask[:-1]
    e_ord = jnp.where(diff_mask, e_ord, 0.0)
    parab = jnp.where(diff_mask, parab, 0.0)
    e_ord = jnp.sum(e_ord, axis=1)
    parab = jnp.sum(parab)

    error_hist = error_hist.at[1:].set(e_ord[:, None])

    de_z = jnp.where(
        mask, de_z, 10.0
    )  # avoid the nan value in the theta_wave calculation

    def no_create_true_fun(carry):
        ## if there is image create or destroy, we need to calculate the error

        mag, parab, error_hist = carry
        critical_row_idx, critical_pos_idx, critical_neg_idx, create_array = Is_create
        total_num = (create_array != 0).sum()
        result_row_idx = critical_row_idx
        critical_error, dApc, magc = jax.vmap(
            error_critical, in_axes=(0, 0, 0, 0, None, None, None, None)
        )(
            critical_pos_idx,
            critical_neg_idx,
            result_row_idx,
            create_array,
            parity,
            deXProde2X,
            z,
            de_z,
        )
        critical_error = jnp.where(
            jnp.arange(len(critical_error)) < total_num, critical_error, 0.0
        )
        magc = jnp.where(jnp.arange(len(magc)) < total_num, magc, 0.0)
        dApc = jnp.where(jnp.arange(len(dApc)) < total_num, dApc, 0.0)
        error_hist = error_hist.at[(result_row_idx - (create_array - 1) // 2), 0].add(
            critical_error
        )
        mag += jnp.sum(magc)
        parab += jnp.sum(dApc)

        return (mag, parab, error_hist)

    carry = jax.lax.cond(
        caustic_crossing,
        no_create_true_fun,
        lambda x: x,
        (0.0, 0.0, jnp.zeros_like(theta)),
    )
    mag_c, parab_c, error_hist_c = carry
    mag += mag_c
    parab += parab_c
    error_hist += error_hist_c

    return error_hist / (np.pi * rho**2), mag, parab
