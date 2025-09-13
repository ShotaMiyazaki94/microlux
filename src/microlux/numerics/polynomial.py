"""
Polynomial root solvers and utilities (JAX) for the binary lens equation.

Implements the Aberth–Ehrlich method with custom_root for differentiability,
plus helpers to generate initial guesses and batched solvers. All functions are
JIT‑friendly and use complex dtype with x64 enabled.

Also provides a small, JAX‑friendly companion‑matrix based `roots(...)`
implementation so tests can compare AE against a second, independent solver
without relying on NumPy inside `jit`.
"""

import jax


# jax config is set in package __init__
from functools import partial

from jax import lax, numpy as jnp


def loop_body(roots0, coeff):
    """
    Single AE update step used inside `lax.scan` over coefficient rows.
    """
    def False_fun(carry):
        coeff, roots0 = carry
        roots_new = Aberth_Ehrlich(coeff, roots0)
        return roots_new

    roots_new = lax.cond((coeff == 0).all(), lambda x: x[1], False_fun, (coeff, roots0))
    return roots_new, roots_new


@partial(jax.jit, static_argnums=0)
def get_roots(sample_n, coeff):
    """
    Solve roots row‑wise using AE iteration, reusing previous guesses.

    Parameters:
        sample_n (int): Unused static arg to stabilize JIT cache keys.
        coeff (jax.Array): Polynomial coefficients per row.

    Returns:
        jax.Array: Roots per row, matching the shape of input rows.
    """
    roots0 = AE_roots0(coeff[0])
    _, roots = lax.scan(loop_body, roots0, coeff)
    return roots


@partial(jax.jit, static_argnums=0)
def get_roots_vmap(sample_n, coeff):
    """
    Solve roots row‑wise using `vmap` when all rows are valid.

    Useful when there are no sentinel rows with zero coefficients.
    """
    roots_solver = lambda x: Aberth_Ehrlich(x, AE_roots0(x))
    roots = jax.vmap(roots_solver, in_axes=(0))(coeff)
    return roots


@jax.jit
def _companion_roots_1d(p: jnp.ndarray, strip_zeros: bool = False) -> jnp.ndarray:
    """
    Compute polynomial roots via the companion matrix (single 1D poly).

    Parameters:
        p (jax.Array): Coefficients `[a0, a1, ..., an]` (descending powers).
        strip_zeros (bool): If True, strip leading zeros (default False).

    Returns:
        jax.Array: Complex roots of the polynomial.
    """
    p = jnp.asarray(p)
    # Optionally strip leading zeros to handle degenerate inputs gracefully.
    if strip_zeros:
        # Find first non-zero coefficient; if all zeros, return empty.
        nz = jnp.argmax(p != 0)
        all_zero = (p != 0).any() == False
        p = jnp.where(all_zero, p, p[nz:])

    n = p.shape[0] - 1
    # If degree < 1, no roots to compute
    def empty_roots(_):
        return jnp.empty((0,), dtype=jnp.complex128)

    def compute_roots(p):
        dtype = jnp.result_type(p) if jnp.iscomplexobj(p) else jnp.complex128
        # Build companion matrix with complex dtype
        C = jnp.zeros((n, n), dtype=dtype)
        # Set subdiagonal ones
        C = C.at[jnp.arange(1, n), jnp.arange(0, n - 1)].set(1)
        # Top row from normalized coefficients
        C = C.at[0, :].set(-p[1:] / p[0])
        # Eigenvalues are the roots
        vals = jnp.linalg.eigvals(C)
        return vals

    return lax.cond(n < 1, empty_roots, compute_roots, p)


def roots(p: jnp.ndarray, strip_zeros: bool = False) -> jnp.ndarray:
    """
    JAX‑friendly polynomial roots for 1D or stacked coefficient arrays.

    Parameters:
        p (jax.Array): Coefficients (descending powers). Shape `(deg+1,)` or `(N, deg+1)`.
        strip_zeros (bool): If True, strip leading zeros before solving.

    Returns:
        jax.Array: Roots with shape `(deg,)` or `(N, deg)` respectively.
    """
    p = jnp.asarray(p)
    if p.ndim == 1:
        return _companion_roots_1d(p, strip_zeros)
    elif p.ndim == 2:
        solver = lambda row: _companion_roots_1d(row, strip_zeros)
        return jax.vmap(solver, in_axes=0)(p)
    else:
        raise ValueError("p must be 1D or 2D array of coefficients")

@jax.jit
def AE_roots0(coeff: jnp.ndarray) -> jnp.ndarray:
    """
    Generate initial guesses for AE using annulus sampling.

    Parameters:
        coeff (jax.Array): Polynomial coefficients (single row).

    Returns:
        jax.Array: Complex initial guesses for all roots.
    """

    def UV(coeff):
        U = 1 + 1 / jnp.abs(coeff[0]) * jnp.max(jnp.abs(coeff[:-1]))
        V = jnp.abs(coeff[-1]) / (jnp.abs(coeff[-1]) + jnp.max(jnp.abs(coeff[:-1])))
        return U, V

    def Roots0(coeff):
        U, V = UV(coeff)
        r = jax.random.uniform(
            jax.random.PRNGKey(0), shape=(coeff.shape[0] - 1,), minval=V, maxval=U
        )
        phi = jax.random.uniform(
            jax.random.PRNGKey(0),
            shape=(coeff.shape[0] - 1,),
            minval=0,
            maxval=2 * jnp.pi,
        )
        return r * jnp.exp(1j * phi)

    roots = Roots0(coeff)
    return roots


@jax.jit
def Aberth_Ehrlich(
    coeff: jnp.ndarray, roots: jnp.ndarray, MAX_ITER: int = 50
) -> jnp.ndarray:
    """
    Aberth–Ehrlich iteration with differentiable custom_root wrapper.

    Parameters:
        coeff (jax.Array): Polynomial coefficients.
        roots (jax.Array): Initial guesses for the roots.
        MAX_ITER (int): Maximum number of iterations. Default 50.

    Returns:
        jax.Array: Refined roots for the input polynomial.
    """
    derp = jnp.polyder(coeff)
    mask = 1 - jnp.eye(roots.shape[0])
    # alpha = jnp.abs(coeff)*((2*jnp.sqrt(2))*1j+1)

    def loop_body(carry):
        roots, coeff, cond, ratio_old, n_iter = carry
        # h = jnp.polyval(coeff, roots)
        # b = jnp.polyval(alpha, jnp.abs(roots))
        ratio = jnp.polyval(coeff, roots) / jnp.polyval(derp, roots)

        sum_term = jnp.nansum(mask * 1 / (roots - roots[:, None]), axis=0)
        w = ratio / (1 - (ratio * sum_term))
        cond = jnp.abs(w) > 2e-14
        # cond = jnp.abs(h) > 1e-15*b
        roots -= w
        return (roots, coeff, cond, ratio, n_iter + 1)

    def cond_fun(carry):
        roots, coeff, cond, ratio, n_iter = carry
        return cond.any() & (n_iter < MAX_ITER)

    f = lambda x: jnp.polyval(coeff, x)
    solution = lambda f, x0: lax.while_loop(
        cond_fun, loop_body, (x0, coeff, jnp.ones_like(x0, dtype=bool), x0, 0)
    )[0]
    sclar = lambda g, y: jnp.linalg.solve(jax.jacobian(g, holomorphic=True)(y), y)

    return lax.custom_root(f, roots, solve=solution, tangent_solve=sclar)
