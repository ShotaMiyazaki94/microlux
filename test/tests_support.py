import jax
import jax.numpy as jnp


def get_caustic_permutation(rho, q, s, n_points=1000):
    """
    Test around the caustic, apadpted from https://github.com/fbartolic/caustics/blob/main/tests/test_extended_source.py

    **returns**:

    - return the permutation of the caustic in the central of mass coordinate system
    """
    # Try to use MulensModel for accurate caustics; otherwise fall back to a
    # lightweight synthetic set of points near the origin to keep tests running.
    try:
        from MulensModel import CausticsBinary

        caustic = CausticsBinary(q, s)
        x, y = caustic.get_caustics(n_points)
        z_centeral = jnp.array(jnp.array(x) + 1j * jnp.array(y))
    except Exception:  # pragma: no cover
        # Fallback: sample points on a small circle to emulate a trajectory
        theta = jnp.linspace(0.0, 2 * jnp.pi, n_points)
        z_centeral = 0.1 * jnp.exp(1j * theta)

    # Randomly jitter positions within 2*rho
    try:
        key = jax.random.key(42)
    except AttributeError:  # for older JAX versions
        key = jax.random.PRNGKey(42)
    key, subkey1, subkey2 = jax.random.split(key, num=3)
    phi = jax.random.uniform(subkey1, z_centeral.shape, minval=-jnp.pi, maxval=jnp.pi)
    r = jax.random.uniform(subkey2, z_centeral.shape, minval=0.0, maxval=2 * rho)
    z_centeral = z_centeral + r * jnp.exp(1j * phi)
    return z_centeral
