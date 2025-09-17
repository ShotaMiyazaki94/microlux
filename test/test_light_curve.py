import jax
import jax.numpy as jnp
from microlux import binary_mag


def test_binary_mag_smoke():
    t_0, u_0, t_E = 8280, 0.1, 10.0
    rho, q, s, alpha_deg = 1e-2, 0.2, 0.9, 270
    times = jnp.linspace(t_0 - 0.1 * t_E, t_0 + 0.1 * t_E, 16)
    mag = binary_mag(t_0, u_0, t_E, rho, q, s, alpha_deg, times, tol=1e-3, retol=1e-3)
    assert mag.shape == times.shape
    assert jnp.isfinite(mag).all()
