import jax
import jax.numpy as jnp
from microlux import point_light_curve, to_lowmass


def test_point_light_curve_smoke():
    t_0, u_0, t_E = 0.0, 0.1, 10.0
    rho, q, s, alphadeg = 1e-2, 0.2, 0.9, 270.0
    times = jnp.linspace(-0.1 * t_E, 0.1 * t_E, 16)
    alpha_rad = alphadeg * 2 * jnp.pi / 360
    tau = (times - t_0) / t_E
    traj = tau * jnp.exp(1j * alpha_rad) + 1j * u_0 * jnp.exp(1j * alpha_rad)
    traj_l = to_lowmass(s, q, traj)
    mag, cond = point_light_curve(traj_l, s, q, rho, tol=1e-3)
    assert mag.shape == times.shape
    assert cond.shape == times.shape
    assert jnp.isfinite(mag).all()

