import jax
import jax.numpy as jnp
from microlux import binary_mag


def test_grad_fwd_rev_consistency():
    # Tiny problem size to keep CI fast
    t_0, u_0, t_E = 8280.0, 0.1, 20.0
    rho, q, s, alpha_deg = 1e-2, 0.2, 0.9, 60.0
    times = jnp.linspace(t_0 - 0.1 * t_E, t_0 + 0.1 * t_E, 16)

    def scalar_fun(params):
        t_0, u_0, t_E, rho, q, s, alpha_deg = params
        mag = binary_mag(t_0, u_0, t_E, rho, q, s, alpha_deg, times, tol=1e-3, retol=1e-3)
        return jnp.sum(mag)

    params = (t_0, u_0, t_E, rho, q, s, alpha_deg)
    g_fwd = jax.jacfwd(scalar_fun)(params)
    g_rev = jax.jacrev(scalar_fun)(params)
    assert jnp.allclose(jnp.stack(g_fwd), jnp.stack(g_rev), rtol=1e-6, atol=1e-8)

