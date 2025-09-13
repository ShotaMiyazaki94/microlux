import pytest
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

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from microlux import binary_mag
from tests_support import timeit, VBBL_light_curve



jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def time_test(t_0, u_0, t_E, rho, q, s, alpha_deg, times, retol, tol):
    ####################编译时间
    uniform_mag, time = timeit(binary_mag)(
        t_0, b, t_E, rho, q, s, alpha_deg, times, tol, retol=retol
    )
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    # jax.profiler.start_trace("/tmp/tensorboard")
    # with jax.disable_jit():
    # binary_mag(**parm).block_until_ready()
    # jax.profiler.stop_trace()
    VBBL_mag, _ = timeit(VBBL_light_curve)(
        t_0, b, t_E, rho, q, s, alpha_deg, times, retol=1e-3, tol=1e-3
    )

    plt.figure()
    plt.plot(times, VBBL_mag, label="VBBL")
    plt.plot(times, uniform_mag, label="microlux")
    plt.legend()
    plt.savefig("picture/diff.png")

    delta = np.abs(np.array(uniform_mag) - VBBL_mag) / VBBL_mag
    print("max error is", np.max(delta))
    print("indice of max error is", np.argmax(delta))
    # print('sample number is',info[-2].sample_num[np.argmax(delta)])
    # print('out loop number is',info[-1].outloop[np.argmax(delta)])
    # print('the array not enough is',info[-1].exceed_flag[np.argmax(delta)])


if __name__ == "__main__":
    b = 0.1
    t_0 = 8280
    t_E = 40
    alphadeg = 270
    q = 0.2
    s = 0.9
    rho = 10 ** (-2)
    trajectory_n = 1000
    times = jnp.linspace(t_0 - 1.0 * t_E, t_0 + 1.0 * t_E, trajectory_n)  # [idy:idy+1]
    time_test(t_0, b, t_E, rho, q, s, alphadeg, times, retol=1e-3, tol=1e-3)
