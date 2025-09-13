"""
Benchmark magnification map generation with VBBL vs. JAX (microlux).

This is a standalone, heavy benchmark script (not a pytest). Run directly:

    python scripts/benchmark_mag_map.py

Optional deps: VBBinaryLensing (for VBBL reference), tqdm (progress bar).
"""

import os
import time
from functools import partial
from multiprocessing import Pool

import numpy as np

# Lightweight tqdm fallback
try:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


def VBBL_light_curve(
    t_0,
    u_0,
    t_E,
    rho,
    q,
    s,
    alpha_deg,
    times,
    retol=0.0,
    tol=1e-2,
    limb_darkening: None | float = None,
):
    """
    VBBinaryLensing light curve in the centroid coordinate system.

    Returns numpy array of magnification values. Requires VBBinaryLensing.
    """
    try:
        import VBBinaryLensing
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "VBBinaryLensing is required for VBBL comparison. Please install it."
        ) from e

    VBBL = VBBinaryLensing.VBBinaryLensing()
    if limb_darkening is not None:
        VBBL.a1 = limb_darkening
    alpha_VBBL = np.pi + alpha_deg / 180 * np.pi
    VBBL.Tol = tol
    VBBL.RelTol = retol
    times = np.array(times)
    tau = (times - t_0) / t_E
    y1 = -u_0 * np.sin(alpha_VBBL) + tau * np.cos(alpha_VBBL)
    y2 = u_0 * np.cos(alpha_VBBL) + tau * np.sin(alpha_VBBL)
    params = [np.log(s), np.log(q), u_0, alpha_VBBL, np.log(rho), np.log(t_E), t_0]
    VBBL_mag = VBBL.BinaryLightCurve(params, times, y1, y2)
    return np.array(VBBL_mag)


np.seterr(divide="ignore", invalid="ignore")


def mag_map_vbbl(i, all_params, fix_params):
    """Compute a VBBL magnification map for a single parameter triple.

    Note: Avoids using globals so it works under multiprocessing 'spawn'.
    """
    rho, q, s = all_params[i]
    t_0, b_map, t_E, alphadeg, times, tol = fix_params
    sample_n = len(b_map)
    trajectory_n = len(times)
    VBBL_mag_map = np.zeros((sample_n, trajectory_n), dtype=np.float32)
    mag_vbbl = lambda i: VBBL_light_curve(
        t_0, b_map[i], t_E, rho, q, s, alphadeg, times, 1e-4, 1e-3
    )
    for i in range(sample_n):
        VBBL_mag_map[i, :] = mag_vbbl(i)
    return VBBL_mag_map, i


if __name__ == "__main__":
    # Optional: allow user to control chunk size via env var; default 64
    chunk_size = int(os.getenv("MICROLUX_BENCH_CHUNK", "64"))
    # mp.set_start_method('spawn')
    trajectory_n = 1000
    sample_n = 1000
    tol = 1e-3
    t_0 = 8000
    t_E = 50
    times = np.linspace(t_0 - 0.0 * t_E, t_0 + 2.0 * t_E, trajectory_n)
    alphadeg = 270
    tau = (times - t_0) / t_E

    b_map = np.linspace(-4.0, 3.0, sample_n)

    np.random.seed(42)

    N_Runs = 1000
    N_save = 100

    parameter_space = np.zeros((N_Runs, 3))
    VBBL_mag_map_list, jax_map_list = [], []
    sample_num_list, exceed_flag_list = [], []

    jax_time = 0

    for k in range(N_Runs):
        q = 10 ** (np.random.uniform(-6.0, 0.0))
        s = 10 ** np.random.uniform(-0.5, 0.5)
        rho = 10 ** (np.random.uniform(-3.0, -1.0))
        parameter_space[k] = [rho, q, s]

    # vbbl mag map test

    start = time.perf_counter()
    mag_vbbl_warp = partial(
        mag_map_vbbl,
        all_params=parameter_space,
        fix_params=[t_0, b_map, t_E, alphadeg, times, tol],
    )
    # Use a reasonable worker count for portability
    vbbl_workers = min(max(1, os.cpu_count() or 1), 8)
    with Pool(processes=vbbl_workers) as pool:
        for VBBL_mag, i in tqdm(
            pool.imap(mag_vbbl_warp, range(N_Runs)), total=N_Runs, mininterval=1
        ):
            VBBL_mag = VBBL_mag.astype(np.float32)
            VBBL_mag_map_list.append(VBBL_mag)

    print(f"vbbl time took: {time.perf_counter() - start:.4f}")
    os.makedirs("test_result", exist_ok=True)
    np.savez("test_result/vbbl_map_time_test.npz", VBBL_mag_map_list=VBBL_mag_map_list)

    # jax mag map test

    # Import JAX after any XLA flags/env setup (none by default)
    import jax
    import jax.numpy as jnp
    from microlux import binary_mag

    def mag_jax(i, rho, q, s, parm):
        t_0, b_map, t_E, alphadeg, times_jax, tol = parm
        uniform_mag, info = binary_mag(
            t_0,
            b_map[i],
            t_E,
            rho,
            q,
            s,
            alphadeg,
            times_jax,
            tol=1e-3,
            retol=1e-3,
            analytic=False,
            return_info=True,
            default_strategy=(30, 30, 60, 120, 240, 480),
        )
        return uniform_mag, info[-2].sample_num, info[-1].exceed_flag

    # Vectorized mapping over sample indices, processed in chunks
    vmapped = jax.vmap(mag_jax, in_axes=(0, None, None, None, None))

    for k in range(N_Runs):
        rho, q, s = parameter_space[k]
        jax_map = []
        sample_num = []
        exceed_flag = []

        parm = [t_0, jnp.array(b_map), t_E, alphadeg, jnp.array(times), tol]
        all_nodes = jnp.arange(sample_n)
        # Warmup compile on first run
        if k == 0:
            _ = vmapped(all_nodes[: min(chunk_size, sample_n)], rho, q, s, parm)

        start = time.monotonic()
        for i in range(0, sample_n, chunk_size):
            slic = all_nodes[i : i + chunk_size]
            mag_i, sample_num_i, exceed_flag_i = vmapped(slic, rho, q, s, parm)
            jax_map.append(mag_i)
            sample_num.append(sample_num_i)
            exceed_flag.append(exceed_flag_i)

        jax_map = jnp.concatenate(jax_map, axis=0).astype(jnp.float32)
        sample_num = jnp.concatenate(sample_num, axis=0).astype(jnp.int32)
        exceed_flag = jnp.concatenate(exceed_flag, axis=0)

        jax_time += time.monotonic() - start
        jax_map_list.append(jax_map)
        sample_num_list.append(sample_num)
        exceed_flag_list.append(exceed_flag)

        if (k + 1) % N_save == 0:
            np.savez(
                "test_result/jax_map_test.npz",
                jax_map_list=jax_map_list,
                sample_num_list=sample_num_list,
                exceed_flag_list=exceed_flag_list,
            )
            print("{}/{} run saved".format(k + 1, N_Runs))
    print("jax time took: {:.4f}".format(jax_time))
