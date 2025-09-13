"""
microlux public API.

Re‑exports commonly used functions and classes for convenience and sets JAX
defaults for double precision globally.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Enable 64-bit floats everywhere (set once at import time).
# Be robust if JAX is mocked (e.g. during Sphinx autodoc).
try:  # pragma: no cover - docs build compatibility
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass
# # -*- coding: utf-8 -*-
__all__ = [
    "point_light_curve",
    "extended_light_curve",
    "contour_integral",
    "binary_mag",
    "Iterative_State",
    "Error_State",
    "to_lowmass",
    "to_centroid",
    "LinearLimbDarkening",
]

from .core.lens_equation import (
    to_centroid as to_centroid,
    to_lowmass as to_lowmass,
)
from .algorithms.contour import contour_integral as contour_integral
from .model import (
    binary_mag as binary_mag,
    extended_light_curve as extended_light_curve,
    point_light_curve as point_light_curve,
)
from .core.state import (
    Error_State as Error_State,
    Iterative_State as Iterative_State,
)
from .physics.limb_darkening import LinearLimbDarkening as LinearLimbDarkening

# Provide a JAX-friendly `jnp.roots` if missing (used in tests)
try:  # pragma: no cover - compatibility shim
    _ = jnp.roots  # type: ignore[attr-defined]
except AttributeError:  # Assign only if not present
    from .numerics.polynomial import roots as _poly_roots

    # Expose as jnp.roots so downstream code/tests can call `jnp.roots(...)`
    setattr(jnp, "roots", _poly_roots)
