"""
microlux public API.

Re‑exports commonly used functions and classes for convenience.
"""
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
