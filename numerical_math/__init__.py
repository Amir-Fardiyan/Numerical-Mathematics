"""Numerical methods for small scientific computing problems."""

from .nonlinear import broyden_trust_region, calculate_jacobian
from .polynomial import bairstow
from .utilities import cubic_spline, cubic_spline_interpolate, gauss_sidel, sort1d, sort_pairs

__all__ = [
    "bairstow",
    "broyden_trust_region",
    "calculate_jacobian",
    "cubic_spline",
    "cubic_spline_interpolate",
    "gauss_sidel",
    "sort1d",
    "sort_pairs",
]
