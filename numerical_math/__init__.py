"""Numerical methods for small scientific computing problems."""

from .nonlinear import BroydenResult, broyden_trust_region, calculate_jacobian, solve_nonlinear
from .polynomial import bairstow
from .utilities import cubic_spline, cubic_spline_interpolate, gauss_sidel, sort1d, sort_pairs

__all__ = [
    "BroydenResult",
    "bairstow",
    "broyden_trust_region",
    "calculate_jacobian",
    "cubic_spline",
    "cubic_spline_interpolate",
    "gauss_sidel",
    "solve_nonlinear",
    "sort1d",
    "sort_pairs",
]
