from numerical_math import bairstow, cubic_spline, cubic_spline_interpolate, gauss_sidel
from numerical_math import sort1d, sort_pairs
from numpy.typing import ArrayLike


class utilities:
    @staticmethod
    def sort1D(u):
        return sort1d(u)

    @staticmethod
    def sort2D(u, v):
        return sort_pairs(u, v)

    gauss_sidel = staticmethod(gauss_sidel)
    cubic_spline = staticmethod(cubic_spline)
    cubic_spline_interpolate = staticmethod(cubic_spline_interpolate)

    @staticmethod
    def bairstow(coefficients: ArrayLike, max_it: int, err: float):
        roots = bairstow(coefficients, max_it, err)
        return roots.real if not roots.imag.any() else roots

    bairstow_EoS = bairstow
