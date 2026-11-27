"""Polynomial roots by quadratic deflation with a companion-matrix fallback."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _synthetic_division(coefficients: NDArray[np.float64], r: float, s: float) -> NDArray[np.float64]:
    """Divide by x² - r*x - s, storing the remainder in the final two slots."""
    result = coefficients.copy()
    result[1] += r * result[0]
    for i in range(2, result.size):
        result[i] += r * result[i - 1] + s * result[i - 2]
    return result


def bairstow(coefficients: ArrayLike, max_it: int, err: float) -> NDArray[np.complex128]:
    """Find roots from real coefficients ordered highest degree first.

    Attempt at most max_it Bairstow updates per quadratic factor. A stalled
    or singular correction falls back to companion-matrix roots. Root order
    is unspecified. The zero polynomial has no finite root list.
    """
    original = np.asarray(coefficients, dtype=float)
    if original.ndim != 1 or not np.all(np.isfinite(original)):
        raise ValueError("coefficients must be a finite real vector")
    original = np.trim_zeros(original, trim="f")
    if original.size == 0:
        raise ValueError("the zero polynomial has infinitely many roots")
    if max_it < 0 or not np.isfinite(err) or err <= 0:
        raise ValueError("require max_it >= 0 and finite err > 0")

    roots: list[complex] = []
    polynomial = original
    while polynomial.size > 3:
        polynomial = polynomial / np.max(np.abs(polynomial))
        r, s = -1.0, -1.0
        with np.errstate(over="ignore", invalid="ignore"):
            for updates in range(max_it + 1):
                divided = _synthetic_division(polynomial, r, s)
                if not np.all(np.isfinite(divided)):
                    break
                if np.max(np.abs(divided[-2:])) <= err:
                    roots.extend(np.roots([1, -r, -s]))
                    polynomial = divided[:-2]
                    break
                if updates == max_it:
                    break
                derivative = _synthetic_division(divided[:-1], r, s)
                jacobian = np.array([[derivative[-2], derivative[-3]], [derivative[-1], derivative[-2]]])
                if not np.all(np.isfinite(jacobian)):
                    break
                try:
                    correction = np.linalg.solve(jacobian, -divided[-2:])
                except np.linalg.LinAlgError:
                    break
                r, s = np.array([r, s]) + correction
        if polynomial.size == divided.size:
            return np.asarray(np.roots(original), dtype=complex)

    roots.extend(np.roots(polynomial))
    candidates = np.asarray(roots, dtype=complex)
    # Deflation can magnify error. Check coefficient-scaled residuals against
    # the original polynomial before accepting the full set of roots.
    normalized = original / np.max(np.abs(original))
    with np.errstate(over="ignore", invalid="ignore"):
        residual = np.abs(np.polyval(normalized, candidates))
        scale = np.polyval(np.abs(normalized), np.abs(candidates))
        if np.all(np.isfinite(residual) & (residual <= 10 * err * scale)):
            return candidates
    return np.asarray(np.roots(original), dtype=complex)
