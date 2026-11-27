"""Sorting, linear iteration, and natural cubic spline interpolation."""

from bisect import bisect_right
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike

T = TypeVar("T")
U = TypeVar("U")
SplineCoefficients = tuple[list[float], list[float], list[float], list[float]]


def sort1d(values: list[T]) -> list[T]:
    """Sort values in place and return the same list."""
    values.sort()
    return values


def sort_pairs(keys: list[T], values: list[U]) -> tuple[list[T], list[U]]:
    """Stably sort two same-length lists in place by the first list."""
    if len(keys) != len(values):
        raise ValueError("paired lists must have the same length")
    order = sorted(range(len(keys)), key=keys.__getitem__)
    sorted_values = [values[i] for i in order]
    keys[:] = [keys[i] for i in order]
    values[:] = sorted_values
    return keys, values


def gauss_sidel(A: ArrayLike, b: ArrayLike, max_it: int, max_err: float) -> list[float]:
    """Iterate from zero and return the last iterate as a list.

    Stop when the largest component update is below max_err or the budget
    is exhausted. Check the residual separately for an accuracy estimate.
    """
    matrix = np.asarray(A, dtype=float)
    rhs = np.asarray(b, dtype=float)
    if rhs.ndim != 1 or not rhs.size or matrix.shape != (rhs.size, rhs.size):
        raise ValueError("A must be square and match a nonempty vector b")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        raise ValueError("A and b must be finite")
    if max_it < 0 or not np.isfinite(max_err) or max_err <= 0:
        raise ValueError("require max_it >= 0 and finite max_err > 0")
    diagonal = matrix.diagonal()
    if np.any(diagonal == 0):
        raise ValueError("Gauss–Seidel requires a nonzero diagonal")
    x = np.zeros_like(rhs)
    for _ in range(max_it):
        previous = x.copy()
        for i, row in enumerate(matrix):
            lower = row[:i] @ x[:i]
            upper = row[i + 1:] @ previous[i + 1:]
            x[i] = (rhs[i] - lower - upper) / diagonal[i]
        if not np.all(np.isfinite(x)):
            raise ArithmeticError("Gauss–Seidel produced a nonfinite iterate")
        if np.max(np.abs(x - previous)) < max_err:
            break
    return x.tolist()


def cubic_spline(x: list[float], y: list[float]) -> SplineCoefficients:
    """Sort the input lists in place and return coefficients in descending order."""
    if len(x) != len(y) or len(x) < 2 or not np.all(np.isfinite([x, y])):
        raise ValueError("a spline needs at least two finite x/y pairs")
    if len(set(x)) != len(x):
        raise ValueError("spline knots must be distinct")
    sort_pairs(x, y)
    widths = np.diff(np.asarray(x, dtype=float))
    slopes = np.diff(np.asarray(y, dtype=float)) / widths

    # Eliminate the tridiagonal system for the interior second derivatives.
    # Positive knot widths make the system strictly diagonally dominant.
    diagonal = 2 * (widths[:-1] + widths[1:])
    rhs = 6 * np.diff(slopes)
    for i in range(1, diagonal.size):
        factor = widths[i] / diagonal[i - 1]
        diagonal[i] -= factor * widths[i]
        rhs[i] -= factor * rhs[i - 1]
    second = np.zeros(len(x))
    for i in range(diagonal.size - 1, -1, -1):
        second[i + 1] = (rhs[i] - widths[i + 1] * second[i + 2]) / diagonal[i]

    a = np.diff(second) / (6 * widths)
    b = second[:-1] / 2
    c = slopes - widths * (2 * second[:-1] + second[1:]) / 6
    return a.tolist(), b.tolist(), c.tolist(), list(y[:-1])


def cubic_spline_interpolate(
    x: Sequence[float],
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
    d: Sequence[float],
    x_val: float,
) -> tuple[float, float, float] | None:
    """Return the spline and its first two derivatives inside the knot interval."""
    intervals = len(x) - 1
    if intervals < 1 or any(len(values) != intervals for values in (a, b, c, d)):
        raise ValueError("each coefficient sequence must match the number of intervals")
    if not x[0] <= x_val <= x[-1]:
        return None
    i = min(bisect_right(x, x_val) - 1, intervals - 1)
    offset = x_val - x[i]
    return (
        ((a[i] * offset + b[i]) * offset + c[i]) * offset + d[i],
        (3 * a[i] * offset + 2 * b[i]) * offset + c[i],
        6 * a[i] * offset + 2 * b[i],
    )
