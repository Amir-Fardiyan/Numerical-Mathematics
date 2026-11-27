"""Dense Broyden iteration with residual-based trust-region acceptance."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
VectorFunction = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True)
class BroydenResult:
    """Final iterate, stopping reason, and residual norms at accepted iterates."""

    x: FloatArray
    accepted_steps: int
    trial_steps: int
    residual_norms: tuple[float, ...]
    status: str

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def _vector(values: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a nonempty, finite vector")
    return vector


def _residual(func: VectorFunction, x: FloatArray) -> FloatArray:
    values = _vector(func(x), "residual")
    if values.shape != x.shape:
        raise ValueError("residual must have the same shape as x")
    return values


def calculate_jacobian(func: VectorFunction, x: ArrayLike, delta: float = 1e-5) -> FloatArray:
    """Estimate the Jacobian of a square system with scaled centered differences."""
    point = _vector(x, "x")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    jacobian = np.empty((point.size, point.size))
    for column, value in enumerate(point):
        step = delta * max(1.0, abs(value))
        left, right = point.copy(), point.copy()
        left[column] -= step
        right[column] += step
        span = right[column] - left[column]
        if span == 0 or not np.isfinite(span):
            raise ValueError("delta does not yield a finite, representable difference")
        right_residual = _residual(func, right).copy()
        jacobian[:, column] = (right_residual - _residual(func, left)) / span
    return jacobian


def solve_nonlinear(
    func: VectorFunction,
    x0: ArrayLike,
    grad: VectorFunction | None = None,
    *,
    epsilon: float = 1e-12,
    rho: float = 0.1,
    c: float = 0.5,
    max_iter: int = 100,
) -> BroydenResult:
    """Solve F(x) = 0 using at most max_iter trial steps.

    The residual history includes the initial norm. A status of
    `stagnated` indicates the trial step cannot change the iterate.
    """
    if not (np.isfinite(epsilon) and epsilon > 0 and 0 < rho < 1 and 0 < c < 1):
        raise ValueError("require epsilon > 0 and rho and c in (0, 1)")
    if max_iter < 0:
        raise ValueError("max_iter must be nonnegative")

    x = _vector(x0, "x0").copy()
    residual = _residual(func, x).copy()
    history = [float(np.linalg.norm(residual))]
    trials = 0
    status = "converged" if history[-1] <= epsilon else "trial_limit"
    if status == "converged" or max_iter == 0:
        return BroydenResult(x, 0, trials, tuple(history), status)

    matrix = np.array(grad(x) if grad else calculate_jacobian(func, x), dtype=float, copy=True)
    if matrix.shape != (x.size, x.size) or not np.all(np.isfinite(matrix)):
        raise ValueError("Jacobian must be a finite square matrix matching x")
    radius = c * max(1.0, float(np.linalg.norm(x)))
    for trials in range(1, max_iter + 1):
        # SVD-based least squares also gives a direction for rank-deficient models.
        step = np.linalg.lstsq(matrix, -residual, rcond=None)[0]
        length = np.linalg.norm(step)
        if length > radius:
            step *= radius / length
        candidate = x + step
        if np.array_equal(candidate, x):
            status = "stagnated"
            break
        step = candidate - x
        model = residual + matrix @ step
        # Factored differences avoid subtracting two nearly equal squared norms.
        predicted = float((residual - model) @ (residual + model))
        trial_values = np.asarray(func(candidate), dtype=float)
        if trial_values.shape != x.shape:
            raise ValueError("residual must have the same shape as x")
        if not np.all(np.isfinite(trial_values)) or predicted <= 0:
            radius *= c
            continue
        actual = float((residual - trial_values) @ (residual + trial_values))
        ratio = actual / predicted
        if not np.isfinite(ratio) or ratio < rho:
            radius *= c
            continue

        if ratio < 0.25:
            radius *= c
        elif ratio > 0.75 and length >= 0.9 * radius:
            radius /= c
        step_norm = np.linalg.norm(step)
        direction = step / step_norm
        matrix += np.outer((trial_values - residual - matrix @ step) / step_norm, direction)
        x, residual = candidate, trial_values.copy()
        history.append(float(np.linalg.norm(residual)))
        if history[-1] <= epsilon:
            status = "converged"
            break

    return BroydenResult(x, len(history) - 1, trials, tuple(history), status)


def broyden_trust_region(
    func: VectorFunction,
    grad: VectorFunction,
    x0: ArrayLike,
    epsilon: float = 1e-12,
    rho: float = 0.1,
    c: float = 0.5,
    max_iter: int = 100,
) -> tuple[FloatArray, int]:
    """Return the final iterate and accepted-step count using the original API."""
    result = solve_nonlinear(func, x0, grad, epsilon=epsilon, rho=rho, c=c, max_iter=max_iter)
    return result.x, result.accepted_steps
