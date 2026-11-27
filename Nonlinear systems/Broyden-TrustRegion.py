"""Compatibility entry point for the original Broyden script."""

import sys
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

# Direct script execution starts with this directory on the module search path.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.transport import Dt, M, N, Res, e, f, main
from numerical_math.nonlinear import FloatArray, broyden_trust_region, calculate_jacobian


def calculate_norm(vector: ArrayLike, order: int = 2) -> float:
    """Compute the L1 or L2 norm of a vector."""
    vector = np.asarray(vector, dtype=float)
    if order == 1:
        return float(np.linalg.norm(vector, ord=1))
    elif order == 2:
        return float(np.linalg.norm(vector))
    else:
        raise ValueError("Unsupported norm order")

def lu_decomposition(A: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Factor a square matrix without pivoting for the legacy helper API."""
    U = np.array(A, dtype=float, copy=True)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("A must be square")
    n = U.shape[0]
    L = np.eye(n)  # Initialize L as identity matrix

    for i in range(n):
        if U[i, i] == 0:
            raise np.linalg.LinAlgError("zero pivot in unpivoted LU")
        # Update rows below the pivot
        factors = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factors
        U[i+1:] -= np.outer(factors, U[i])

    return L, U

def solve_lu(L: FloatArray, U: FloatArray, b: ArrayLike) -> FloatArray:
    """Solve a factored system by forward and backward substitution."""
    b = np.asarray(b, dtype=float)
    n = len(b)

    # Forward substitution to solve L * y = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Backward substitution to solve U * x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

s0 = np.full(N, 1e-8)
delta = 0.1177


def func(S: ArrayLike) -> FloatArray:
    """Evaluate the original example's residual."""
    return Res(S, N, Dt)


def jacobian(S: ArrayLike) -> FloatArray:
    """Estimate the original example's Jacobian by centered differences."""
    return calculate_jacobian(func, S)


if __name__ == "__main__":
    main()
