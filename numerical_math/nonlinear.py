"""Broyden iteration with a residual-based trust-region radius."""

import numpy as np

def calculate_norm(vector, order=2):
    if order == 1:
        return np.sum(np.abs(vector))
    elif order == 2:
        return np.sqrt(np.sum(vector**2))
    else:
        raise ValueError("Unsupported norm order")

def lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)  # Initialize L as identity matrix
    U = A.copy()   # Initialize U as a copy of A

    for i in range(n):
        # Update rows below the pivot
        factors = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factors
        U[i+1:] -= np.outer(factors, U[i])

    return L, U

def solve_lu(L, U, b):
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

def broyden_trust_region(func, grad, x0, epsilon=1e-12, rho=0.1, c=0.5, max_iter=100):
    def trust_region_subproblem(B, F, delta):

        L, U = lu_decomposition(B)
        d = solve_lu(L, U, -F)

        if calculate_norm(d) <= delta:
            return d
        else:
            return delta * d / calculate_norm(d)

    def actual_reduction(func, xk, d):
        """Calculate actual reduction."""
        return (func(xk).T @ func(xk) - func(xk + d).T @ func(xk + d))

    def predicted_reduction(Fk, Bk, d):
        """Calculate predicted reduction."""
        return (-Fk.T @ d - 0.5 * d.T @ Bk @ d)

    xk = x0
    Bk = grad(x0)  # Initial Jacobian approximation
    delta = c * calculate_norm(func(xk))  # Initial trust region radius
    iter_count = 0

    while iter_count < max_iter:
        Fk = func(xk)
        if calculate_norm(Fk) <= epsilon:
            print("Converged!")
            break

        # Solve the trust region subproblem
        d = trust_region_subproblem(Bk, Fk, delta)

        # Compute the ratio
        ared = actual_reduction(func, xk, d)
        pred = predicted_reduction(Fk, Bk, d)
        rk = ared / pred if pred != 0 else -1

        if rk >= rho:
            xk = xk + d
        else:
            delta *= c  # Reduce trust region radius
            continue

        # Update Bk using Broyden's formula
        sk = d
        yk = func(xk) - Fk
        Bk += (yk - Bk @ sk).reshape(-1, 1) @ sk.reshape(1, -1) / (sk.T @ sk)

        # Reset the trust region radius and increment iteration
        delta = c * calculate_norm(func(xk))
        iter_count += 1

    return xk, iter_count

def calculate_jacobian(func, x, delta=1e-5):
    n = x.shape[0]
    J = np.zeros((n, n))
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += delta
        x_minus[i] -= delta
        J[:, i] = (func(x_plus) - func(x_minus)) / (2 * delta)
    return J
