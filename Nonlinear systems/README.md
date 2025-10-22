# Broyden--Trust‑Region Solver

## Abstract

This repository provides a concise implementation of a Broyden (rank‑1) quasi‑Newton method embedded in a trust‑region framework for solving square nonlinear systems \(F:\mathbb{R}^n \to \mathbb{R}^n\) with \(F(x^*)=0\). The code is dependency‑light (NumPy only) and aims to be readable and reproducible for research and coursework.

---

## Contents

* `broyden_trust_region(func, grad, x0, epsilon=1e-12, rho=0.1, c=0.5, max_iter=100)` — core solver
* `calculate_jacobian(func, x, delta=1e-5)` — centered finite‑difference Jacobian (optional helper)
* Linear algebra utilities used by the solver:
  * `lu_decomposition(A)` — dense LU factorization (no pivoting)
  * `solve_lu(L, U, b)` — forward/backward substitution
* `calculate_norm(vector, order=2)` — \(\ell_1\) and \(\ell_2\) norms

---

## Requirements and Installation

* Python \(\ge\) 3.10

Install:

```bash
pip install numpy
pip install matplotlib
pip install seaborn

```

---

## Mathematical Formulation

We seek \(x \in \mathbb{R}^n\) such that \(F(x)=0\). At iterate \(x_k\), the method maintains an approximation \(B_k \approx J_F(x_k)\), solves the local linear model \(B_k\, d_k = -F(x_k)\), and restricts the step to a trust region of radius \(\Delta_k\). Step acceptance uses the ratio

$$
r_k =\frac{\|F(x_k)\|_2^2 - \|F(x_k + d_k)\|_2^2}{-F(x_k)^\top d_k - 0.5\, d_k^\top B_k d_k}.
$$

If \(r_k \ge \rho\), the step is accepted; otherwise, the radius is reduced. The Jacobian approximation is updated by Broyden’s rank‑1 formula

$$
B_{k+1} = B_k + \frac{\big(y_k - B_k s_k\big) s_k^\top}{s_k^\top s_k},
\qquad s_k = d_k,\quad y_k = F(x_{k+1}) - F(x_k).
$$

The trust‑region radius is reset as

$$
\Delta_{k+1} = c\, \|F(x_{k+1})\|_2.
$$

---

## Algorithm (high‑level)

1. **Initialization:** set \(x_0\), build \(B_0\) (e.g., by finite differences).
2. **TR subproblem:** solve \(B_k d_k = -F(x_k)\); if \(\|d_k\|_2 > \Delta_k\), scale radially to the boundary.
3. **Acceptance:** compute \(r_k\); accept if \(r_k \ge \rho\), else reduce \(\Delta_k \leftarrow c\, \Delta_k\) and retry.
4. **Update:** apply rank‑1 Broyden update to obtain \(B_{k+1}\); set \(\Delta_{k+1} = c\,\|F(x_{k+1})\|_2\).
5. **Termination:** stop when \(\|F(x_k)\|_2 \le \epsilon\) or `max_iter` is reached.

---

## API Specification

### `broyden_trust_region(func, grad, x0, epsilon=1e-12, rho=0.1, c=0.5, max_iter=100)`

**Parameters**

* `func`: callable, returns `np.ndarray` shape `(n,)` for input `x` of shape `(n,)`.
* `grad`: callable, returns Jacobian approximation `np.ndarray` shape `(n, n)` at `x`.
* `x0`: initial guess, `np.ndarray` shape `(n,)`.
* `epsilon`: tolerance on \(\|F(x)\|_2\) for convergence.
* `rho`: acceptance threshold for the ratio \(r_k\).
* `c`: trust‑region radius scaling factor.
* `max_iter`: iteration cap.

**Returns**

* `x`: final iterate, `np.ndarray` shape `(n,)`.
* `iter_count`: number of accepted iterations (`int`).

### `calculate_jacobian(func, x, delta=1e-5)`

Centered finite‑difference estimate of the Jacobian. Useful to initialize \(B_0\) when an analytic Jacobian is unavailable.

---

## Usage Example (2‑D System)

```python
import numpy as np
from broyden_trust_region import broyden_trust_region, calculate_jacobian

# Define F: R^2 -> R^2
def F(x):
    return np.array([
        x[0]**2 + x[1]**2 - 1.0,  # unit circle
        x[0] - x[1]                # line x = y
    ])

def J(x):
    return calculate_jacobian(F, x)

x0 = np.array([0.8, 0.3])
sol, iters = broyden_trust_region(F, J, x0, epsilon=1e-10, rho=0.1, c=0.5, max_iter=100)
print("Solution:", sol)
print("Iterations:", iters)
```

---

## Practical Considerations

* **Scaling:** rescale variables/equations to comparable magnitudes to improve conditioning.
* **Linear algebra:** current LU is dense and without pivoting. For ill‑conditioned problems, add partial pivoting or use a robust factorization.
* **Large‑scale use:** complexity is \(\mathcal{O}(n^3)\) per factorization. For large \(n\), employ sparse/banded structures and solvers.
* **Refresh strategy:** periodically reinitialize \(B_k\) with a finite‑difference Jacobian if progress stalls.
* **Stopping criteria:** besides \(\|F(x)\|_2\), one may also monitor step norm and relative decrease of \(\|F\|_2^2\).

---

## Troubleshooting

* **Stagnation or frequent rejection:** decrease \(c\) to reduce step size; relax \(\rho\); or rebuild \(B_k\) from FD Jacobian.
* **Singular model:** if solving \(B_k d = -F\) fails, refresh \(B_k\) or add regularization (e.g., \(B_k + \lambda I\)).
* **Non‑smooth systems:** Broyden may struggle with kinks; consider smoothing or using safeguarded methods.

---

## References

* Broyden, C. G. (1965). *A class of methods for solving nonlinear simultaneous equations*. Mathematics of Computation, 19(92), 577–593.
* Zeng, M. L., & Fu, H. W. (2019). *A Broyden trust‑region quasi‑Newton method for nonlinear equations*. IAENG International Journal of Computer Science, 46(3).
* Dennis, J. E., & Schnabel, R. B. (1996). *Numerical Methods for Unconstrained Optimization and Nonlinear Equations*. SIAM.
* Conn, A. R., Gould, N. I. M., & Toint, Ph. L. (2000). *Trust‑Region Methods*. SIAM.
* Kelley, C. T. (1995). *Iterative Methods for Linear and Nonlinear Equations*. SIAM.
* Nocedal, J., & Wright, S. J. (2006, 2nd ed.). *Numerical Optimization*. Springer.

