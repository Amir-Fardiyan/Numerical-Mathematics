# Numerical Mathematics

Python implementations of nonlinear root finding, iterative linear solvers, natural cubic splines, and polynomial root finding. The examples show how to measure convergence, verify numerical results, and visualize solutions.

![Log-scale residual histories for Broyden and Gauss–Seidel](docs/figures/convergence.png)

## Quick start

With Python 3.10 or later, activate a virtual environment and run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The core package uses NumPy. Matplotlib and Seaborn are used by the demonstrations.

```python
from numerical_math import solve_nonlinear


def residual(x):
    return [x @ x - 1, x[0] - x[1]]


result = solve_nonlinear(residual, [0.8, 0.3])
print(result.x, result.status, result.residual_norms[-1])
```

The solution near this starting point is $(1/\sqrt{2}, 1/\sqrt{2})$. Check `result.converged` and the final residual before using the iterate.

## Methods and examples

| Method | Example | Documentation |
| --- | --- | --- |
| Broyden with trust-region step control | Find the intersection of a unit circle and a line | [Nonlinear systems](Nonlinear%20systems/README.md) |
| Gauss–Seidel | Solve a diagonally dominant system and check its residual | [Linear systems and utilities](utilities/README.md#gaussseidel) |
| Natural cubic spline | Interpolate $\sin x$ on nonuniform knots | [Spline construction and evaluation](utilities/README.md#natural-cubic-spline) |
| Bairstow and quadratic deflation | Find real and complex polynomial roots | [Polynomial roots](utilities/README.md#bairstows-method) |

## Reproduce the figures

```sh
python -m examples.demonstrations --output-dir generated
python -m examples.transport --output generated/transport.png
```

[Interpolation plot](docs/figures/interpolation.png) · [500-cell transport heatmap](docs/figures/transport.png)

The convergence figure plots accepted Broyden iterates and Gauss–Seidel sweeps on a logarithmic scale. Values at or below machine epsilon are displayed at machine epsilon. The transport example solves the original implicit upwind system and reports the final residual.

## Project layout

| Path | Purpose |
| --- | --- |
| [numerical_math](numerical_math) | Importable numerical methods |
| [examples](examples) | Scripts that regenerate the figures |
| [tests](tests) | Analytic solutions, edge cases, and compatibility checks |
| [CI workflow](.github/workflows/ci.yml) | Tests and plot generation on Linux and Windows |

The original `utilities.Utilities.utilities` import and [Broyden script](Nonlinear%20systems/Broyden-TrustRegion.py) remain available. These implementations use dense routines for small problems. The method guides describe where to check residuals and how scaling and conditioning can affect results.
