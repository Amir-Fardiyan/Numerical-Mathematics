"""Solve the original implicit transport model and render a solution heatmap."""

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from numerical_math import solve_nonlinear

N = 500
Dt = 0.01
e = 0.1167
M = 1.1177


def f(s: ArrayLike) -> NDArray[np.float64]:
    """Evaluate the fractional-flow function on physical saturations."""
    clipped = np.clip(s, 0.0, 1.0)
    raised = clipped ** (2 + e)
    return raised / (raised + M * (1 - clipped) ** (2 + e))


def Res(S: ArrayLike, N: int, Dt: float) -> NDArray[np.float64]:
    """Assemble the upwind residual with unit inlet saturation."""
    saturation = np.asarray(S, dtype=float)
    if N < 1 or saturation.shape != (N,):
        raise ValueError("S must have one entry per cell")
    flux = f(saturation)
    upstream = np.concatenate(([f(1.0)], flux[:-1]))
    return saturation + N * Dt * (flux - upstream)


def transport_jacobian(saturation: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    """Build the lower bidiagonal Jacobian of the transport residual."""
    n = saturation.size
    clipped = np.clip(saturation, 0.0, 1.0)
    power = 2 + e
    denominator = clipped**power + M * (1 - clipped)**power
    slope = power * M * (clipped * (1 - clipped))**(power - 1) / denominator**2
    scaled = n * dt * slope
    return np.diag(1.0 + scaled) - np.diag(scaled[:-1], k=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=N)
    parser.add_argument("--output", type=Path, help="save the heatmap to this path")
    args = parser.parse_args()
    if args.cells < 1:
        parser.error("--cells must be positive")

    import matplotlib.pyplot as plt
    import seaborn as sns

    def residual(point: NDArray[np.float64]) -> NDArray[np.float64]:
        return Res(point, args.cells, Dt)

    def jacobian(point: NDArray[np.float64]) -> NDArray[np.float64]:
        return transport_jacobian(point, Dt)

    result = solve_nonlinear(residual, np.full(args.cells, 1e-8), jacobian)
    print(f"{result.status}: {result.accepted_steps} accepted steps, residual {result.residual_norms[-1]:.3e}")
    navy, muted, white = "#0b1728", "#abc0d6", "#e8f0fb"
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.7), layout="constrained", facecolor=navy)
    fig.suptitle(f"Implicit transport · {args.cells} cells", color=white, fontsize=16, fontweight="bold")
    views = (("Full domain", result.x), ("Inlet detail", result.x[:30]))
    for ax, (title, values) in zip(axes, views):
        sns.heatmap(values[None, :], cmap="viridis", vmin=0, vmax=1, xticklabels=False,
                    yticklabels=False, cbar_kws={"label": "Saturation"}, ax=ax)
        ticks = np.unique(np.linspace(0, values.size - 1, min(6, values.size), dtype=int))
        ax.set_xticks(ticks + 0.5, labels=ticks + 1)
        ax.tick_params(colors=muted)
        ax.set_title(title, color=white, loc="left", fontsize=11)
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(colors=muted)
        colorbar.ax.yaxis.label.set_color(muted)
    axes[-1].set_xlabel("Cell index (1-based)", color=muted)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
    else:
        plt.show()
    plt.close(fig)
    if not result.converged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
