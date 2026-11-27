"""Generate the repository's convergence and interpolation figures."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numerical_math import cubic_spline, cubic_spline_interpolate, gauss_sidel, solve_nonlinear

NAVY = "#0b1728"
PANEL = "#13243a"
TEXT = "#e8f0fb"
MUTED = "#abc0d6"
TEAL = "#59e0cf"
GOLD = "#ffc776"


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=MUTED, alpha=0.14)
    for spine in ax.spines.values():
        spine.set_color("#334d66")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)

    def residual(x):
        return [x @ x - 1, x[0] - x[1]]

    result = solve_nonlinear(residual, [0.8, 0.3])
    if not result.converged:
        raise RuntimeError(f"Circle–line example stopped with status {result.status}")
    matrix = np.array([[4, 1], [2, 3]])
    rhs = np.array([1, 2])
    budgets = range(15)
    linear = [np.linalg.norm(matrix @ gauss_sidel(matrix, rhs, k, 1e-15) - rhs) for k in budgets]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained", facecolor=NAVY)
    fig.suptitle("Convergence measured by the residual", color=TEXT, fontsize=16, fontweight="bold")
    cases = (
        ("Broyden · circle and line", "Accepted steps", result.residual_norms, TEAL),
        ("Gauss–Seidel · two-variable system", "Sweeps", linear, GOLD),
    )
    for ax, (title, label, norms, color) in zip(axes, cases):
        style(ax)
        ax.semilogy(range(len(norms)), np.maximum(norms, np.finfo(float).eps),
                    marker="o", markersize=4, linewidth=2, color=color)
        ax.set_title(title, color=TEXT, loc="left", fontsize=11)
        ax.set(xlabel=label, ylabel="Euclidean residual norm")
        ax.xaxis.get_major_locator().set_params(integer=True)
    fig.savefig(output / "convergence.png", dpi=160)
    plt.close(fig)

    knots = [0.0, 0.6, 1.6, 2.6, 3.8, 5.2, 2 * np.pi]
    samples = np.sin(knots).tolist()
    coefficients = cubic_spline(knots, samples)
    domain = np.linspace(knots[0], knots[-1], 400)
    interpolated = [cubic_spline_interpolate(knots, *coefficients, x)[0] for x in domain]
    fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained", facecolor=NAVY)
    style(ax)
    ax.set_title("Natural cubic spline on nonuniform knots", color=TEXT, loc="left",
                 fontsize=15, fontweight="bold")
    ax.plot(domain, interpolated, color=TEAL, linewidth=2.6, label="Natural spline")
    ax.plot(domain, np.sin(domain), color=TEXT, linewidth=1.2, linestyle="--", label="sin(x)")
    ax.scatter(knots, samples, color=GOLD, zorder=3, s=30, label="Sampled knots")
    ax.set(xlabel="x (radians)", ylabel="Function value")
    ax.legend(facecolor=NAVY, edgecolor="#334d66", labelcolor=TEXT, loc="upper right")
    fig.savefig(output / "interpolation.png", dpi=160)
    plt.close(fig)
    print(f"Broyden residual {result.residual_norms[-1]:.3e} after {result.accepted_steps} accepted steps")
    print(f"Gauss–Seidel residual {linear[-1]:.3e} after {budgets[-1]} sweeps")
    print(f"Spline maximum sampled error {np.max(np.abs(interpolated - np.sin(domain))):.3e}")


if __name__ == "__main__":
    main()
