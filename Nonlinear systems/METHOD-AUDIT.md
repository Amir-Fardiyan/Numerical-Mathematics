# Mathematical audit of the Broyden trust-region solver

The original implementation is Amir Fardiyan’s [source at revision 566271f](https://github.com/Amir-Fardiyan/Numerical-Mathematics/blob/566271fe4a9f8d1863c373ffa2e5702cbd40c6f4/Nonlinear%20systems/Broyden-TrustRegion.py). Its [original accompanying README](https://github.com/Amir-Fardiyan/Numerical-Mathematics/blob/566271fe4a9f8d1863c373ffa2e5702cbd40c6f4/Nonlinear%20systems/README.md) corroborates the original reduction formula and accepted-step radius reset. The comparison uses the review-branch [solver](../numerical_math/nonlinear.py), [legacy entry point](Broyden-TrustRegion.py), and [transport demo](../examples/transport.py). The prior redesign documentation largely described the review-branch equations, but omitted their provenance and how significantly they diverged from the original.

**Finding.** The redesign changes the acceptance model, radius policy, direction at numerical rank deficiency, and transport initialization. The two solvers can generate different iterates even for smooth scalar systems with exact Jacobians. The original radius reset and acceptance denominator are part of the historical method and require an explicit decision before any replacement. In particular, keeping the original function signature preserves caller compatibility while the current wrapper runs the revised algorithm.

## Comparison

**Refactoring** covers software organization and equivalent formulas under stated assumptions. **Numerical safeguard** covers finite-precision handling, failure detection, or termination. **Method change** covers rules that can alter the mathematical iteration. Some changes serve more than one purpose.

| Component | Original implementation | Review-branch implementation | Mathematical difference | Justification and classification |
| --- | --- | --- | --- | --- |
| Nonlinear problem | Finds a root of a square real residual system. The method treats `grad` as the initial Jacobian approximation. | Same target $F(x)=0$ and square Jacobian. Converts arrays to float, checks shapes and initial finiteness. | Same root equation for valid real inputs. Input conversion and validation affect which inputs reach it. | **Refactoring and safeguards.** Support real array-like inputs and catch malformed values early. |
| Initial Jacobian | Calls `grad(x0)` before evaluating the stopping test and retains the result for in-place updates. | Evaluates the initial residual first, returns early for a converged start or zero budget, and otherwise copies the callback result. The new API can generate the matrix when no callback is supplied. | Identical initial matrix for a pure callback returning the same matrix. Callback timing, caller-visible mutation, and the optional default differ. | **Refactoring and safeguards.** Avoid unnecessary Jacobian evaluations and prevent the solver from overwriting a matrix owned by the caller. |
| Finite differences | Constant coordinate offsets $`\pm\delta_{\mathrm{FD}}`$ and denominator $`2\delta_{\mathrm{FD}}`$. Default $10^{-5}$. Input copies have the original dtype. | Offset magnitude $`\delta_{\mathrm{FD}}\max(1,\vert x_j\vert)`$, with the realized floating-point span as denominator. Copies use float. Checks the span and residuals. | The nominal offset differs when $`\vert x_j\vert>1`$. The realized denominator can differ through rounding at any scale. Integer perturbations are retained. | **Method change and safeguards.** Relative scaling is a different Jacobian approximation. Float conversion, the realized span, and copying callback output address integer truncation, rounding, and reused residual buffers. |
| Raw direction | Unpivoted dense LU followed by forward and backward substitution solves $Bd=-f$ when those operations succeed. | Dense minimum-norm least squares minimizes $`\Vert f+Bd\Vert _2`$ with NumPy’s default numerical rank cutoff. | Same exact direction for nonsingular $B$ when the original LU is defined. A numerically rank-deficient model can give a different, finite direction. | **Method change and safeguard.** Least squares avoids unpivoted zero pivots and defines a rank-deficient direction. Pivoted square-system solves would address zero pivots for nonsingular matrices while keeping the original equation. |
| Trust region | Radially clips the unconstrained direction to $\Delta$. Uses the nominal clipped vector in the formulas. | Uses the same clipping rule but computes the realized step $s=(x+p)-x$ in floating-point arithmetic. | Identical scaling for identical $d$ and $\Delta$ in exact arithmetic. The actual directions and radii can differ. Radial clipping generally differs from solving the ball-constrained least-squares problem. | **Refactoring and safeguard.** Preserve radial clipping and use the displacement actually taken in the secant relation. |
| Actual reduction | $`\Vert f\Vert _2^2-\Vert F(x+p)\Vert _2^2`$, evaluated by repeatedly calling the residual function. | Same difference for the same trial point and deterministic residual, evaluated as $(f-t)^\top(f+t)$ using cached values. | Algebraically identical for a fixed trial. Evaluation order and floating-point rounding differ. Stateful callbacks may behave differently. | **Refactoring and finite-precision safeguard.** Avoid repeated callbacks and direct subtraction of two accumulated squared norms. |
| Predicted reduction | $`P_{\mathrm{orig}}=-f^\top p-\frac{1}{2} p^\top Bp`$. | $`P_{\mathrm{rev}}=\Vert f\Vert _2^2-\Vert f+Bs\Vert _2^2`$. | These are different local models. They generally disagree even for symmetric or scalar $B$. | **Method change.** The revised denominator measures predicted reduction in the same squared-residual merit function as the numerator. This is an alternate model and needs approval. |
| Acceptance | Computes $`r=A/P_{\mathrm{orig}}`$ for any nonzero denominator, using $r=-1$ at zero. Accepts when $r\geq\rho$. | Rejects nonfinite trial residuals, nonpositive predictions, and nonfinite ratios. Accepts the other trials when $r\geq\rho$. | The original can accept negative actual and predicted reductions. In exact arithmetic the revised policy with $`\rho>0`$ requires a decrease in squared residual norm. | **Method change and safeguards.** The positive-denominator gate changes valid finite decisions. Nonfinite checks prevent accepting invalid steps. |
| Initial radius | $`\Delta_0=c\Vert F(x_0)\Vert _2`$. | $`\Delta_0=c\max(1,\Vert x_0\Vert _2)`$. | A residual-scaled radius is replaced by a state-scaled radius with a unit floor. It responds differently to rescaling $F$. | **Method change.** The revised rule avoids choosing a very small initial radius solely from a small residual. It removes the original residual-dependent initialization. |
| Rejected radius | $\Delta\gets c\Delta$. The point, Jacobian, and accepted count are unchanged. | Same contraction and unchanged point, residual, and Jacobian. The entry consumes trial budget. | Same retry rule up to termination and handling of invalid trials. | **Preserved rule plus safeguard.** Bound the number of retries. |
| Accepted radius | Always resets $`\Delta\gets c\Vert F(x^+)\Vert _2`$. | Shrinks when $`r<0.25`$, expands by $1/c$ when $`r>0.75`$ and the unscaled direction length is at least $0.9\Delta$, and otherwise retains the radius. | The original has no ratio-based radius policy after acceptance. The revised rule changes future trial steps and introduces three thresholds. Equality at $0.25$ or $0.75$ retains the radius. | **Method change.** The ratio is used as a model-agreement signal in the revised variant. The constants are design choices introduced by the refactor, absent from the historical method. |
| Broyden update | On acceptance sets $y=F(x^+)-f$ and adds $(y-Bp)p^\top/(p^\top p)$. | Same rank-one formula with realized displacement $s$, evaluated as an outer product of two vectors each divided by $`\Vert s\Vert _2`$. | Same secant equation in exact arithmetic for the same nonzero displacement and residual change. Iterates and matrices can differ because other rules select different steps. | **Refactoring and finite-precision safeguard.** Keep the rank-one formula and represent the displacement as taken. |
| Rejected iterations and budget | Increments `iter_count` only after acceptance. Repeated rejected trials can run indefinitely. | Each trial loop entry counts toward `max_iter`, while `accepted_steps` counts accepted trials only. | A bounded run may stop before an acceptance that the historic loop would keep seeking. | **Termination bug fix with a contract change.** Bounding rejections prevents a hang. The legacy wrapper now interprets its cap differently. |
| Convergence | Tests $`\Vert F(x)\Vert _2\leq\epsilon`$ at the top of the loop, only while the accepted budget remains. A last accepted step can return a root without printing convergence. | Tests the same norm at initialization and after each accepted trial. Returns status and the residual history, including the initial norm. | Same residual criterion. Reporting and when the check runs differ, including at zero budget and after the final allowed step. | **Diagnostic refactoring and safeguard.** Check every returned accepted iterate against the stated tolerance. |
| Stagnation | No explicit zero-displacement test, minimum radius, or rejection limit. A zero step normally yields zero prediction and is retried indefinitely for the default acceptance threshold. | Stops when the candidate equals the current point componentwise, before evaluating a trial residual. The loop entry is counted. | Can stop at a non-root with a zero least-squares direction or when rounded addition yields no movement. There is no Jacobian refresh or independent small-step tolerance. | **Termination safeguard.** Report no representable progress explicitly. |
| Singularities and invalid arithmetic | Unpivoted LU can divide by zero for a singular matrix or an invertible matrix with a zero pivot. There is no explicit finite-value gate or recovery. | Least squares uses a numerically rank-truncated model when necessary. Initial invalid data raise errors and invalid trial residuals are rejected. A factorization failure can still raise. | Adds a defined direction for some singular models. A finite direction does not imply existence of a root or guaranteed convergence. | **Method change and safeguards.** Improve behavior for bad pivots and singular models while reporting nonconvergence separately. |
| Transport residual | The original implicit upwind formula, with $`S_0=1`$ and the flux argument clipped at $10^{-50}$ and $1-10^{-50}$. The upper bound rounds to $1$ in double precision. | Same upwind formula, vectorized, with the argument clipped at $0$ and $1$, plus float conversion and shape checks. | Same interior model above the old lower bound and below $1$. The extended flux differs near and below zero. Casting changes behavior on integer input. | **Refactoring with a small model change.** Vectorization preserves the stencil. Exact physical endpoints replace the old lower floor. |
| Transport initialization | Uses the fixed centered-difference helper at the default initial vector $`S_i=10^{-8}`$. | Starts from the same vector and supplies the analytic lower bidiagonal Jacobian for the clipped-flux model. | The original finite-difference stencil straddles the lower clipping boundary at the starting point. The analytic derivative is local to the initial point, so the two initial matrices need not agree. | **Method change in the demonstration.** The analytic Jacobian exploits the stencil and avoids finite-difference error. It does not reproduce the original initialization. |
| Packaging and output | A single script runs the demonstration, plots a heatmap, prints convergence when detected and returns the iterate and accepted iteration count. | Importable package, demo isolated from import, saved figures supported, result object and compatibility tuple wrapper. Legacy LU helpers remain callable and now raise on a zero pivot. | No new equation arises from packaging or plotting. The compatibility wrapper uses the revised iteration, and the helpers expose a different error path. | **Refactoring and safeguard.** Separate computation from display and report errors instead of returning nonfinite factors. |

## Parameters

| Parameter | Default | Original meaning | Review-branch meaning |
| --- | --- | --- | --- |
| `func` | Required | Residual callback. | Same mathematical role, with explicit shape and finiteness rules. |
| `grad` | Required in the historical API and compatibility wrapper | Initial Jacobian callback, always evaluated before the loop. | Optional in `solve_nonlinear`, used only if the solver takes trials. The wrapper retains a required argument. |
| `x0` | Required | Starting point and base point for the Jacobian. | Same role, copied to a validated floating-point vector. |
| `epsilon` | `1e-12` | Absolute Euclidean residual tolerance. | Same tolerance. Must be finite and positive. |
| `rho` | `0.1` | Acceptance threshold against the original quadratic predictor. | Acceptance threshold against the residual-model predictor, with $`0<\rho<1`$. |
| `c` | `0.5` | Multiplier for initial and accepted radius resets and for rejection contraction. | Initial state-scale multiplier, contraction multiplier, and reciprocal for expansion, with $`0<c<1`$. |
| `max_iter` | `100` | Caps accepted iterations. | Caps trial loop entries. A nonnegative integer is required for the loop when it is entered. |
| `delta` in the Jacobian helper | `1e-5` | Absolute centered-difference perturbation in every coordinate. | Base scale for coordinate-dependent perturbations. Must be finite and positive. |
| Transport constants | $N=500$, $\Delta t=0.01$, $e=0.1167$, $M=1.1177$ | Cell count, time step, flux exponent offset, and flux parameter. The script sets $M=1+0.1177$. | Same defaults. The CLI can vary $N$. The original transport variable named `delta` was $M-1$, independent of the radius and finite-difference step. |

## Why the reduction models differ

Write $f=F(x)$ and hold the trial displacement fixed at $p$. The original denominator is the predicted decrease from zero of the scalar quadratic $q(p)=f^\top p+\frac{1}{2} p^\top Bp$. Its quadratic term depends only on the symmetric part of $B$, even though the direction is computed with all of $B$. The revised denominator expands to

```math
P_{\mathrm{rev}}=-2f^\top Bp-p^\top B^\top Bp.
```

This follows by applying the squared-residual merit function to the affine residual model. The original code and its README both specify $`P_{\mathrm{orig}}`$, so the text and the source agree on that historical choice. Changing it to $`P_{\mathrm{rev}}`$ is a change to the globalization strategy.

The predictors also respond differently to residual scaling. For a fixed step $p$, multiplying both $f$ and $B$ by a positive scalar $\alpha$ multiplies $A$ by $\alpha^2$, $`P_{\mathrm{orig}}`$ by $\alpha$, and $`P_{\mathrm{rev}}`$ by $\alpha^2$. Where the ratios are defined, the historical ratio changes by a factor of $\alpha$ while the revised ratio is unchanged for that fixed step. This comparison holds the step fixed. The original radius policy can also change the step when the residual is rescaled.

### Radial clipping and the constrained model

For $B=\mathrm{diag}(1,2)$, $f=(-1,-2)^\top$, and $\Delta=1$, the unconstrained direction is $(1,1)^\top$. Radial clipping gives $(1/\sqrt{2},1/\sqrt{2})^\top$ and a squared model residual of $5(1-1/\sqrt{2})^2$, approximately $0.429$. The feasible boundary step $(0.6,0.8)^\top$ gives $0.32$. This is a counterexample to interpreting either implementation as an exact solver of the least-squares trust-region subproblem.

## Historical algorithm in pseudocode

```text
x ← x0
B ← grad(x0)
radius ← c × norm2(F(x))
accepted ← 0
While accepted < max_iter:
    f ← F(x)
    If norm2(f) ≤ epsilon:
        print convergence
        exit loop
    d ← solve B d = -f by unpivoted LU
    p ← d if norm2(d) ≤ radius, otherwise radius × d / norm2(d)
    A ← norm2(F(x))² - norm2(F(x + p))²
    P ← -dot(f, p) - 0.5 × dot(p, B p)
    r ← A / P if P ≠ 0, otherwise -1
    If r ≥ rho:
        x ← x + p
        y ← F(x) - f
        B ← B + outer(y - B p, p) / dot(p, p)
        radius ← c × norm2(F(x))
        accepted ← accepted + 1
    Else:
        radius ← c × radius
Return x, accepted.
```

The pseudocode assumes deterministic residual evaluations and combines repeated calls for readability. The [review-branch pseudocode](README.md#complete-pseudocode) describes the separate implementation that the draft now runs.

## Reproduced differences

These checks used the original solver functions loaded directly from Git history and the current solver. No source files were changed to perform them.

1. **Different trajectory with an exact Jacobian.** For $F(x)=x-1$ from $`x_0=0`$, both first steps are $p=0.5$. At that trial $A=0.75$, $`P_{\mathrm{orig}}=0.375`$, and $`P_{\mathrm{rev}}=0.75`$. Both accept. The original resets the radius to $0.25$ while the revised rule expands it to $1$. With defaults the original took 40 accepted steps and the revised solver took 2.
2. **Opposite acceptance decisions.** For $F(x)=1-2x+100x^2$ from $`x_0=0`$ with $`B_0=-2`$, both first trial points are $0.5$ and $F(0.5)=25$. The original computes $A=-624$, $`P_{\mathrm{orig}}=-0.25`$, and $r=2496$, so it accepts. The revised solver computes $`P_{\mathrm{rev}}=1`$ and rejects. This example establishes the effect of the model and acceptance change without assuming what the original author intended.
3. **Invertible zero-pivot matrix.** For the matrix with off-diagonal entries $1$ and diagonal entries $0$, the historical LU-based direction returned nonfinite values. The matrix is invertible, so this failure is due to unpivoted elimination. The audit distinguishes that numerical bug from the separate decision to define a least-squares direction for rank-deficient systems.

The circle–line example also exposes a historical retry case in exact arithmetic. At $(0.8,0.3)$ the exact Jacobian gives $f=(-0.27,0.5)^\top$ and $d=(-3/220,107/220)^\top$. The initial radius gives a clipped step $p=\lambda d$ with $`0<\lambda<0.59`$. With $Bd=-f$, the original predictor is $`-\lambda(1-\lambda/2)f^\top d<0`$. The second residual component decreases from $0.5$ to $0.5(1-\lambda)$. The first moves toward zero by $0.27\lambda+\lambda^2 d^\top d$, which is positive and less than $0.27$ on this interval. Hence $`A>0`$ and the original rule rejects every such trial, including the smaller trials obtained by halving the radius. This explains why success of the plotted circle–line example under the revised method is evidence about that variant only.

## Review boundary

The original source remains available at its historical revision. This audit records the review-branch variant as implemented so its convergence figures and tests can be interpreted correctly. The denominator, radius policy, rank-deficient direction, and transport Jacobian are explicit algorithm decisions for the author’s review.
