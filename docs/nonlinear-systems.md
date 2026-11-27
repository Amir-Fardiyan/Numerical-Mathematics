# Broyden method with a residual-based trust-region radius

## Nonlinear system and Jacobian approximation

Given a square residual function $F$, the problem is to find $`x_\star`$ satisfying

```math
F:\mathbb{R}^n\to\mathbb{R}^n,\qquad F(x_\star)=0.
```

Let $k$ count accepted steps, $`f_k=F(x_k)`$, and $`B_k`$ denote the Jacobian approximation at $`x_k`$. The callback `grad` supplies $`B_0`$ before the first stopping test. When the centered-difference helper is used, column $j$ is computed with the coordinate vector $`e_j`$ and a fixed offset $h$ (default $10^{-5}$).

```math
(B_0)_{:,j}=\frac{F(x_0+h e_j)-F(x_0-h e_j)}{2h}.
```

For smooth $F$, subtracting the Taylor expansions at offsets $`\pm h e_j`$ cancels the even-order terms. The leading term is $2h$ times the derivative in direction $`e_j`$, giving the denominator $2h$ above. The initial radius is

```math
\Delta_0=c\Vert F(x_0)\Vert_2.
```

## Search direction and trial point

The affine residual model at $`x_k`$ is $`f_k+B_k d`$. Setting this model to zero gives $`B_k d=-f_k`$. The implementation factors $`B_k=L_k U_k`$ by dense LU elimination without pivoting. At pivot $j$, it stores $`L_{ij}=U_{ij}/U_{jj}`$ for $`i>j`$, then replaces row $i$ of $U$ with that row minus $`L_{ij}`$ times row $j$. The lower factor has unit diagonal. With $`b=-f_k`$, forward and backward substitution give, respectively,

```math
w_i=b_i-\sum_{j<i}L_{ij}w_j,\qquad
d_i=\frac{w_i-\sum_{j>i}U_{ij}d_j}{U_{ii}}.
```

If the direction fits inside the Euclidean ball of radius $`\Delta`$, the trial displacement is $p=d$. Otherwise the code scales $d$ by $`\Delta/\Vert d\Vert_2`$. Equivalently, for nonzero $d$,

```math
p=d\min\!\left(1,\frac{\Delta}{\Vert d\Vert_2}\right),
\qquad z=x_k+p.
```

A zero direction follows the unscaled branch.

## Reduction, acceptance, and radius adjustment

The actual reduction $A$ is the decrease in the squared Euclidean residual. With $`v=F(z)-f_k`$, expanding the square gives

```math
A=\Vert F(x_k)\Vert_2^2-\Vert F(z)\Vert_2^2
  =-2 f_k^\top v-v^\top v .
```

The implementation computes the difference of the two residual inner products. The predicted reduction $P$ is the decrease from zero displacement in the scalar quadratic $`q_k`$ specified by $`f_k`$ and $`B_k`$.

```math
q_k(p)=f_k^\top p+\frac{1}{2}p^\top B_k p,
\qquad
P=q_k(0)-q_k(p)=-f_k^\top p-\frac{1}{2}p^\top B_k p.
```

For nonzero $P$, the ratio is $r=A/P$. For $P=0$, the code sets $r=-1$. The trial is accepted when $`r\geq\rho`$ and rejected otherwise. A negative nonzero prediction enters the same ratio test. Acceptance is therefore determined by the ratio and need not imply a decrease in the residual norm.

On rejection, $`x_k`$ and $`B_k`$ are retained and the trial is repeated with radius $`c\Delta`$. On acceptance the point becomes $`x_{k+1}=z`$, the Jacobian approximation is updated, and the radius is reset to

```math
\Delta_{k+1}=c\Vert F(x_{k+1})\Vert_2.
```

Only acceptance increments the iteration counter.

## Broyden update and stopping test

For an accepted trial, set $`s_k=p`$ and $`y_k=F(x_{k+1})-f_k`$. The secant condition is $`B_{k+1}s_k=y_k`$. Writing the correction as $`u s_k^\top`$ gives $`u(s_k^\top s_k)=y_k-B_k s_k`$. For nonzero $`s_k`$, substitution gives

```math
B_{k+1}=B_k+\frac{(y_k-B_ks_k)s_k^\top}{s_k^\top s_k}.
```

At the top of each loop entry, the code tests $`\Vert F(x_k)\Vert_2\leq\epsilon`$. The loop is entered only while the accepted-step count is below `max_iter`. If the last allowed acceptance reaches the tolerance, the loop exits by the count without another residual test. The function returns the final iterate and the accepted-step count.

Repeated rejections can continue indefinitely because the accepted-step count stays fixed. A zero pivot leads to division by zero in elimination or back substitution.

## Algorithm

```text
x ← x0
B ← grad(x0)
radius ← c × norm2(F(x))
accepted ← 0

While accepted < max_iter:
    f ← F(x)
    If norm2(f) ≤ epsilon:
        print "Converged!"
        break

    L, U ← unpivoted LU factorization of B
    w ← forward substitution in L w = -f
    d ← backward substitution in U d = w
    p ← d if norm2(d) ≤ radius, otherwise radius × d / norm2(d)
    A ← dot(F(x), F(x)) - dot(F(x + p), F(x + p))
    P ← dot(-f, p) - dot((0.5 × pᵀ) B, p)
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

| Parameter | Default | Meaning |
| --- | --- | --- |
| `func` | Required | Residual callback $F$, returning a NumPy vector. |
| `grad` | Required | Callback providing the initial Jacobian approximation $`B_0`$. |
| `x0` | Required | Initial NumPy vector $`x_0`$. |
| `epsilon` | `1e-12` | Absolute tolerance on the Euclidean residual norm. |
| `rho` | `0.1` | Acceptance threshold for the actual-to-predicted reduction ratio. |
| `c` | `0.5` | Multiplier for the initial and accepted-step radii and for radius contraction on rejection. |
| `max_iter` | `100` | Maximum number of accepted steps. |
| `delta` in `calculate_jacobian` | `1e-5` | Fixed centered-difference offset $h$. |

## Numerical examples

For $F(x)=x-1$ with $`x_0=0`$ and $`B_0=1`$, the initial direction is $d=1$ and the initial radius is $0.5$. The first trial is $p=0.5$, with $A=0.75$, $P=0.375$, and ratio $r=2$. It is accepted, and the radius resets to $0.25$. In exact arithmetic, the accepted iterates satisfy $`x_k=1-2^{-k}`$ for these parameters. With the default tolerance, the implementation returned after 40 accepted steps with residual norm approximately $`9.09\times10^{-13}`$.

For $F(x)=x^2-2$ with $`x_0=1`$, the following example initializes the Jacobian by centered differences. It returned $`x\approx1.414213562373095`$ in six accepted steps with residual norm approximately $`8.88\times10^{-16}`$.

```python
import numpy as np

from numerical_math import broyden_trust_region, calculate_jacobian


def residual(x):
    return x**2 - 2


def jacobian(x):
    return calculate_jacobian(residual, x)


x, accepted = broyden_trust_region(residual, jacobian, np.array([1.0]))
print(x, accepted, np.linalg.norm(residual(x)))
```

[Convergence figure](figures/convergence.png)

### Implicit transport

The [transport script](../Nonlinear%20systems/Broyden-TrustRegion.py) solves for cell values $`S_i`$, indexed from $1$ to $N$ with inlet $`S_0=1`$. Its clipped flux and residual are

```math
q=2+e,\qquad v(s)=\min\!\left(1-10^{-50},\max(10^{-50},s)\right),
\qquad \phi(s)=\frac{v(s)^q}{v(s)^q+M(1-v(s))^q}.
```

```math
R_i(S)=S_i+N\Delta t\,[\phi(S_i)-\phi(S_{i-1})],
\qquad i=1,\ldots,N.
```

The calculation uses $N=500$, $`\Delta t=0.01`$, $e=0.1167$, and $M=1+0.1177$. It starts from $`S_i=10^{-8}`$ and initializes the Jacobian with the centered-difference helper. The upper clipping endpoint evaluates to $1$ in double precision. For this grid, 43 steps were accepted and the final residual norm was $`\Vert R(S)\Vert_2\approx 4.22\times 10^{-13}`$.

[Transport heatmap](figures/transport.png)
