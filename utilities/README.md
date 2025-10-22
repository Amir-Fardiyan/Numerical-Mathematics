# utilities — A tiny grab‑bag of numerical helpers

*A compact, classroom‑style set of static methods wrapped in a single `utilities` class. Clean, dependency‑light, and easy to drop into small projects or coursework.*

---

## Why this exists
When you’re prototyping or solving homework‑sized problems, you don’t always want to pull in big libraries or wire up a full package. This repo gives you a single `utilities` class containing a few classic algorithms:

* Bubble sort for 1D lists (`sort1D`) and paired 2D lists (`sort2D`)
* Gauss–Seidel linear solver (`gauss_sidel`)
* Natural cubic spline coefficient builder (`cubic_spline`) and evaluator (`cubic_spline_interpolate`)
* Bairstow’s method for polynomial roots (`bairstow_EoS`)

They’re intentionally simple and readable (great for learning), and shipped as `@staticmethod`s so you can call them without instantiating the class.

---


## Usage
### 1) Sorting

#### `utilities.sort1D(u)`

Bubble‑sorts a single list **in place** and returns it.

```python
nums = [5, 2, 9, 1]
utilities.sort1D(nums)           # -> [1, 2, 5, 9]  (nums is also mutated)
```

#### `utilities.sort2D(u, v)`

Bubble‑sorts list `u` ascending and **reorders `v` in lock‑step** (think x/y pairs).

```python
x = [3, 1, 2]
y = [30, 10, 20]
ux, uy = utilities.sort2D(x, y)  # -> [1, 2, 3], [10, 20, 30]
```

> **Complexity:** Both sorts are ~O(n²). Good for small lists; prefer Python’s built‑in `sorted()` for performance.

---

### 2) Linear systems — Gauss–Seidel

#### `utilities.gauss_sidel(A, b, max_it, max_err)`

Iterative solver for `Ax = b`.

```python
A = [[4, 1],
     [2, 3]]
b = [1, 2]
x = utilities.gauss_sidel(A, b, max_it=1000, max_err=1e-6)
print(x)  # e.g. [0.0909..., 0.6363...]
```

**Notes**

* Starts from the zero vector; stops when all component‑wise updates are `< max_err` or `max_it` is hit.
* Requires non‑singular `A`. Convergence typically needs diagonally dominant or symmetric positive‑definite matrices.

---

### 3) Interpolation — Natural cubic spline

#### `utilities.cubic_spline(x, y)` → `(a, b, c, d)`

Builds natural spline coefficients for consecutive intervals `[x[i], x[i+1]]`. Internally it sorts `x, y` by `x`.

```python
x = [0.0, 1.0, 2.0, 3.0]
y = [0.0, 0.5, 2.0, 1.5]
a, b, c, d = utilities.cubic_spline(x, y)
```

#### `utilities.cubic_spline_interpolate(x, a, b, c, d, x_val)`

Evaluates the spline and its first two derivatives at `x_val`.

```python
val, d1, d2 = utilities.cubic_spline_interpolate(x, a, b, c, d, 1.6)
print(val, d1, d2)
```

**Notes**

* Natural boundary conditions (`M[0]=M[-1]=0`).
* No extrapolation guard: calling outside `[x[0], x[-1]]` returns `None`.

---

### 4) Polynomial roots — Bairstow

#### `utilities.bairstow_EoS(coefficients, max_it, err)`

Finds (possibly complex) roots of a polynomial using Bairstow’s method.

```python
# 2x^3 - 4x^2 - 22x + 24 = 0
coeffs = [2, -4, -22, 24]  # highest degree -> constant
roots = utilities.bairstow_EoS(coeffs, max_it=1000, err=1e-6)
print(roots)  # numpy array of roots
```

**Important arguments & behavior**

* Pass coefficients **from highest degree to constant**. Internally they’re reversed to match the original implementation.
* Initial quadratic guess uses `r = s = -1`.
* Returns a NumPy array of roots. Install NumPy if you plan to use this function.

---

## API reference

| Method       | Signature                                        | Mutates inputs  | Returns                |
| ------------ | ------------------------------------------------ | --------------- | ---------------------- |
| Sort 1D      | `sort1D(u)`                                      | ✅               | sorted list `u`        |
| Sort 2D      | `sort2D(u, v)`                                   | ✅               | `(u, v)` sorted by `u` |
| Gauss–Seidel | `gauss_sidel(A, b, max_it, max_err)`             | ❌               | list `x`               |
| Cubic spline | `cubic_spline(x, y)`                             | ✅ (sorts `x,y`) | `(a, b, c, d)`         |
| Spline eval  | `cubic_spline_interpolate(x, a, b, c, d, x_val)` | ❌               | `(value, d1, d2)`      |
| Bairstow     | `bairstow_EoS(coeffs, max_it, err)`              | ❌               | `np.array` of roots    |

> **Mutation gotcha:** `sort1D`, `sort2D`, and `cubic_spline` modify the lists you pass in. If you need the originals, pass copies.

---

## Limitations & tips

* **Performance:** Sorts are O(n²). Use Python’s `sorted()` / `list.sort()` for large lists.
* **Stability:** Gauss–Seidel convergence isn’t guaranteed for arbitrary `A`. Consider pre‑checking diagonal dominance.
* **Input safety:** No explicit input validation. For production, add shape checks and better error messages.
* **Numerical sensitivity:** Bairstow can be sensitive to initial guesses and scaling.

---
