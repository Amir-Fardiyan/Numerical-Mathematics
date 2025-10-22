"""
Utility algorithms
"""

from typing import List, Tuple
import numpy as np


class utilities:
    @staticmethod
    def sort1D(u):
        n = len(u)
        for i in range(n):
            for j in range(0, n - i - 1):
                if u[j] > u[j + 1]:
                    u[j], u[j + 1] = u[j + 1], u[j]
        return u

    @staticmethod
    def sort2D(u, v):
        n = len(u)
        for i in range(n):
            for j in range(0, n - i - 1):
                if u[j] > u[j + 1]:
                    u[j], u[j + 1] = u[j + 1], u[j]
                    v[j], v[j + 1] = v[j + 1], v[j]
        return u, v

    @staticmethod
    def gauss_sidel(A, b, max_it, max_err):
        n = len(A)
        x = [0] * n
        y = [0] * max_it
        err = [0] * n
        for m in range(max_it):
            y[m] = [0] * n

        for k in range(0, max_it):
            # Update x using Gauss-Seidel method
            for i in range(0, n):
                Sum = []
                y[k][i] = x[i]
                for j in range(0, n):
                    if j != i:
                        s = A[i][j] * x[j]
                        Sum.append(s)
                x[i] = (1 / A[i][i]) * (b[i] - sum(Sum))
                err[i] = abs(x[i] - y[k][i])

            # Check for convergence
            if all(abs(x[i] - y[k][i]) < max_err for i in range(n)):
                break
        return x

    @staticmethod
    def cubic_spline(x, y):
        utilities.sort2D(x, y)  # Sort the input arrays
        n = len(x)
        h = [0 for col in range(n - 1)]
        for i in range(0, n - 1):
            h[i] = x[i + 1] - x[i]
        A = [[0 for col in range(n)] for row in range(n)]
        B = [0 for col in range(n)]
        A[0][0] = 1
        A[n - 1][n - 1] = 1
        for i in range(1, n - 1):
            A[i][i - 1] = h[i - 1]
            A[i][i] = 2 * (h[i - 1] + h[i])
            A[i][i + 1] = h[i]
            B[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

        # Solve the linear system to find second derivatives
        M = utilities.gauss_sidel(A, B, 1000, 1e-5)

        # Compute spline coefficients
        d_coef = [y[i] for i in range(0, n - 1)]
        c_coef = [(y[i + 1] - y[i]) / h[i] - (h[i] * (2 * M[i] + M[i + 1]) / 6) for i in range(0, n - 1)]
        b_coef = [M[i] / 2 for i in range(0, n - 1)]
        a_coef = [(M[i + 1] - M[i]) / (6 * h[i]) for i in range(0, n - 1)]

        return a_coef, b_coef, c_coef, d_coef

    @staticmethod
    def cubic_spline_interpolate(x, a, b, c, d, x_val):
        for i in range(len(x) - 1):
            if x[i] <= x_val and x_val <= x[i + 1]:
                h = x_val - x[i]
                # Use the cubic polynomial for the interval to interpolate
                interpolated_value = d[i] + c[i] * h + b[i] * h ** 2 + a[i] * h ** 3
                # Compute the derivative
                derivative_value = c[i] + 2 * b[i] * h + 3 * a[i] * h ** 2
                # Compute the second derivative
                second_derivative_value = 2 * b[i] + 6 * a[i] * h
                return interpolated_value, derivative_value, second_derivative_value

    @staticmethod
    def bairstow(coefficients, max_it, err):  # enter coefficients greater to smaller
        a = list(coefficients)
        a.reverse()  # use reverse if coefficients are smaller to greater
        roots = []
        r = -1
        s = -1
        n = len(a) - 1
        b = [0] * (n + 1)
        c = [0] * (n + 1)

        # check the degree of polynomial and solve if it's degree is less than 3:
        for _ in range(n // 2 + 1):
            if n > 2:
                for _ in range(max_it):
                    b[n] = a[n]
                    b[n - 1] = a[n - 1] + r * b[n]
                    for k in range(n - 2, -1, -1):
                        b[k] = a[k] + r * b[k + 1] + s * b[k + 2]
                    c[n] = b[n]
                    c[n - 1] = b[n - 1] + r * c[n]
                    for i in range(n - 2, 0, -1):
                        c[i] = b[i] + r * c[i + 1] + s * c[i + 2]
                    delta_r = float((c[3] * b[0] - c[2] * b[1])) / float(((c[2]) ** 2 - c[3] * c[1]))
                    delta_s = float((c[1] * b[1] - c[2] * b[0])) / float(((c[2]) ** 2 - c[3] * c[1]))
                    r += delta_r
                    s += delta_s
                    # print(c)
                    if abs(b[1]) < err and abs(b[0]) < err:
                        d = r ** 2 + 4 * s
                        if d >= 0:
                            x1 = (r + d ** 0.5) / 2
                            x2 = (r - d ** 0.5) / 2
                        else:
                            x1 = complex(r, abs(d) ** 0.5) / 2
                            x2 = complex(r, abs(d) ** 0.5) / 2
                        roots.append(x1)
                        roots.append(x2)

                        a = [0] * (n - 1)
                        for l in range(n, 1, -1):
                            a[l - 2] = b[l]
                        n = n - 2
                        break
                    # m += 1
            elif n == 2:  # solve quad. eq.
                d = a[1] ** 2 - 4 * a[2] * a[0]
                if d >= 0:
                    x1 = (-a[1] + d ** 0.5) / (2 * a[2])
                    x2 = (-a[1] - d ** 0.5) / (2 * a[2])
                else:
                    x1 = complex(-a[1], np.sqrt(abs(d)) / (2 * a[2]))
                    x2 = complex(-a[1], -np.sqrt(abs(d)) / (2 * a[2]))
                roots.append(x1)
                roots.append(x2)
                n -= 2
            elif n == 1:
                if a[1] == 0:
                    x = 0
                else:
                    x = -a[0] / a[1]
                roots.append(x)
            else:
                break
            # print('roots =', roots)

            rootss = np.array(roots)
        return rootss

