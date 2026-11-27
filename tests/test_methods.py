"""Analytic solutions, convergence checks, and regressions for the public API."""

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from numerical_math import (
    bairstow,
    broyden_trust_region,
    calculate_jacobian,
    cubic_spline,
    cubic_spline_interpolate,
    gauss_sidel,
    sort1d,
    sort_pairs,
)
from numerical_math.nonlinear import lu_decomposition, solve_lu
from utilities.Utilities import utilities


class NonlinearTests(unittest.TestCase):
    def test_solver_definitions_match_the_transport_script(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "Nonlinear systems" / "Broyden-TrustRegion.py"
        module = root / "numerical_math" / "nonlinear.py"

        def definitions(path):
            source = path.read_text(encoding="utf-8")
            return {
                node.name: ast.get_source_segment(source, node)
                for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef)
            }

        script_functions = definitions(script)
        module_functions = definitions(module)
        self.assertEqual(
            set(module_functions),
            {"calculate_norm", "lu_decomposition", "solve_lu", "broyden_trust_region", "calculate_jacobian"},
        )
        self.assertEqual(module_functions, {name: script_functions[name] for name in module_functions})

    def test_fixed_centered_difference(self):
        samples = []

        def residual(x):
            samples.append(x.copy())
            return np.array([x[0] * x[1], x[0] ** 2 - x[1]])

        point = np.array([2.0, 3.0])
        jacobian = calculate_jacobian(residual, point)
        assert_allclose(jacobian, [[3, 2], [4, -1]], atol=1e-9)
        assert_allclose(
            np.array(samples) - point,
            [[1e-5, 0], [-1e-5, 0], [0, 1e-5], [0, -1e-5]],
            rtol=0, atol=1e-15,
        )

    def test_linear_radius_reset_and_residual_tolerance(self):
        def residual(x):
            return x - 1

        def jacobian(x):
            return np.eye(1)

        start = np.array([0.0])
        first, accepted = broyden_trust_region(residual, jacobian, start, max_iter=1)
        assert_allclose(first, [0.5])
        self.assertEqual(accepted, 1)
        second, accepted = broyden_trust_region(residual, jacobian, start, max_iter=2)
        assert_allclose(second, [0.75])
        self.assertEqual(accepted, 2)
        solution, accepted = broyden_trust_region(residual, jacobian, start)
        self.assertEqual(accepted, 40)
        self.assertLessEqual(np.linalg.norm(residual(solution)), 1e-12)

    def test_quadratic_with_finite_difference_jacobian(self):
        def residual(x):
            return x**2 - 2

        solution, accepted = broyden_trust_region(
            residual, lambda x: calculate_jacobian(residual, x), np.array([1.0])
        )
        self.assertEqual(accepted, 6)
        assert_allclose(solution, [np.sqrt(2)], atol=1e-12, rtol=0)
        self.assertLessEqual(np.linalg.norm(residual(solution)), 1e-12)

    def test_rejection_shrinks_radius_without_consuming_accepted_budget(self):
        solution, accepted = broyden_trust_region(
            lambda x: x**2 - 10, lambda x: np.array([[2 * x[0]]]),
            np.array([1.0]), max_iter=1,
        )
        assert_allclose(solution, [3.25])
        self.assertEqual(accepted, 1)

    def test_negative_prediction_uses_the_ratio(self):
        solution, accepted = broyden_trust_region(
            lambda x: 1 - 2*x + 100*x**2, lambda x: np.array([[-2.0]]),
            np.array([0.0]), max_iter=1,
        )
        assert_allclose(solution, [0.5])
        self.assertEqual(accepted, 1)

    def test_zero_budget_still_evaluates_initial_jacobian(self):
        jacobian = unittest.mock.Mock(return_value=np.eye(1))
        start = np.array([0.0])
        solution, accepted = broyden_trust_region(
            lambda x: x - 1, jacobian, start, max_iter=0
        )
        jacobian.assert_called_once_with(start)
        assert_allclose(solution, start)
        self.assertEqual(accepted, 0)

    def test_unpivoted_lu_and_substitution(self):
        matrix = np.array([[4., 1.], [2., 3.]])
        lower, upper = lu_decomposition(matrix)
        assert_allclose(lower @ upper, matrix)
        assert_allclose(solve_lu(lower, upper, np.array([1., 2.])), [0.1, 0.6])


class UtilityTests(unittest.TestCase):
    def test_sorts_mutate_in_place_and_preserve_pair_order(self):
        keys, values = [3, 1, 1], ["c", "a", "b"]
        self.assertIs(sort_pairs(keys, values)[0], keys)
        self.assertEqual((keys, values), ([1, 1, 3], ["a", "b", "c"]))
        unsorted = [3, 2, 1]
        self.assertIs(sort1d(unsorted), unsorted)
        self.assertEqual(unsorted, [1, 2, 3])
        with self.assertRaises(ValueError):
            sort_pairs([1], [])

    def test_gauss_sidel_matches_direct_solution(self):
        matrix = np.array([[5, 1, -1], [2, 7, 1], [-1, 1, 6]])
        rhs = [3, -2, 4]
        actual = gauss_sidel(matrix, rhs, 100, 1e-12)
        assert_allclose(actual, np.linalg.solve(matrix, rhs), atol=1e-12)
        self.assertLess(np.linalg.norm(matrix @ actual - rhs), 1e-11)
        self.assertEqual(gauss_sidel(matrix, rhs, 0, 1e-12), [0, 0, 0])
        with self.assertRaises(ValueError):
            gauss_sidel([[0]], [1], 100, 1e-12)

    def test_spline_interpolation_and_continuity_on_nonuniform_knots(self):
        knots, values = [4.0, 0.0, 0.4, 1.5], [1, 0, 2, -1]
        a, b, c, d = cubic_spline(knots, values)
        for knot, value in zip(knots, values):
            self.assertAlmostEqual(cubic_spline_interpolate(knots, a, b, c, d, knot)[0], value)
        for i, width in enumerate(np.diff(knots)[:-1]):
            self.assertAlmostEqual(6 * a[i] * width + 2 * b[i], 2 * b[i + 1])
            self.assertAlmostEqual((3 * a[i] * width + 2 * b[i]) * width + c[i], c[i + 1])
        self.assertAlmostEqual(cubic_spline_interpolate(knots, a, b, c, d, knots[0])[2], 0)
        self.assertAlmostEqual(cubic_spline_interpolate(knots, a, b, c, d, knots[-1])[2], 0)
        self.assertIsNone(cubic_spline_interpolate(knots, a, b, c, d, -1))

    def test_spline_two_points_and_duplicate_knots(self):
        knots = [0, 2]
        coefficients = cubic_spline(knots, [1, 5])
        assert_allclose(cubic_spline_interpolate(knots, *coefficients, 1), [3, 2, 0])
        with self.assertRaises(ValueError):
            cubic_spline([0, 0], [1, 2])


class PolynomialTests(unittest.TestCase):
    def test_real_and_complex_roots_and_residuals(self):
        expected = np.array([-2, 0.4, 1j, -1j])
        coefficients = np.poly(expected).real
        roots = bairstow(coefficients, 100, 1e-12)
        assert_allclose(np.sort_complex(roots.round(9)), np.sort_complex(expected), atol=1e-9)
        assert_allclose(np.polyval(coefficients, roots), 0, atol=1e-10)

    def test_exact_bairstow_deflation_before_fallback(self):
        # (x² + x + 1)(x - 2) uses the exact initial Bairstow factor.
        roots = bairstow([1, -1, -1, -2], 0, 1e-12)
        assert_allclose(np.sort_complex(roots), [-0.5 - 0.5j * np.sqrt(3), -0.5 + 0.5j * np.sqrt(3), 2])

    def test_bairstow_updates_and_coefficient_scaling(self):
        # The initial factor differs from both factors of (x - 2)(x² + 1).
        with patch("numerical_math.polynomial.np.roots", wraps=np.roots) as roots_solver:
            roots = bairstow(np.array([1, -2, 1, -2]) * 1e100, 100, 1e-12)
        assert_allclose(np.sort_complex(roots.round(9)), [-1j, 1j, 2])
        self.assertTrue(roots_solver.called)
        self.assertLessEqual(max(len(call.args[0]) for call in roots_solver.call_args_list), 3)

    def test_repeated_root_and_real_dtype(self):
        roots = bairstow([1, -6, 12, -8], 0, 1e-12)
        self.assertEqual(roots.size, 3)
        assert_allclose(roots, 2, atol=1e-4)
        self.assertTrue(np.isrealobj(utilities.bairstow([1, -3, 2], 100, 1e-12)))

    def test_quadratic_conjugates_with_nonunit_leading_coefficient(self):
        assert_allclose(np.sort_complex(bairstow([2, 4, 10], 100, 1e-12)), [-1-2j, -1+2j])

    def test_polynomial_degree_and_fallback(self):
        self.assertEqual(bairstow([3], 0, 1e-12).size, 0)
        assert_allclose(bairstow([0, 2, -4], 0, 1e-12), [2])
        assert_allclose(np.sort_complex(utilities.bairstow_EoS([1, 0, 0, -8], 0, 1e-12)),
                        [-1 - 1j * np.sqrt(3), -1 + 1j * np.sqrt(3), 2])
        with self.assertRaises(ValueError):
            bairstow([0, 0], 0, 1e-12)
