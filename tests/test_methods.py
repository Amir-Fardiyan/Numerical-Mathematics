"""Analytic solutions, convergence checks, and regressions for the public API."""

from pathlib import Path
from runpy import run_path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from examples.transport import Res, transport_jacobian
from numerical_math import (
    bairstow,
    broyden_trust_region,
    calculate_jacobian,
    cubic_spline,
    cubic_spline_interpolate,
    gauss_sidel,
    solve_nonlinear,
    sort1d,
    sort_pairs,
)
from utilities.Utilities import utilities


class NonlinearTests(unittest.TestCase):
    def test_centered_jacobian_accepts_integer_input(self):
        def residual(x):
            return [x[0] * x[1], x[0] ** 2 - x[1]]

        assert_allclose(calculate_jacobian(residual, [2, 3]), [[3, 2], [4, -1]], atol=1e-9)

    def test_circle_line_converges_with_monotone_residuals(self):
        def residual(x):
            return [x @ x - 1, x[0] - x[1]]

        result = solve_nonlinear(residual, [0.8, 0.3])
        self.assertTrue(result.converged)
        assert_allclose(result.x, np.sqrt(0.5), atol=1e-10, rtol=0)
        self.assertLessEqual(np.linalg.norm(residual(result.x)), 1e-12)
        self.assertTrue(np.all(np.diff(result.residual_norms) < 0))
        self.assertEqual(len(result.residual_norms), result.accepted_steps + 1)

    def test_zero_pivot_system_and_legacy_return(self):
        jacobian = np.array([[0.0, 1.0], [1.0, 0.0]])
        solution, accepted = broyden_trust_region(
            lambda x: jacobian @ x - [2, 3], lambda x: jacobian, [0, 0]
        )
        assert_allclose(solution, [3, 2], atol=1e-12)
        assert_allclose(jacobian, [[0, 1], [1, 0]])
        self.assertGreater(accepted, 0)

    def test_rank_deficient_consistent_system(self):
        result = solve_nonlinear(lambda x: [x.sum() - 1, 2 * (x.sum() - 1)], [0, 0])
        self.assertTrue(result.converged)
        assert_allclose(result.x, [0.5, 0.5], atol=1e-12)

    def test_rejected_trials_are_bounded(self):
        result = solve_nonlinear(lambda x: [1], [1], lambda x: [[1]], max_iter=3)
        self.assertEqual((result.status, result.trial_steps, result.accepted_steps), ("trial_limit", 3, 0))
        assert_allclose(result.x, [1])

    def test_nonfinite_trial_is_rejected(self):
        result = solve_nonlinear(
            lambda x: [np.nan] if x[0] < 0.1 else [x[0] - 0.2],
            [1], lambda x: [[0.5]], c=0.95, max_iter=1,
        )
        self.assertEqual(result.accepted_steps, 0)
        self.assertEqual(result.trial_steps, 1)

    def test_exact_initial_solution_needs_no_jacobian(self):
        with patch("numerical_math.nonlinear.calculate_jacobian", side_effect=AssertionError):
            result = solve_nonlinear(lambda x: x, [0])
        self.assertTrue(result.converged)
        self.assertEqual(result.trial_steps, 0)

    def test_invalid_shape_and_zero_trial_budget(self):
        with self.assertRaises(ValueError):
            solve_nonlinear(lambda x: [1, 2], [1])
        result = solve_nonlinear(lambda x: x, [1], max_iter=0)
        self.assertEqual(result.status, "trial_limit")

    def test_transport_jacobian_and_mass_balance(self):
        nodes = np.array([0.8, 0.5, 0.1])
        assert_allclose(
            transport_jacobian(nodes, 0.01),
            calculate_jacobian(lambda x: Res(x, 3, 0.01), nodes),
            atol=1e-9,
        )
        result = solve_nonlinear(
            lambda x: Res(x, 30, 0.01), np.full(30, 1e-8),
            lambda x: transport_jacobian(x, 0.01),
        )
        self.assertTrue(result.converged)
        self.assertLessEqual(np.linalg.norm(Res(result.x, 30, 0.01)), 1e-12)
        self.assertAlmostEqual(float(result.x.sum()), 0.3, places=10)

    def test_legacy_lu_helpers(self):
        script = Path(__file__).resolve().parents[1] / "Nonlinear systems" / "Broyden-TrustRegion.py"
        legacy = run_path(str(script))
        matrix = [[4, 1], [2, 3]]
        lower, upper = legacy["lu_decomposition"](matrix)
        assert_allclose(lower @ upper, matrix)
        assert_allclose(legacy["solve_lu"](lower, upper, [1, 2]), [0.1, 0.6])
        with self.assertRaises(np.linalg.LinAlgError):
            legacy["lu_decomposition"]([[0, 1], [1, 0]])


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

    def test_repeated_root_and_legacy_real_dtype(self):
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
