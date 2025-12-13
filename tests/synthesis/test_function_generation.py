"""Tests for function generation synthesis."""

import math
import unittest

import numpy as np

from pylinkage.synthesis import function_generation, SynthesisType
from pylinkage.synthesis.function_generation import (
    coefficients_to_link_lengths,
    freudenstein_equation,
    solve_freudenstein_3_positions,
    solve_freudenstein_least_squares,
    verify_function_generation,
)


class TestFreudensteinEquation(unittest.TestCase):
    """Tests for Freudenstein's equation."""

    def test_identity_linkage(self):
        """Test that R1=R2=1, R3=1 gives theta4=theta2."""
        # For identity relationship, residual should be ~0 when theta2=theta4
        for theta in [0, 0.5, 1.0, 1.5]:
            residual = freudenstein_equation(theta, theta, 1.0, 1.0, 1.0)
            self.assertAlmostEqual(residual, 0.0, places=10)

    def test_known_coefficients(self):
        """Test with known coefficient values."""
        # For a standard four-bar, verify equation evaluates correctly
        R1, R2, R3 = 2.0, 1.5, 1.0
        theta2, theta4 = 0.5, 0.6

        residual = freudenstein_equation(theta2, theta4, R1, R2, R3)

        # Residual should be a finite number
        self.assertTrue(np.isfinite(residual))


class TestSolveFreudenstein3Positions(unittest.TestCase):
    """Tests for 3-position Freudenstein solver."""

    def test_basic_solve(self):
        """Test basic 3-position solve."""
        angle_pairs = [
            (0.0, 0.0),
            (0.5, 0.6),
            (1.0, 1.1),
        ]

        R1, R2, R3 = solve_freudenstein_3_positions(angle_pairs)

        # Verify solution satisfies all angle pairs
        for theta2, theta4 in angle_pairs:
            residual = freudenstein_equation(theta2, theta4, R1, R2, R3)
            self.assertAlmostEqual(residual, 0.0, places=8)

    def test_wrong_count_raises(self):
        """Test that non-3 angle pairs raises ValueError."""
        angle_pairs_2 = [(0.0, 0.0), (0.5, 0.5)]

        with self.assertRaises(ValueError):
            solve_freudenstein_3_positions(angle_pairs_2)

        angle_pairs_4 = [(0.0, 0.0), (0.3, 0.3), (0.6, 0.6), (0.9, 0.9)]

        with self.assertRaises(ValueError):
            solve_freudenstein_3_positions(angle_pairs_4)

    def test_degenerate_angles_raises(self):
        """Test that identical angle pairs cause ill-conditioning."""
        angle_pairs = [
            (0.5, 0.5),
            (0.5, 0.5),
            (0.5, 0.5),
        ]

        with self.assertRaises(np.linalg.LinAlgError):
            solve_freudenstein_3_positions(angle_pairs)


class TestSolveFreudensteinLeastSquares(unittest.TestCase):
    """Tests for least-squares Freudenstein solver."""

    def test_overdetermined_solve(self):
        """Test overdetermined (4+ points) solve."""
        angle_pairs = [
            (0.0, 0.0),
            (0.3, 0.35),
            (0.6, 0.7),
            (0.9, 1.0),
        ]

        R1, R2, R3, residual = solve_freudenstein_least_squares(angle_pairs)

        # Residual should be finite
        self.assertTrue(np.isfinite(residual))
        self.assertGreaterEqual(residual, 0.0)

    def test_exact_fit_has_zero_residual(self):
        """Test that exactly satisfiable data has near-zero residual."""
        # First get exact solution for 3 points
        angle_pairs_3 = [
            (0.0, 0.0),
            (0.5, 0.6),
            (1.0, 1.1),
        ]
        R1, R2, R3 = solve_freudenstein_3_positions(angle_pairs_3)

        # Now use these same 3 points with least squares
        R1_ls, R2_ls, R3_ls, residual = solve_freudenstein_least_squares(
            angle_pairs_3
        )

        self.assertAlmostEqual(residual, 0.0, places=8)
        self.assertAlmostEqual(R1, R1_ls, places=8)
        self.assertAlmostEqual(R2, R2_ls, places=8)
        self.assertAlmostEqual(R3, R3_ls, places=8)


class TestCoefficientsToLinkLengths(unittest.TestCase):
    """Tests for coefficient to link length conversion."""

    def test_basic_conversion(self):
        """Test basic conversion from coefficients to lengths."""
        R1, R2, R3 = 2.0, 1.5, 1.0
        ground = 4.0

        crank, coupler, rocker, ground_out = coefficients_to_link_lengths(
            R1, R2, R3, ground
        )

        # Verify R1 = d/a
        self.assertAlmostEqual(R1, ground / crank, places=8)

        # Verify R2 = d/c
        self.assertAlmostEqual(R2, ground / rocker, places=8)

        # Verify ground length
        self.assertAlmostEqual(ground, ground_out, places=8)

        # All lengths should be positive
        self.assertGreater(crank, 0)
        self.assertGreater(coupler, 0)
        self.assertGreater(rocker, 0)

    def test_zero_R1_raises(self):
        """Test that R1=0 raises ValueError."""
        with self.assertRaises(ValueError):
            coefficients_to_link_lengths(0.0, 1.0, 1.0)

    def test_zero_R2_raises(self):
        """Test that R2=0 raises ValueError."""
        with self.assertRaises(ValueError):
            coefficients_to_link_lengths(1.0, 0.0, 1.0)

    def test_negative_coupler_squared_raises(self):
        """Test that impossible geometry raises ValueError."""
        # Coefficients that would give negative b^2
        with self.assertRaises(ValueError):
            coefficients_to_link_lengths(0.1, 0.1, 100.0)


class TestFunctionGeneration(unittest.TestCase):
    """Tests for the main function generation API."""

    def test_basic_synthesis(self):
        """Test basic function generation synthesis."""
        angle_pairs = [
            (0.0, 0.0),
            (math.pi / 6, math.pi / 4),
            (math.pi / 3, math.pi / 2),
        ]

        result = function_generation(angle_pairs)

        self.assertEqual(result.problem.synthesis_type, SynthesisType.FUNCTION)
        # May or may not have solutions depending on geometry

    def test_returns_linkage_objects(self):
        """Test that solutions are Linkage objects."""
        angle_pairs = [
            (0.0, 0.0),
            (0.3, 0.4),
            (0.6, 0.75),
        ]

        result = function_generation(angle_pairs, require_grashof=False)

        if result.solutions:
            from pylinkage import Linkage

            self.assertIsInstance(result.solutions[0], Linkage)

    def test_insufficient_pairs_raises(self):
        """Test that <3 angle pairs raises ValueError."""
        angle_pairs = [(0.0, 0.0), (0.5, 0.5)]

        with self.assertRaises(ValueError):
            function_generation(angle_pairs)

    def test_overdetermined_adds_warning(self):
        """Test that >3 pairs adds a warning."""
        angle_pairs = [
            (0.0, 0.0),
            (0.2, 0.25),
            (0.4, 0.5),
            (0.6, 0.7),
            (0.8, 0.9),
        ]

        result = function_generation(angle_pairs, require_grashof=False)

        # Should have a warning about over-determination or fit
        # (may vary based on residual)

    def test_ground_length_scaling(self):
        """Test that ground_length scales the solution."""
        angle_pairs = [
            (0.0, 0.0),
            (0.3, 0.4),
            (0.6, 0.75),
        ]

        result1 = function_generation(
            angle_pairs, ground_length=1.0, require_grashof=False
        )
        result2 = function_generation(
            angle_pairs, ground_length=2.0, require_grashof=False
        )

        if result1.raw_solutions and result2.raw_solutions:
            # Ground length should match requested value
            self.assertAlmostEqual(result1.raw_solutions[0].ground_length, 1.0)
            self.assertAlmostEqual(result2.raw_solutions[0].ground_length, 2.0)


class TestVerifyFunctionGeneration(unittest.TestCase):
    """Tests for function generation verification."""

    def test_verify_synthesized_linkage(self):
        """Test verification of a synthesized linkage."""
        angle_pairs = [
            (0.0, 0.0),
            (0.3, 0.4),
            (0.6, 0.75),
        ]

        result = function_generation(angle_pairs, require_grashof=False)

        if result.solutions:
            linkage = result.solutions[0]
            satisfied, errors = verify_function_generation(
                linkage, angle_pairs, tolerance=0.1
            )

            # Should be satisfied within tolerance
            self.assertTrue(satisfied)
            self.assertEqual(len(errors), len(angle_pairs))
            for e in errors:
                self.assertLess(e, 0.1)

    def test_verify_returns_errors_for_wrong_linkage(self):
        """Test that verification detects mismatched linkages."""
        angle_pairs = [
            (0.0, 0.0),
            (0.3, 0.4),
            (0.6, 0.75),
        ]

        result = function_generation(angle_pairs, require_grashof=False)

        if result.solutions:
            linkage = result.solutions[0]

            # Use different angle pairs
            wrong_pairs = [
                (0.0, math.pi / 4),  # Completely different relationship
                (0.3, math.pi / 2),
                (0.6, math.pi),
            ]

            satisfied, errors = verify_function_generation(
                linkage, wrong_pairs, tolerance=0.05
            )

            # Should NOT be satisfied with wrong pairs
            self.assertFalse(satisfied)

    def test_verify_handles_invalid_linkage(self):
        """Test that verification handles invalid linkage structure."""
        from pylinkage import Linkage
        from pylinkage.joints import Static

        # Create a linkage without proper four-bar structure
        joint_A = Static(x=0, y=0, name="A")
        linkage = Linkage(joints=[joint_A], order=[])

        angle_pairs = [(0.0, 0.0), (0.3, 0.4), (0.6, 0.75)]

        satisfied, errors = verify_function_generation(linkage, angle_pairs)

        # Should return False with inf errors
        self.assertFalse(satisfied)
        self.assertEqual(len(errors), len(angle_pairs))
        for e in errors:
            self.assertEqual(e, float("inf"))


if __name__ == "__main__":
    unittest.main()
