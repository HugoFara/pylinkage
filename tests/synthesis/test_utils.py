"""Tests for synthesis utility functions."""

import math
import unittest

from pylinkage.synthesis import (
    FourBarSolution,
    grashof_check,
    GrashofType,
    is_crank_rocker,
    is_grashof,
    validate_fourbar,
)
from pylinkage.synthesis.utils import (
    angle_between_points,
    distance,
    normalize_angle,
    rotate_point,
)


class TestGrashofCheck(unittest.TestCase):
    """Tests for Grashof criterion checking."""

    def test_crank_rocker(self):
        """Test crank-rocker classification."""
        # Classic crank-rocker: shortest is crank
        # s=1, l=4, p=3, q=3; s+l=5 < p+q=6
        result = grashof_check(crank=1, coupler=3, rocker=3, ground=4)

        self.assertEqual(result, GrashofType.GRASHOF_CRANK_ROCKER)

    def test_double_crank(self):
        """Test double-crank (drag link) classification."""
        # Shortest link is ground, s+l < p+q
        # ground=1, crank=2, coupler=2.5, rocker=2: s+l=1+2.5=3.5 < p+q=2+2=4
        result = grashof_check(crank=2, coupler=2.5, rocker=2, ground=1)

        self.assertEqual(result, GrashofType.GRASHOF_DOUBLE_CRANK)

    def test_double_rocker(self):
        """Test double-rocker classification."""
        # Shortest link is coupler, s+l < p+q
        # crank=2, coupler=1, rocker=2, ground=2.5: s+l=1+2.5=3.5 < p+q=2+2=4
        result = grashof_check(crank=2, coupler=1, rocker=2, ground=2.5)

        self.assertEqual(result, GrashofType.GRASHOF_DOUBLE_ROCKER)

    def test_non_grashof(self):
        """Test non-Grashof classification."""
        # s+l > p+q: 1+5=6 > 2+3=5
        result = grashof_check(crank=5, coupler=2, rocker=3, ground=1)

        self.assertEqual(result, GrashofType.NON_GRASHOF)

    def test_change_point(self):
        """Test change point (special Grashof) classification."""
        # s+l = p+q: 1+4=5 = 2+3=5
        result = grashof_check(crank=1, coupler=2, rocker=3, ground=4)

        self.assertEqual(result, GrashofType.CHANGE_POINT)


class TestIsGrashof(unittest.TestCase):
    """Tests for is_grashof helper."""

    def test_grashof_linkage(self):
        """Test that Grashof linkage returns True."""
        self.assertTrue(is_grashof(1, 3, 3, 4))

    def test_non_grashof_linkage(self):
        """Test that non-Grashof linkage returns False."""
        self.assertFalse(is_grashof(5, 2, 3, 1))


class TestIsCrankRocker(unittest.TestCase):
    """Tests for is_crank_rocker helper."""

    def test_crank_rocker_returns_true(self):
        """Test crank-rocker returns True."""
        self.assertTrue(is_crank_rocker(1, 3, 3, 4))

    def test_double_crank_returns_false(self):
        """Test double-crank returns False."""
        # ground=1 is shortest, so this is double-crank not crank-rocker
        self.assertFalse(is_crank_rocker(2, 2.5, 2, 1))

    def test_non_grashof_returns_false(self):
        """Test non-Grashof returns False."""
        self.assertFalse(is_crank_rocker(5, 2, 3, 1))


class TestValidateFourbar(unittest.TestCase):
    """Tests for four-bar solution validation."""

    def test_valid_solution(self):
        """Test validation of valid solution."""
        solution = FourBarSolution(
            ground_pivot_a=(0, 0),
            ground_pivot_d=(4, 0),
            crank_pivot_b=(1, 0),
            coupler_pivot_c=(3, 2),
            crank_length=1.0,
            coupler_length=3.0,
            rocker_length=3.0,
            ground_length=4.0,
        )

        is_valid, errors = validate_fourbar(solution)

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_zero_length_link(self):
        """Test that zero-length link is invalid."""
        solution = FourBarSolution(
            ground_pivot_a=(0, 0),
            ground_pivot_d=(4, 0),
            crank_pivot_b=(0, 0),  # Same as A
            coupler_pivot_c=(3, 2),
            crank_length=0.0,  # Zero length
            coupler_length=3.0,
            rocker_length=3.0,
            ground_length=4.0,
        )

        is_valid, errors = validate_fourbar(solution)

        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)

    def test_negative_length_link(self):
        """Test that negative-length link is invalid."""
        solution = FourBarSolution(
            ground_pivot_a=(0, 0),
            ground_pivot_d=(4, 0),
            crank_pivot_b=(1, 0),
            coupler_pivot_c=(3, 2),
            crank_length=-1.0,  # Negative
            coupler_length=3.0,
            rocker_length=3.0,
            ground_length=4.0,
        )

        is_valid, errors = validate_fourbar(solution)

        self.assertFalse(is_valid)


class TestGeometryHelpers(unittest.TestCase):
    """Tests for geometry helper functions."""

    def test_distance(self):
        """Test distance calculation."""
        d = distance((0, 0), (3, 4))
        self.assertAlmostEqual(d, 5.0)

    def test_distance_same_point(self):
        """Test distance between same point is zero."""
        d = distance((1, 2), (1, 2))
        self.assertAlmostEqual(d, 0.0)

    def test_normalize_angle_positive(self):
        """Test normalizing positive angle > pi."""
        angle = normalize_angle(3 * math.pi)
        self.assertAlmostEqual(angle, math.pi, places=10)

    def test_normalize_angle_negative(self):
        """Test normalizing negative angle < -pi."""
        angle = normalize_angle(-3 * math.pi)
        self.assertAlmostEqual(angle, -math.pi, places=10)

    def test_normalize_angle_in_range(self):
        """Test that angle in range is unchanged."""
        angle = normalize_angle(0.5)
        self.assertAlmostEqual(angle, 0.5)

    def test_rotate_point_90_degrees(self):
        """Test 90-degree rotation."""
        rotated = rotate_point((1, 0), math.pi / 2)
        self.assertAlmostEqual(rotated[0], 0.0, places=10)
        self.assertAlmostEqual(rotated[1], 1.0, places=10)

    def test_rotate_point_about_center(self):
        """Test rotation about non-origin center."""
        # Rotate (2, 0) about (1, 0) by 90 degrees
        rotated = rotate_point((2, 0), math.pi / 2, center=(1, 0))
        self.assertAlmostEqual(rotated[0], 1.0, places=10)
        self.assertAlmostEqual(rotated[1], 1.0, places=10)

    def test_angle_between_points(self):
        """Test angle calculation."""
        angle = angle_between_points((0, 0), (1, 1))
        self.assertAlmostEqual(angle, math.pi / 4, places=10)

    def test_angle_between_points_horizontal(self):
        """Test horizontal angle."""
        angle = angle_between_points((0, 0), (1, 0))
        self.assertAlmostEqual(angle, 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
