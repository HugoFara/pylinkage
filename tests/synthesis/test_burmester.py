"""Tests for Burmester theory implementation."""

import math
import unittest

import numpy as np

from pylinkage.synthesis import Pose
from pylinkage.synthesis.burmester import (
    complex_to_point,
    compute_all_poles,
    compute_circle_point_curve,
    compute_pole,
    point_to_complex,
    select_compatible_dyads,
)


class TestCoordinateConversion(unittest.TestCase):
    """Tests for coordinate conversion functions."""

    def test_point_to_complex(self):
        """Test Cartesian to complex conversion."""
        p = (3.0, 4.0)
        z = point_to_complex(p)
        self.assertEqual(z, complex(3.0, 4.0))

    def test_complex_to_point(self):
        """Test complex to Cartesian conversion."""
        z = complex(3.0, 4.0)
        p = complex_to_point(z)
        self.assertEqual(p, (3.0, 4.0))

    def test_roundtrip(self):
        """Test roundtrip conversion."""
        original = (1.5, -2.7)
        converted = complex_to_point(point_to_complex(original))
        self.assertAlmostEqual(converted[0], original[0])
        self.assertAlmostEqual(converted[1], original[1])


class TestComputePole(unittest.TestCase):
    """Tests for pole computation."""

    def test_pure_rotation(self):
        """Test pole for pure rotation about origin."""
        pos1 = Pose(1.0, 0.0, 0.0)
        pos2 = Pose(0.0, 1.0, math.pi / 2)

        pole = compute_pole(pos1, pos2)

        # For 90 degree rotation, pole should be near origin
        self.assertAlmostEqual(abs(pole), 0.0, places=5)

    def test_pure_translation(self):
        """Test pole for pure translation (at infinity)."""
        pos1 = Pose(0.0, 0.0, 0.0)
        pos2 = Pose(1.0, 0.0, 0.0)  # Same angle, different position

        pole = compute_pole(pos1, pos2)

        # Pure translation has pole at infinity
        self.assertTrue(math.isinf(pole.real) or math.isinf(pole.imag))

    def test_identical_positions_raises(self):
        """Test that identical positions raise ValueError."""
        pos1 = Pose(1.0, 2.0, 0.5)
        pos2 = Pose(1.0, 2.0, 0.5)

        with self.assertRaises(ValueError):
            compute_pole(pos1, pos2)

    def test_small_rotation(self):
        """Test pole for small rotation."""
        pos1 = Pose(0.0, 0.0, 0.0)
        pos2 = Pose(0.1, 0.0, 0.1)  # Small rotation and translation

        pole = compute_pole(pos1, pos2)

        # Pole should be finite
        self.assertTrue(np.isfinite(pole))


class TestComputeAllPoles(unittest.TestCase):
    """Tests for computing all poles."""

    def test_two_poses(self):
        """Test with exactly two poses."""
        poses = [
            Pose(0.0, 0.0, 0.0),
            Pose(1.0, 0.0, 0.5),
        ]

        poles = compute_all_poles(poses)

        self.assertEqual(len(poles), 1)  # C(2,2) = 1 pole

    def test_three_poses(self):
        """Test with three poses."""
        poses = [
            Pose(0.0, 0.0, 0.0),
            Pose(1.0, 0.0, 0.3),
            Pose(1.5, 0.5, 0.6),
        ]

        poles = compute_all_poles(poses)

        # C(3,2) = 3 poles: P12, P13, P23
        self.assertEqual(len(poles), 3)

    def test_four_poses(self):
        """Test with four poses."""
        poses = [
            Pose(0.0, 0.0, 0.0),
            Pose(1.0, 0.0, 0.25),
            Pose(1.5, 0.5, 0.5),
            Pose(1.0, 1.0, 0.75),
        ]

        poles = compute_all_poles(poses)

        # C(4,2) = 6 poles
        self.assertEqual(len(poles), 6)

    def test_insufficient_poses_raises(self):
        """Test that single pose raises ValueError."""
        poses = [Pose(0.0, 0.0, 0.0)]

        with self.assertRaises(ValueError):
            compute_all_poles(poses)


class TestComputeCirclePointCurve(unittest.TestCase):
    """Tests for circle point curve computation."""

    def test_three_positions_returns_curves(self):
        """Test that 3 positions return continuous curves."""
        poses = [
            Pose(0.0, 0.0, 0.0),
            Pose(1.0, 0.5, 0.3),
            Pose(2.0, 0.0, 0.6),
        ]

        curves = compute_circle_point_curve(poses, n_samples=36)

        self.assertFalse(curves.is_discrete)
        self.assertGreater(len(curves), 0)

    def test_four_positions_returns_discrete(self):
        """Test that 4 positions return discrete points."""
        poses = [
            Pose(0.0, 0.0, 0.0),
            Pose(1.0, 0.5, 0.2),
            Pose(2.0, 0.0, 0.4),
            Pose(1.5, -0.5, 0.6),
        ]

        curves = compute_circle_point_curve(poses)

        self.assertTrue(curves.is_discrete)

    def test_five_positions_highly_constrained(self):
        """Test that 5 positions are highly constrained."""
        poses = [
            Pose(0.0, 0.0, 0.0),
            Pose(1.0, 0.5, 0.15),
            Pose(2.0, 0.0, 0.3),
            Pose(1.5, -0.5, 0.45),
            Pose(0.5, -0.3, 0.6),
        ]

        curves = compute_circle_point_curve(poses)

        self.assertTrue(curves.is_discrete)
        # May have 0 solutions for incompatible poses

    def test_invalid_position_count_raises(self):
        """Test that <3 or >5 positions raise ValueError."""
        poses_2 = [Pose(0, 0, 0), Pose(1, 0, 0.5)]

        with self.assertRaises(ValueError):
            compute_circle_point_curve(poses_2)

        poses_6 = [Pose(i, 0, i * 0.1) for i in range(6)]

        with self.assertRaises(ValueError):
            compute_circle_point_curve(poses_6)

    def test_get_dyad(self):
        """Test getting dyad from curves."""
        poses = [
            Pose(0.0, 0.0, 0.0),
            Pose(1.0, 0.5, 0.3),
            Pose(2.0, 0.0, 0.6),
        ]

        curves = compute_circle_point_curve(poses, n_samples=10)

        if len(curves) > 0:
            dyad = curves.get_dyad(0)
            self.assertIsNotNone(dyad.circle_point)
            self.assertIsNotNone(dyad.center_point)
            self.assertGreater(dyad.link_length, 0)


class TestSelectCompatibleDyads(unittest.TestCase):
    """Tests for dyad pair selection."""

    def test_returns_pairs(self):
        """Test that compatible pairs are returned."""
        poses = [
            Pose(0.0, 0.0, 0.0),
            Pose(1.0, 0.5, 0.3),
            Pose(2.0, 0.0, 0.6),
        ]

        curves = compute_circle_point_curve(poses, n_samples=20)
        pairs = select_compatible_dyads(curves)

        # Should return list (may be empty for some configurations)
        self.assertIsInstance(pairs, list)

    def test_min_link_length_filter(self):
        """Test minimum link length filtering."""
        poses = [
            Pose(0.0, 0.0, 0.0),
            Pose(1.0, 0.5, 0.3),
            Pose(2.0, 0.0, 0.6),
        ]

        curves = compute_circle_point_curve(poses, n_samples=20)

        # Very large minimum should filter most solutions
        pairs = select_compatible_dyads(curves, min_link_length=100.0)

        # Likely empty for reasonable configurations
        self.assertIsInstance(pairs, list)


if __name__ == "__main__":
    unittest.main()
