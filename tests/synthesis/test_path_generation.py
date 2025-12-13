"""Tests for path generation synthesis."""

import unittest

from pylinkage.synthesis import path_generation, SynthesisType
from pylinkage.synthesis.path_generation import (
    _estimate_orientations_from_path,
    _points_to_poses,
)


class TestEstimateOrientations(unittest.TestCase):
    """Tests for orientation estimation from path."""

    def test_straight_line_path(self):
        """Test orientations for straight line path."""
        points = [(0, 0), (1, 0), (2, 0), (3, 0)]

        orientations = _estimate_orientations_from_path(points)

        self.assertEqual(len(orientations), 4)
        # All orientations should be ~0 for horizontal line
        for o in orientations:
            self.assertAlmostEqual(o, 0.0, places=5)

    def test_diagonal_path(self):
        """Test orientations for diagonal path."""
        import math

        points = [(0, 0), (1, 1), (2, 2)]

        orientations = _estimate_orientations_from_path(points)

        # Should be ~45 degrees for 45-degree diagonal
        for o in orientations:
            self.assertAlmostEqual(o, math.pi / 4, places=5)

    def test_curved_path(self):
        """Test orientations for curved path."""
        points = [(0, 0), (1, 1), (2, 0)]

        orientations = _estimate_orientations_from_path(points)

        self.assertEqual(len(orientations), 3)
        # First segment goes up-right, second goes down-right


class TestPointsToPoses(unittest.TestCase):
    """Tests for converting points to poses."""

    def test_basic_conversion(self):
        """Test basic points to poses conversion."""
        points = [(0, 0), (1, 1), (2, 0)]
        orientations = [0.0, 0.5, 1.0]

        poses = _points_to_poses(points, orientations)

        self.assertEqual(len(poses), 3)

        self.assertEqual(poses[0].x, 0)
        self.assertEqual(poses[0].y, 0)
        self.assertEqual(poses[0].angle, 0.0)

        self.assertEqual(poses[1].x, 1)
        self.assertEqual(poses[1].y, 1)
        self.assertEqual(poses[1].angle, 0.5)


class TestPathGeneration(unittest.TestCase):
    """Tests for the main path generation API."""

    def test_basic_synthesis(self):
        """Test basic path generation synthesis."""
        points = [(0, 1), (1, 2), (2, 1.5), (3, 0)]

        result = path_generation(points, require_grashof=False)

        self.assertEqual(result.problem.synthesis_type, SynthesisType.PATH)
        self.assertEqual(len(result.problem.precision_points), 4)

    def test_three_points(self):
        """Test with 3 precision points."""
        points = [(0, 0), (1, 1), (2, 0)]

        result = path_generation(points, require_grashof=False)

        self.assertEqual(result.problem.synthesis_type, SynthesisType.PATH)

    def test_insufficient_points_warns(self):
        """Test that <3 points adds warning."""
        points = [(0, 0), (1, 1)]

        result = path_generation(points)

        self.assertTrue(len(result.warnings) > 0)
        self.assertEqual(len(result.solutions), 0)

    def test_many_points_warns(self):
        """Test that >5 points adds warning."""
        points = [(i, i % 2) for i in range(7)]

        result = path_generation(points, require_grashof=False)

        # Should have warning about over-constraint
        has_overconstraint_warning = any(
            "over-constrained" in w.lower() for w in result.warnings
        )
        self.assertTrue(has_overconstraint_warning)

    def test_max_solutions_limit(self):
        """Test that max_solutions limits output."""
        points = [(0, 1), (1, 2), (2, 1)]

        result = path_generation(
            points, max_solutions=3, require_grashof=False
        )

        self.assertLessEqual(len(result.solutions), 3)

    def test_returns_linkage_objects(self):
        """Test that solutions are Linkage objects."""
        points = [(0, 1), (1, 2), (2, 1.5)]

        result = path_generation(points, require_grashof=False)

        if result.solutions:
            from pylinkage import Linkage

            self.assertIsInstance(result.solutions[0], Linkage)

    def test_ground_constraint(self):
        """Test with ground pivot constraints."""
        points = [(0, 1), (1, 2), (2, 1)]

        result = path_generation(
            points,
            ground_pivot_a=(0, 0),
            ground_pivot_d=(3, 0),
            require_grashof=False,
        )

        # Should return result (possibly empty if constraints incompatible)
        self.assertIsInstance(result.solutions, list)


class TestPathGenerationWithTiming(unittest.TestCase):
    """Tests for path generation with prescribed timing."""

    def test_basic_with_timing(self):
        """Test basic path generation with timing."""
        import math

        from pylinkage.synthesis.path_generation import (
            path_generation_with_timing,
        )

        points = [(0, 1), (1, 2), (2, 1)]
        angles = [0, math.pi / 3, 2 * math.pi / 3]

        result = path_generation_with_timing(
            points, angles, require_grashof=False
        )

        self.assertEqual(result.problem.synthesis_type, SynthesisType.PATH)

    def test_mismatched_lengths_warns(self):
        """Test that mismatched points/angles adds warning."""
        from pylinkage.synthesis.path_generation import (
            path_generation_with_timing,
        )

        points = [(0, 1), (1, 2), (2, 1)]
        angles = [0, 0.5]  # Only 2 angles for 3 points

        result = path_generation_with_timing(points, angles)

        self.assertTrue(len(result.warnings) > 0)


if __name__ == "__main__":
    unittest.main()
