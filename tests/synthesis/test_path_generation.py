"""Tests for path generation synthesis."""

import math
import unittest

from pylinkage.synthesis import SynthesisType, path_generation
from pylinkage.synthesis.path_generation import (
    _estimate_orientations_from_path,
    _points_to_poses,
    verify_path_generation,
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
        has_overconstraint_warning = any("over-constrained" in w.lower() for w in result.warnings)
        self.assertTrue(has_overconstraint_warning)

    def test_max_solutions_limit(self):
        """Test that max_solutions limits output."""
        points = [(0, 1), (1, 2), (2, 1)]

        result = path_generation(points, max_solutions=3, require_grashof=False)

        self.assertLessEqual(len(result.solutions), 3)

    def test_returns_linkage_objects(self):
        """Test that solutions are Linkage objects."""
        points = [(0, 1), (1, 2), (2, 1.5)]

        result = path_generation(points, require_grashof=False)

        if result.solutions:
            from pylinkage.simulation import Linkage as SimLinkage

            self.assertIsInstance(result.solutions[0], SimLinkage)

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

    def test_solutions_have_coupler_point(self):
        """Test that solutions include a coupler point joint P."""
        points = [(0, 1), (1, 2), (2, 1.5)]

        result = path_generation(points, require_grashof=False)

        for linkage in result.solutions:
            joint_names = [j.name for j in linkage.components]
            self.assertIn("P", joint_names, "Synthesized linkage should have coupler point joint P")

    def test_raw_solutions_have_coupler_point(self):
        """Test that raw FourBarSolutions include coupler_point."""
        points = [(0, 1), (1, 2), (2, 1.5)]

        result = path_generation(points, require_grashof=False)

        for sol in result.raw_solutions:
            self.assertIsNotNone(
                sol.coupler_point, "Path generation solutions should have coupler_point set"
            )


class TestPathGenerationWithTiming(unittest.TestCase):
    """Tests for path generation with prescribed timing."""

    def test_basic_with_timing(self):
        """Test basic path generation with timing."""
        from pylinkage.synthesis.path_generation import (
            path_generation_with_timing,
        )

        points = [(0, 1), (1, 2), (2, 1)]
        angles = [0, math.pi / 3, 2 * math.pi / 3]

        result = path_generation_with_timing(points, angles, require_grashof=False)

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


class TestVerifyPathGeneration(unittest.TestCase):
    """Tests for the path generation verification function."""

    def test_verify_known_good_fourbar(self):
        """Test verification with a known crank-rocker four-bar.

        Uses fourbar_from_lengths to create a linkage, simulates it to
        find points on its coupler curve, then verifies those points.
        """
        from pylinkage.synthesis.conversion import fourbar_from_lengths

        # Create a known-good Grashof crank-rocker
        linkage = fourbar_from_lengths(
            crank_length=1.0,
            coupler_length=3.0,
            rocker_length=3.0,
            ground_length=4.0,
        )

        # Simulate to get the trajectory of joint C (coupler-rocker joint)
        trajectory = []
        for positions in linkage.step():
            # Joint C is index 3
            cx, cy = positions[3]
            if cx is not None and cy is not None:
                trajectory.append((cx, cy))

        # Pick 3 points from the trajectory
        if len(trajectory) >= 30:
            sample_indices = [0, len(trajectory) // 3, 2 * len(trajectory) // 3]
            sample_points = [trajectory[i] for i in sample_indices]

            # Verify: joint C should pass through these points
            is_ok, distances = verify_path_generation(linkage, sample_points, tolerance=0.5)

            # At least some distances should be small (trajectory was sampled from this linkage)
            self.assertTrue(
                any(d < 0.5 for d in distances),
                f"At least one sample point should be close to trajectory, got {distances}",
            )

    def test_verify_returns_correct_structure(self):
        """Test that verify returns (bool, list[float])."""
        from pylinkage.synthesis.conversion import fourbar_from_lengths

        linkage = fourbar_from_lengths(1.0, 3.0, 3.0, 4.0)
        points = [(1.0, 1.0), (2.0, 2.0)]

        is_ok, distances = verify_path_generation(linkage, points)

        self.assertIsInstance(is_ok, bool)
        self.assertIsInstance(distances, list)
        self.assertEqual(len(distances), 2)

    def test_verify_empty_points(self):
        """Test verification with no precision points."""
        from pylinkage.synthesis.conversion import fourbar_from_lengths

        linkage = fourbar_from_lengths(1.0, 3.0, 3.0, 4.0)

        is_ok, distances = verify_path_generation(linkage, [])

        self.assertTrue(is_ok)
        self.assertEqual(len(distances), 0)


class TestPathGenerationIntegration(unittest.TestCase):
    """Integration tests: synthesize and verify coupler path."""

    def test_synthesized_linkage_can_simulate(self):
        """Test that synthesized linkages can be simulated without error."""
        points = [(0, 1), (1, 2), (2, 1.5)]

        result = path_generation(points, require_grashof=False)

        for linkage in result.solutions[:3]:
            try:
                positions = list(linkage.step(iterations=20))
                self.assertGreater(len(positions), 0)
            except Exception as e:
                self.fail(f"Simulation of synthesized linkage failed: {e}")

    def test_coupler_point_moves_during_simulation(self):
        """Test that the coupler point P actually moves during simulation."""
        points = [(0, 1), (1, 2), (2, 1.5)]

        result = path_generation(points, require_grashof=False)

        for linkage in result.solutions[:2]:
            p_joint = None
            p_idx = None
            for idx, j in enumerate(linkage.components):
                if j.name == "P":
                    p_joint = j
                    p_idx = idx
                    break

            if p_joint is None:
                continue

            positions_list = list(linkage.step(iterations=20))
            p_positions = [pos[p_idx] for pos in positions_list if pos[p_idx][0] is not None]

            if len(p_positions) < 2:
                continue

            # Check that the coupler point actually moves
            x_range = max(p[0] for p in p_positions) - min(p[0] for p in p_positions)
            y_range = max(p[1] for p in p_positions) - min(p[1] for p in p_positions)
            self.assertGreater(
                x_range + y_range, 0.01, "Coupler point P should move during simulation"
            )


if __name__ == "__main__":
    unittest.main()
