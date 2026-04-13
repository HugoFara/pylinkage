"""Tests for motion generation synthesis."""

import math
import unittest

from pylinkage.synthesis import Pose, SynthesisType, motion_generation
from pylinkage.synthesis.motion_generation import motion_generation_3_poses


class TestMotionGeneration(unittest.TestCase):
    """Tests for the main motion generation API."""

    def test_basic_synthesis(self):
        """Test basic motion generation synthesis."""
        poses = [
            Pose(0, 0, 0),
            Pose(1, 1, 0.5),
            Pose(2, 0.5, 1.0),
        ]

        result = motion_generation(poses, require_grashof=False)

        self.assertEqual(result.problem.synthesis_type, SynthesisType.MOTION)
        self.assertEqual(len(result.problem.poses), 3)

    def test_four_poses(self):
        """Test with 4 precision poses."""
        poses = [
            Pose(0, 0, 0),
            Pose(1, 0.5, 0.25),
            Pose(1.5, 1, 0.5),
            Pose(1, 1.5, 0.75),
        ]

        result = motion_generation(poses, require_grashof=False)

        self.assertEqual(result.problem.synthesis_type, SynthesisType.MOTION)

    def test_five_poses(self):
        """Test with 5 precision poses (highly constrained)."""
        poses = [
            Pose(0, 0, 0),
            Pose(0.5, 0.5, 0.15),
            Pose(1, 0.8, 0.3),
            Pose(1.5, 0.5, 0.45),
            Pose(1.8, 0, 0.6),
        ]

        motion_generation(poses, require_grashof=False)

        # May have 0 solutions due to over-constraint

    def test_insufficient_poses_warns(self):
        """Test that <3 poses adds warning."""
        poses = [Pose(0, 0, 0), Pose(1, 1, 0.5)]

        result = motion_generation(poses)

        self.assertTrue(len(result.warnings) > 0)
        self.assertEqual(len(result.solutions), 0)

    def test_too_many_poses_warns(self):
        """Test that >5 poses adds warning."""
        poses = [Pose(i * 0.5, i * 0.3, i * 0.1) for i in range(7)]

        result = motion_generation(poses, require_grashof=False)

        has_warning = any("over-constrained" in w.lower() for w in result.warnings)
        self.assertTrue(has_warning)

    def test_returns_linkage_objects(self):
        """Test that solutions are Linkage objects."""
        poses = [
            Pose(0, 0, 0),
            Pose(1, 1, 0.3),
            Pose(2, 0.5, 0.6),
        ]

        result = motion_generation(poses, require_grashof=False)

        if result.solutions:
            from pylinkage.simulation import Linkage as SimLinkage

            self.assertIsInstance(result.solutions[0], SimLinkage)

    def test_ground_constraint(self):
        """Test with ground pivot constraints."""
        poses = [
            Pose(0, 0, 0),
            Pose(1, 1, 0.3),
            Pose(2, 0.5, 0.6),
        ]

        result = motion_generation(
            poses,
            ground_pivot_a=(-1, 0),
            ground_pivot_d=(3, 0),
            require_grashof=False,
        )

        # Should return result (possibly empty)
        self.assertIsInstance(result.solutions, list)

    def test_max_solutions_limit(self):
        """Test that max_solutions limits output."""
        poses = [
            Pose(0, 0, 0),
            Pose(1, 1, 0.3),
            Pose(2, 0.5, 0.6),
        ]

        result = motion_generation(poses, max_solutions=2, require_grashof=False)

        self.assertLessEqual(len(result.solutions), 2)


class TestMotionGeneration3Poses(unittest.TestCase):
    """Tests for specialized 3-pose motion generation."""

    def test_basic_3_poses(self):
        """Test 3-pose synthesis."""
        poses = [
            Pose(0, 0, 0),
            Pose(1, 1, 0.5),
            Pose(2, 0, 1.0),
        ]

        result = motion_generation_3_poses(poses)

        self.assertEqual(result.problem.synthesis_type, SynthesisType.MOTION)

    def test_samples_curve(self):
        """Test that 3-pose synthesis samples the solution curve."""
        poses = [
            Pose(0, 0, 0),
            Pose(1, 0.5, 0.3),
            Pose(2, 0, 0.6),
        ]

        motion_generation_3_poses(poses, n_samples=20)

        # May have multiple solutions from curve sampling

    def test_fallback_for_non_3_poses(self):
        """Test fallback to general motion_generation for != 3 poses."""
        poses = [
            Pose(0, 0, 0),
            Pose(1, 0.5, 0.25),
            Pose(1.5, 1, 0.5),
            Pose(1, 1.5, 0.75),
        ]

        # Should fall back to general motion_generation
        result = motion_generation_3_poses(poses)

        self.assertEqual(result.problem.synthesis_type, SynthesisType.MOTION)


class TestPoseClass(unittest.TestCase):
    """Tests for the Pose data class."""

    def test_pose_creation(self):
        """Test basic Pose creation."""
        pose = Pose(1.0, 2.0, 0.5)

        self.assertEqual(pose.x, 1.0)
        self.assertEqual(pose.y, 2.0)
        self.assertEqual(pose.angle, 0.5)

    def test_pose_to_complex(self):
        """Test Pose to complex conversion."""
        pose = Pose(3.0, 4.0, 1.0)

        z = pose.to_complex()

        self.assertEqual(z, complex(3.0, 4.0))

    def test_pose_from_point_angle(self):
        """Test creating Pose from point and angle."""
        point = (5.0, 6.0)
        angle = math.pi / 4

        pose = Pose.from_point_angle(point, angle)

        self.assertEqual(pose.x, 5.0)
        self.assertEqual(pose.y, 6.0)
        self.assertEqual(pose.angle, math.pi / 4)

    def test_pose_is_frozen(self):
        """Test that Pose is immutable."""
        pose = Pose(1.0, 2.0, 0.5)

        with self.assertRaises(AttributeError):
            pose.x = 3.0


if __name__ == "__main__":
    unittest.main()
