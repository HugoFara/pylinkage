"""Tests for synthesis solution conversion."""

import math
import unittest

from pylinkage.synthesis import FourBarSolution, SynthesisType
from pylinkage.synthesis.conversion import (
    fourbar_from_lengths,
    linkage_to_synthesis_params,
    solution_to_linkage,
    solutions_to_linkages,
)


class TestSolutionToLinkage(unittest.TestCase):
    """Tests for converting FourBarSolution to Linkage."""

    def test_basic_conversion(self):
        """Test basic conversion from solution to linkage."""
        solution = FourBarSolution(
            ground_pivot_a=(0, 0),
            ground_pivot_d=(4, 0),
            crank_pivot_b=(1, 0),
            coupler_pivot_c=(3, 2),
            crank_length=1.0,
            coupler_length=2.83,  # approx sqrt((3-1)^2 + (2-0)^2)
            rocker_length=2.24,  # approx sqrt((3-4)^2 + (2-0)^2)
            ground_length=4.0,
        )

        linkage = solution_to_linkage(solution, name="test_linkage")

        from pylinkage import Linkage

        self.assertIsInstance(linkage, Linkage)
        self.assertEqual(linkage.name, "test_linkage")

    def test_joint_positions(self):
        """Test that joint positions are correct."""
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

        linkage = solution_to_linkage(solution)

        # Check joint positions
        joints = linkage.joints
        self.assertEqual(len(joints), 4)

        # Joint A (Static)
        self.assertAlmostEqual(joints[0].x, 0.0)
        self.assertAlmostEqual(joints[0].y, 0.0)

        # Joint D (Static)
        self.assertAlmostEqual(joints[1].x, 4.0)
        self.assertAlmostEqual(joints[1].y, 0.0)

    def test_iterations_parameter(self):
        """Test that iterations parameter affects angle step."""
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

        linkage_100 = solution_to_linkage(solution, iterations=100)
        linkage_50 = solution_to_linkage(solution, iterations=50)

        # Crank angle step should be 2*pi/iterations
        from pylinkage import Crank

        crank_100 = [j for j in linkage_100.joints if isinstance(j, Crank)][0]
        crank_50 = [j for j in linkage_50.joints if isinstance(j, Crank)][0]

        self.assertAlmostEqual(crank_100.angle, 2 * math.pi / 100)
        self.assertAlmostEqual(crank_50.angle, 2 * math.pi / 50)


class TestSolutionsToLinkages(unittest.TestCase):
    """Tests for batch conversion of solutions."""

    def test_empty_list(self):
        """Test conversion of empty list."""
        linkages = solutions_to_linkages([], SynthesisType.FUNCTION)
        self.assertEqual(len(linkages), 0)

    def test_multiple_solutions(self):
        """Test conversion of multiple solutions."""
        solutions = [
            FourBarSolution(
                ground_pivot_a=(0, 0),
                ground_pivot_d=(4, 0),
                crank_pivot_b=(1, 0),
                coupler_pivot_c=(3, 2),
                crank_length=1.0,
                coupler_length=3.0,
                rocker_length=3.0,
                ground_length=4.0,
            ),
            FourBarSolution(
                ground_pivot_a=(0, 0),
                ground_pivot_d=(5, 0),
                crank_pivot_b=(1.5, 0),
                coupler_pivot_c=(4, 2),
                crank_length=1.5,
                coupler_length=3.0,
                rocker_length=3.0,
                ground_length=5.0,
            ),
        ]

        linkages = solutions_to_linkages(solutions, SynthesisType.PATH)

        self.assertEqual(len(linkages), 2)

    def test_naming_by_type(self):
        """Test that linkages are named by synthesis type."""
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

        linkages = solutions_to_linkages([solution], SynthesisType.MOTION)

        self.assertTrue("motion" in linkages[0].name.lower())


class TestLinkageToSynthesisParams(unittest.TestCase):
    """Tests for extracting parameters from linkage."""

    def test_roundtrip(self):
        """Test roundtrip: solution -> linkage -> solution."""
        original = FourBarSolution(
            ground_pivot_a=(0, 0),
            ground_pivot_d=(4, 0),
            crank_pivot_b=(1, 0),
            coupler_pivot_c=(3, 2),
            crank_length=1.0,
            coupler_length=3.0,
            rocker_length=2.5,
            ground_length=4.0,
        )

        linkage = solution_to_linkage(original)
        extracted = linkage_to_synthesis_params(linkage)

        # Check ground pivots
        self.assertAlmostEqual(extracted.ground_pivot_a[0], 0.0)
        self.assertAlmostEqual(extracted.ground_pivot_a[1], 0.0)
        self.assertAlmostEqual(extracted.ground_pivot_d[0], 4.0)
        self.assertAlmostEqual(extracted.ground_pivot_d[1], 0.0)

        # Check link lengths (approximately due to numerical precision)
        self.assertAlmostEqual(extracted.crank_length, 1.0, places=5)
        self.assertAlmostEqual(extracted.ground_length, 4.0, places=5)


class TestFourbarFromLengths(unittest.TestCase):
    """Tests for creating four-bar from link lengths."""

    def test_basic_creation(self):
        """Test basic four-bar creation from lengths."""
        linkage = fourbar_from_lengths(
            crank_length=1.0,
            coupler_length=3.0,
            rocker_length=3.0,
            ground_length=4.0,
        )

        from pylinkage import Linkage

        self.assertIsInstance(linkage, Linkage)
        self.assertEqual(len(linkage.joints), 4)

    def test_custom_ground_pivot(self):
        """Test custom ground pivot position."""
        linkage = fourbar_from_lengths(
            crank_length=1.0,
            coupler_length=3.0,
            rocker_length=3.0,
            ground_length=4.0,
            ground_pivot_a=(10, 5),
        )

        # First joint should be at specified position
        self.assertAlmostEqual(linkage.joints[0].x, 10.0)
        self.assertAlmostEqual(linkage.joints[0].y, 5.0)

    def test_initial_crank_angle(self):
        """Test initial crank angle."""
        linkage = fourbar_from_lengths(
            crank_length=1.0,
            coupler_length=3.0,
            rocker_length=3.0,
            ground_length=4.0,
            initial_crank_angle=math.pi / 4,
        )

        from pylinkage import Crank

        crank = [j for j in linkage.joints if isinstance(j, Crank)][0]

        # Crank should be at 45 degrees from A
        expected_x = 1.0 * math.cos(math.pi / 4)
        expected_y = 1.0 * math.sin(math.pi / 4)

        self.assertAlmostEqual(crank.x, expected_x, places=5)
        self.assertAlmostEqual(crank.y, expected_y, places=5)

    def test_invalid_geometry_raises(self):
        """Test that impossible geometry raises ValueError."""
        # Coupler and rocker too short to reach
        with self.assertRaises(ValueError):
            fourbar_from_lengths(
                crank_length=5.0,  # Very long crank
                coupler_length=0.1,  # Very short coupler
                rocker_length=0.1,  # Very short rocker
                ground_length=1.0,
            )

    def test_can_simulate(self):
        """Test that created linkage can be simulated."""
        linkage = fourbar_from_lengths(
            crank_length=1.0,
            coupler_length=3.0,
            rocker_length=3.0,
            ground_length=4.0,
        )

        # Should be able to step without error
        try:
            positions = list(linkage.step(iterations=10))
            self.assertEqual(len(positions), 10)
        except Exception as e:
            self.fail(f"Simulation failed: {e}")


if __name__ == "__main__":
    unittest.main()
