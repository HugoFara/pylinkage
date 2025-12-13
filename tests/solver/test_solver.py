"""Tests for the pure-numba solver module."""

import math
import unittest

import numpy as np

import pylinkage as pl
from pylinkage.solver import (
    JOINT_CRANK,
    JOINT_REVOLUTE,
    JOINT_STATIC,
    SolverData,
    has_nan_positions,
    linkage_to_solver_data,
    simulate,
    solve_crank,
    solve_revolute,
)


class TestJointSolvers(unittest.TestCase):
    """Tests for individual joint solver functions."""

    def test_solve_crank_basic(self):
        """Test basic crank rotation."""
        # Start at angle 0 (right of anchor)
        x, y = solve_crank(
            current_x=1.0,
            current_y=0.0,
            anchor_x=0.0,
            anchor_y=0.0,
            radius=1.0,
            angle_rate=math.pi / 2,  # 90 degrees
            dt=1.0,
        )
        # After 90 degree rotation, should be at top
        self.assertAlmostEqual(x, 0.0, places=10)
        self.assertAlmostEqual(y, 1.0, places=10)

    def test_solve_crank_with_offset_anchor(self):
        """Test crank rotation with non-zero anchor."""
        x, y = solve_crank(
            current_x=11.0,
            current_y=5.0,
            anchor_x=10.0,
            anchor_y=5.0,
            radius=1.0,
            angle_rate=math.pi,  # 180 degrees
            dt=1.0,
        )
        # After 180 degree rotation, should be on opposite side
        self.assertAlmostEqual(x, 9.0, places=10)
        self.assertAlmostEqual(y, 5.0, places=10)

    def test_solve_revolute_basic(self):
        """Test basic revolute joint solving."""
        # Two circles that intersect
        x, y = solve_revolute(
            current_x=1.5,
            current_y=1.0,  # Hint toward upper intersection
            p0_x=0.0,
            p0_y=0.0,
            r0=2.0,
            p1_x=3.0,
            p1_y=0.0,
            r1=2.0,
        )
        # Should be at intersection point (1.5, ~1.32)
        self.assertAlmostEqual(x, 1.5, places=10)
        self.assertGreater(y, 0)  # Upper intersection

    def test_solve_revolute_returns_nan_when_unbuildable(self):
        """Test that unbuildable configuration returns NaN."""
        # Two circles that don't intersect (too far apart)
        x, y = solve_revolute(
            current_x=0.0,
            current_y=0.0,
            p0_x=0.0,
            p0_y=0.0,
            r0=1.0,
            p1_x=10.0,
            p1_y=0.0,
            r1=1.0,
        )
        self.assertTrue(math.isnan(x))
        self.assertTrue(math.isnan(y))


class TestConversion(unittest.TestCase):
    """Tests for linkage to solver data conversion."""

    def test_fourbar_conversion(self):
        """Test conversion of a four-bar linkage."""
        # Create a simple four-bar linkage
        ground = pl.Static(0, 0, name="ground")
        crank = pl.Crank(
            1,
            0,
            joint0=ground,
            distance=1,
            angle=0.1,
            name="crank",
        )
        pin = pl.Revolute(
            2,
            1,
            joint0=crank,
            joint1=(3, 0),
            distance0=2,
            distance1=2,
            name="pin",
        )
        linkage = pl.Linkage(
            joints=(ground, crank, pin),
            order=(ground, crank, pin),
            name="test_fourbar",
        )

        data = linkage_to_solver_data(linkage)

        # Check array shapes (4 joints: ground, crank, pin, implicit Static at (3, 0))
        self.assertEqual(data.n_joints, 4)
        self.assertEqual(data.positions.shape, (4, 2))
        self.assertEqual(len(data.solve_order), 3)  # Only explicit joints in solve order

        # Check joint types for explicit joints
        self.assertEqual(data.joint_types[0], JOINT_STATIC)
        self.assertEqual(data.joint_types[1], JOINT_CRANK)
        self.assertEqual(data.joint_types[2], JOINT_REVOLUTE)
        self.assertEqual(data.joint_types[3], JOINT_STATIC)  # implicit

        # Check positions are extracted correctly
        np.testing.assert_array_almost_equal(data.positions[0], [0, 0])
        np.testing.assert_array_almost_equal(data.positions[1], [1, 0])
        np.testing.assert_array_almost_equal(data.positions[2], [2, 1])
        np.testing.assert_array_almost_equal(data.positions[3], [3, 0])  # implicit


class TestSimulation(unittest.TestCase):
    """Tests for the simulation loop."""

    def test_simulate_fourbar(self):
        """Test simulation of a four-bar linkage."""
        ground = pl.Static(0, 0, name="ground")
        crank = pl.Crank(
            1,
            0,
            joint0=ground,
            distance=1,
            angle=0.1,
            name="crank",
        )
        pin = pl.Revolute(
            2,
            1,
            joint0=crank,
            joint1=(3, 0),
            distance0=2,
            distance1=2,
            name="pin",
        )
        linkage = pl.Linkage(
            joints=(ground, crank, pin),
            order=(ground, crank, pin),
            name="test_fourbar",
        )

        data = linkage_to_solver_data(linkage)

        # Run simulation
        trajectory = simulate(
            data.positions,
            data.constraints,
            data.joint_types,
            data.parent_indices,
            data.constraint_offsets,
            data.solve_order,
            iterations=10,
            dt=1.0,
        )

        # Check output shape (4 joints including implicit Static)
        self.assertEqual(trajectory.shape, (10, 4, 2))

        # Ground should stay fixed
        for step in range(10):
            np.testing.assert_array_almost_equal(trajectory[step, 0], [0, 0])

        # Implicit Static at (3, 0) should also stay fixed
        for step in range(10):
            np.testing.assert_array_almost_equal(trajectory[step, 3], [3, 0])

        # Crank should move (rotate)
        self.assertFalse(
            np.allclose(trajectory[0, 1], trajectory[9, 1]),
            "Crank should have moved",
        )

    def test_has_nan_positions(self):
        """Test NaN detection in trajectories."""
        # Clean trajectory
        clean = np.zeros((5, 3, 2))
        self.assertFalse(has_nan_positions(clean))

        # Trajectory with NaN
        with_nan = np.zeros((5, 3, 2))
        with_nan[2, 1, 0] = np.nan
        self.assertTrue(has_nan_positions(with_nan))


class TestStepFast(unittest.TestCase):
    """Tests for Linkage.step_fast() integration."""

    def test_step_fast_matches_step(self):
        """Test that step_fast produces same results as step."""
        ground = pl.Static(0, 0, name="ground")
        crank = pl.Crank(
            1,
            0,
            joint0=ground,
            distance=1,
            angle=0.1,
            name="crank",
        )
        pin = pl.Revolute(
            2,
            1,
            joint0=crank,
            joint1=(3, 0),
            distance0=2,
            distance1=2,
            name="pin",
        )
        linkage = pl.Linkage(
            joints=(ground, crank, pin),
            order=(ground, crank, pin),
            name="test_fourbar",
        )

        # Get initial coords
        init_coords = linkage.get_coords()

        # Run step() and collect results
        step_results = list(linkage.step(iterations=10, dt=1.0))

        # Reset positions
        linkage.set_coords(init_coords)

        # Run step_fast()
        fast_trajectory = linkage.step_fast(iterations=10, dt=1.0)

        # Compare results
        for step_idx, step_coords in enumerate(step_results):
            for joint_idx, (x, y) in enumerate(step_coords):
                if x is not None and y is not None:
                    self.assertAlmostEqual(
                        fast_trajectory[step_idx, joint_idx, 0],
                        x,
                        places=10,
                        msg=f"X mismatch at step {step_idx}, joint {joint_idx}",
                    )
                    self.assertAlmostEqual(
                        fast_trajectory[step_idx, joint_idx, 1],
                        y,
                        places=10,
                        msg=f"Y mismatch at step {step_idx}, joint {joint_idx}",
                    )

    def test_step_fast_auto_compile(self):
        """Test that step_fast auto-compiles when needed."""
        ground = pl.Static(0, 0, name="ground")
        crank = pl.Crank(
            1,
            0,
            joint0=ground,
            distance=1,
            angle=0.1,
            name="crank",
        )
        linkage = pl.Linkage(
            joints=(ground, crank),
            order=(ground, crank),
        )

        # Should not have solver data initially
        self.assertIsNone(linkage._solver_data)

        # Run step_fast - should auto-compile
        trajectory = linkage.step_fast(iterations=5)

        # Should have solver data now
        self.assertIsNotNone(linkage._solver_data)

        # Check output is valid
        self.assertEqual(trajectory.shape, (5, 2, 2))

    def test_constraint_change_invalidates_cache(self):
        """Test that changing constraints invalidates solver cache."""
        ground = pl.Static(0, 0, name="ground")
        crank = pl.Crank(
            1,
            0,
            joint0=ground,
            distance=1,
            angle=0.1,
            name="crank",
        )
        linkage = pl.Linkage(
            joints=(ground, crank),
            order=(ground, crank),
        )

        # Compile
        linkage.compile()
        self.assertIsNotNone(linkage._solver_data)

        # Change constraints
        linkage.set_num_constraints([2.0])  # Change crank radius

        # Cache should be invalidated
        self.assertIsNone(linkage._solver_data)


class TestSolverData(unittest.TestCase):
    """Tests for SolverData class."""

    def test_copy(self):
        """Test deep copy of SolverData."""
        data = SolverData(
            positions=np.array([[1.0, 2.0], [3.0, 4.0]]),
            constraints=np.array([1.0, 2.0]),
            joint_types=np.array([0, 1], dtype=np.int32),
            parent_indices=np.array([[0, -1, -1], [0, -1, -1]], dtype=np.int32),
            constraint_offsets=np.array([0, 0], dtype=np.int32),
            constraint_counts=np.array([0, 2], dtype=np.int32),
            solve_order=np.array([0, 1], dtype=np.int32),
        )

        copy = data.copy()

        # Modify original
        data.positions[0, 0] = 999.0
        data.constraints[0] = 999.0

        # Copy should be unchanged
        self.assertEqual(copy.positions[0, 0], 1.0)
        self.assertEqual(copy.constraints[0], 1.0)


if __name__ == "__main__":
    unittest.main()
