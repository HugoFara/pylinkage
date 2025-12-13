"""Tests for the pure-numba solver module."""

import math
import unittest

import numpy as np

import pylinkage as pl
from pylinkage.solver import (
    JOINT_CRANK,
    JOINT_FIXED,
    JOINT_LINEAR,
    JOINT_REVOLUTE,
    JOINT_STATIC,
    SolverData,
    first_nan_step,
    has_nan_positions,
    linkage_to_solver_data,
    simulate,
    solve_crank,
    solve_fixed,
    solve_linear,
    solve_revolute,
    step_single,
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

    def test_solve_revolute_tangent(self):
        """Test revolute at tangent point (single intersection)."""
        x, y = solve_revolute(
            current_x=1.0,
            current_y=0.0,
            p0_x=0.0,
            p0_y=0.0,
            r0=1.0,
            p1_x=2.0,
            p1_y=0.0,
            r1=1.0,
        )
        # Should be at tangent point (1, 0)
        self.assertAlmostEqual(x, 1.0, places=5)
        self.assertAlmostEqual(y, 0.0, places=5)

    def test_solve_fixed_basic(self):
        """Test basic fixed joint solving."""
        x, y = solve_fixed(
            p0_x=0.0,
            p0_y=0.0,
            p1_x=1.0,
            p1_y=0.0,
            radius=1.0,
            angle=0.0,
        )
        # Angle 0 from the line (0,0)->(1,0) should point right
        self.assertAlmostEqual(x, 1.0, places=10)
        self.assertAlmostEqual(y, 0.0, places=10)

    def test_solve_fixed_with_angle(self):
        """Test fixed joint with angle offset."""
        x, y = solve_fixed(
            p0_x=0.0,
            p0_y=0.0,
            p1_x=1.0,
            p1_y=0.0,
            radius=1.0,
            angle=math.pi / 2,
        )
        # Angle pi/2 from the line pointing right should point up
        self.assertAlmostEqual(x, 0.0, places=10)
        self.assertAlmostEqual(y, 1.0, places=10)

    def test_solve_fixed_with_offset(self):
        """Test fixed joint with offset anchor."""
        x, y = solve_fixed(
            p0_x=5.0,
            p0_y=5.0,
            p1_x=6.0,
            p1_y=5.0,
            radius=2.0,
            angle=0.0,
        )
        self.assertAlmostEqual(x, 7.0, places=10)
        self.assertAlmostEqual(y, 5.0, places=10)

    def test_solve_linear_basic(self):
        """Test basic linear joint solving."""
        # Circle at origin with radius 2, line at y=1
        x, y = solve_linear(
            current_x=0.0,
            current_y=1.0,
            circle_x=0.0,
            circle_y=0.0,
            radius=2.0,
            line_p1_x=0.0,
            line_p1_y=1.0,
            line_p2_x=5.0,
            line_p2_y=1.0,
        )
        # Should be on line y=1 at distance 2 from origin
        self.assertAlmostEqual(y, 1.0, places=10)
        # x should be sqrt(2^2 - 1^2) = sqrt(3)
        self.assertAlmostEqual(abs(x), math.sqrt(3), places=5)

    def test_solve_linear_no_intersection(self):
        """Test linear joint with no intersection returns NaN."""
        # Circle at origin with radius 1, line at y=5 (too far)
        x, y = solve_linear(
            current_x=0.0,
            current_y=5.0,
            circle_x=0.0,
            circle_y=0.0,
            radius=1.0,
            line_p1_x=0.0,
            line_p1_y=5.0,
            line_p2_x=5.0,
            line_p2_y=5.0,
        )
        self.assertTrue(math.isnan(x))
        self.assertTrue(math.isnan(y))

    def test_solve_linear_tangent(self):
        """Test linear joint at tangent (single intersection)."""
        # Circle at origin with radius 1, line at y=1 (tangent)
        x, y = solve_linear(
            current_x=0.0,
            current_y=1.0,
            circle_x=0.0,
            circle_y=0.0,
            radius=1.0,
            line_p1_x=-5.0,
            line_p1_y=1.0,
            line_p2_x=5.0,
            line_p2_y=1.0,
        )
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 1.0, places=5)


class TestConversion(unittest.TestCase):
    """Tests for linkage to solver data conversion."""

    def test_fourbar_conversion_with_fixed(self):
        """Test conversion of linkage with Fixed joint."""
        from pylinkage.bridge.solver_conversion import (
            solver_data_to_linkage,
            update_solver_constraints,
            update_solver_positions,
        )

        ground = pl.Static(0, 0, name="ground")
        ref = pl.Static(1, 0, name="ref")
        fixed = pl.Fixed(
            joint0=ground,
            joint1=ref,
            distance=1.5,
            angle=0.5,
            name="fixed",
        )
        linkage = pl.Linkage(
            joints=(ground, ref, fixed),
            order=(ground, ref, fixed),
            name="test_fixed",
        )

        data = linkage_to_solver_data(linkage)

        # Check fixed joint type
        self.assertEqual(data.joint_types[2], JOINT_FIXED)
        # Check constraints (radius, angle)
        offset = data.constraint_offsets[2]
        self.assertAlmostEqual(data.constraints[offset], 1.5)
        self.assertAlmostEqual(data.constraints[offset + 1], 0.5)

    def test_fourbar_conversion_with_prismatic(self):
        """Test conversion of linkage with Prismatic joint."""
        ground = pl.Static(0, 0, name="ground")
        line_start = pl.Static(0, 2, name="line_start")
        line_end = pl.Static(5, 2, name="line_end")
        prismatic = pl.Prismatic(
            2, 2,
            joint0=ground,
            joint1=line_start,
            joint2=line_end,
            revolute_radius=2.5,
            name="prismatic",
        )
        linkage = pl.Linkage(
            joints=(ground, line_start, line_end, prismatic),
            order=(ground, line_start, line_end, prismatic),
            name="test_prismatic",
        )

        data = linkage_to_solver_data(linkage)

        # Check linear joint type
        self.assertEqual(data.joint_types[3], JOINT_LINEAR)
        # Check constraints (revolute_radius)
        offset = data.constraint_offsets[3]
        self.assertAlmostEqual(data.constraints[offset], 2.5)
        # Check parent indices - should have 3 parents
        self.assertEqual(data.parent_indices[3, 0], 0)  # ground
        self.assertEqual(data.parent_indices[3, 1], 1)  # line_start
        self.assertEqual(data.parent_indices[3, 2], 2)  # line_end

    def test_solver_data_to_linkage(self):
        """Test updating linkage positions from solver data."""
        from pylinkage.bridge.solver_conversion import solver_data_to_linkage

        ground = pl.Static(0, 0, name="ground")
        crank = pl.Crank(1, 0, joint0=ground, distance=1, angle=0.1, name="crank")
        linkage = pl.Linkage(
            joints=(ground, crank),
            order=(ground, crank),
        )

        data = linkage_to_solver_data(linkage)
        # Modify positions in solver data
        data.positions[1, 0] = 99.0
        data.positions[1, 1] = 88.0

        # Update linkage from solver data
        solver_data_to_linkage(data, linkage)

        # Check that linkage positions were updated
        self.assertEqual(crank.x, 99.0)
        self.assertEqual(crank.y, 88.0)

    def test_update_solver_constraints(self):
        """Test updating solver constraints from linkage."""
        from pylinkage.bridge.solver_conversion import update_solver_constraints

        ground = pl.Static(0, 0, name="ground")
        crank = pl.Crank(1, 0, joint0=ground, distance=1, angle=0.1, name="crank")
        linkage = pl.Linkage(
            joints=(ground, crank),
            order=(ground, crank),
        )

        data = linkage_to_solver_data(linkage)

        # Modify linkage constraints
        crank.r = 5.0
        crank.angle = 0.5

        # Update solver data
        update_solver_constraints(data, linkage)

        # Check that solver constraints were updated
        offset = data.constraint_offsets[1]
        self.assertAlmostEqual(data.constraints[offset], 5.0)
        self.assertAlmostEqual(data.constraints[offset + 1], 0.5)

    def test_update_solver_constraints_revolute(self):
        """Test updating solver constraints for revolute joint."""
        from pylinkage.bridge.solver_conversion import update_solver_constraints

        ground = pl.Static(0, 0, name="ground")
        crank = pl.Crank(1, 0, joint0=ground, distance=1, angle=0.1, name="crank")
        pin = pl.Revolute(2, 1, joint0=crank, joint1=(3, 0), distance0=2, distance1=2, name="pin")
        linkage = pl.Linkage(
            joints=(ground, crank, pin),
            order=(ground, crank, pin),
        )

        data = linkage_to_solver_data(linkage)

        # Modify linkage constraints
        pin.r0 = 3.5
        pin.r1 = 4.5

        # Update solver data
        update_solver_constraints(data, linkage)

        # Check that solver constraints were updated
        offset = data.constraint_offsets[2]
        self.assertAlmostEqual(data.constraints[offset], 3.5)
        self.assertAlmostEqual(data.constraints[offset + 1], 4.5)

    def test_update_solver_positions(self):
        """Test updating solver positions from linkage."""
        from pylinkage.bridge.solver_conversion import update_solver_positions

        ground = pl.Static(0, 0, name="ground")
        crank = pl.Crank(1, 0, joint0=ground, distance=1, angle=0.1, name="crank")
        linkage = pl.Linkage(
            joints=(ground, crank),
            order=(ground, crank),
        )

        data = linkage_to_solver_data(linkage)

        # Modify linkage positions
        crank.x = 7.0
        crank.y = 8.0

        # Update solver data
        update_solver_positions(data, linkage)

        # Check that solver positions were updated
        self.assertAlmostEqual(data.positions[1, 0], 7.0)
        self.assertAlmostEqual(data.positions[1, 1], 8.0)

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

    def test_first_nan_step(self):
        """Test finding first NaN step."""
        # Clean trajectory - no NaN
        clean = np.zeros((5, 3, 2))
        self.assertEqual(first_nan_step(clean), -1)

        # NaN at step 3
        with_nan = np.zeros((5, 3, 2))
        with_nan[3, 0, 1] = np.nan
        self.assertEqual(first_nan_step(with_nan), 3)

        # NaN at step 0
        with_nan_early = np.zeros((5, 3, 2))
        with_nan_early[0, 2, 0] = np.nan
        self.assertEqual(first_nan_step(with_nan_early), 0)

    def test_step_single_static_joint(self):
        """Test step_single with static joint."""
        positions = np.array([[1.0, 2.0]], dtype=np.float64)
        constraints = np.array([], dtype=np.float64)
        joint_types = np.array([JOINT_STATIC], dtype=np.int32)
        parent_indices = np.array([[-1, -1, -1]], dtype=np.int32)
        constraint_offsets = np.array([0], dtype=np.int32)
        solve_order = np.array([0], dtype=np.int32)

        step_single(positions, constraints, joint_types, parent_indices,
                   constraint_offsets, solve_order, dt=1.0)

        # Static joint should not move
        np.testing.assert_array_equal(positions[0], [1.0, 2.0])

    def test_step_single_crank(self):
        """Test step_single with crank joint."""
        # Crank at (1, 0) around anchor at (0, 0)
        positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        constraints = np.array([1.0, math.pi / 2], dtype=np.float64)  # radius=1, angle_rate=pi/2
        joint_types = np.array([JOINT_STATIC, JOINT_CRANK], dtype=np.int32)
        parent_indices = np.array([[-1, -1, -1], [0, -1, -1]], dtype=np.int32)
        constraint_offsets = np.array([0, 0], dtype=np.int32)
        solve_order = np.array([0, 1], dtype=np.int32)

        step_single(positions, constraints, joint_types, parent_indices,
                   constraint_offsets, solve_order, dt=1.0)

        # After pi/2 rotation, should be at (0, 1)
        np.testing.assert_array_almost_equal(positions[1], [0.0, 1.0])

    def test_step_single_revolute(self):
        """Test step_single with revolute joint."""
        # Two static joints and a revolute
        positions = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 1.5],  # Hint toward upper intersection
        ], dtype=np.float64)
        constraints = np.array([2.0, 2.0], dtype=np.float64)  # Both radii = 2
        joint_types = np.array([JOINT_STATIC, JOINT_STATIC, JOINT_REVOLUTE], dtype=np.int32)
        parent_indices = np.array([[-1, -1, -1], [-1, -1, -1], [0, 1, -1]], dtype=np.int32)
        constraint_offsets = np.array([0, 0, 0], dtype=np.int32)
        solve_order = np.array([0, 1, 2], dtype=np.int32)

        step_single(positions, constraints, joint_types, parent_indices,
                   constraint_offsets, solve_order, dt=1.0)

        # Should be at intersection of circles
        self.assertAlmostEqual(positions[2, 0], 1.0, places=5)
        self.assertGreater(positions[2, 1], 0)  # Upper intersection

    def test_step_single_fixed(self):
        """Test step_single with fixed joint."""
        positions = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],  # Will be computed
        ], dtype=np.float64)
        constraints = np.array([1.0, math.pi / 2], dtype=np.float64)  # radius=1, angle=pi/2
        joint_types = np.array([JOINT_STATIC, JOINT_STATIC, JOINT_FIXED], dtype=np.int32)
        parent_indices = np.array([[-1, -1, -1], [-1, -1, -1], [0, 1, -1]], dtype=np.int32)
        constraint_offsets = np.array([0, 0, 0], dtype=np.int32)
        solve_order = np.array([0, 1, 2], dtype=np.int32)

        step_single(positions, constraints, joint_types, parent_indices,
                   constraint_offsets, solve_order, dt=1.0)

        # Should be at (0, 1) - 90 degrees from line pointing to (1,0)
        np.testing.assert_array_almost_equal(positions[2], [0.0, 1.0])

    def test_step_single_linear(self):
        """Test step_single with linear joint."""
        positions = np.array([
            [0.0, 0.0],  # Circle center
            [0.0, 1.0],  # Line point 1
            [5.0, 1.0],  # Line point 2
            [0.0, 1.0],  # Linear joint (hint)
        ], dtype=np.float64)
        constraints = np.array([2.0], dtype=np.float64)  # radius=2
        joint_types = np.array([JOINT_STATIC, JOINT_STATIC, JOINT_STATIC, JOINT_LINEAR], dtype=np.int32)
        parent_indices = np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [0, 1, 2]], dtype=np.int32)
        constraint_offsets = np.array([0, 0, 0, 0], dtype=np.int32)
        solve_order = np.array([0, 1, 2, 3], dtype=np.int32)

        step_single(positions, constraints, joint_types, parent_indices,
                   constraint_offsets, solve_order, dt=1.0)

        # Should be on line y=1, at distance 2 from origin
        self.assertAlmostEqual(positions[3, 1], 1.0, places=5)


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
