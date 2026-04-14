"""Extended tests for simulation.py.

Covers: step_single_velocity, step_single_acceleration,
simulate_with_kinematics, and additional simulation scenarios.
"""

import math

import numpy as np
import pytest

import pylinkage as pl
from pylinkage.solver import (
    JOINT_CRANK,
    JOINT_FIXED,
    JOINT_LINEAR,
    JOINT_REVOLUTE,
    JOINT_STATIC,
    simulate,
    simulate_with_kinematics,
    step_single,
    step_single_acceleration,
    step_single_velocity,
)


class TestStepSingleVelocity:
    """Tests for step_single_velocity."""

    def _make_crank_arrays(self):
        """Helper to create arrays for a simple crank mechanism."""
        positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        velocities = np.zeros((2, 2), dtype=np.float64)
        constraints = np.array([1.0, math.pi / 2], dtype=np.float64)
        joint_types = np.array([JOINT_STATIC, JOINT_CRANK], dtype=np.int32)
        parent_indices = np.array([[-1, -1, -1], [0, -1, -1]], dtype=np.int32)
        constraint_offsets = np.array([0, 0], dtype=np.int32)
        solve_order = np.array([0, 1], dtype=np.int32)
        omega_values = np.array([5.0], dtype=np.float64)
        crank_indices = np.array([1], dtype=np.int32)
        return (
            positions,
            velocities,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            crank_indices,
        )

    def test_static_joint_zero_velocity(self):
        """Static joints have zero velocity."""
        (
            positions,
            velocities,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            crank_indices,
        ) = self._make_crank_arrays()

        step_single_velocity(
            positions,
            velocities,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            crank_indices,
        )

        np.testing.assert_array_almost_equal(velocities[0], [0.0, 0.0])

    def test_crank_velocity_magnitude(self):
        """Crank velocity magnitude = r * omega."""
        (
            positions,
            velocities,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            crank_indices,
        ) = self._make_crank_arrays()

        step_single_velocity(
            positions,
            velocities,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            crank_indices,
        )

        mag = math.sqrt(velocities[1, 0] ** 2 + velocities[1, 1] ** 2)
        assert mag == pytest.approx(5.0, abs=1e-10)  # r=1 * omega=5

    def test_revolute_velocity(self):
        """Revolute joint velocity is computed without NaN."""
        # Static0, Static1, Crank, Revolute
        positions = np.array(
            [
                [0.0, 0.0],  # Static anchor for crank
                [3.0, 0.0],  # Static anchor for revolute
                [1.0, 0.0],  # Crank
                [2.0, 1.5],  # Revolute (hint)
            ],
            dtype=np.float64,
        )
        velocities = np.zeros((4, 2), dtype=np.float64)
        # Crank: radius=1, angle_rate=0.1
        # Revolute: r0=2, r1=2
        constraints = np.array([1.0, 0.1, 2.0, 2.0], dtype=np.float64)
        joint_types = np.array(
            [JOINT_STATIC, JOINT_STATIC, JOINT_CRANK, JOINT_REVOLUTE], dtype=np.int32
        )
        parent_indices = np.array(
            [
                [-1, -1, -1],
                [-1, -1, -1],
                [0, -1, -1],
                [2, 1, -1],
            ],
            dtype=np.int32,
        )
        constraint_offsets = np.array([0, 0, 0, 2], dtype=np.int32)
        solve_order = np.array([0, 1, 2, 3], dtype=np.int32)
        omega_values = np.array([5.0], dtype=np.float64)
        crank_indices = np.array([2], dtype=np.int32)

        # First solve positions
        step_single(
            positions,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            dt=1.0,
        )

        # Then solve velocities
        step_single_velocity(
            positions,
            velocities,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            crank_indices,
        )

        # Static joints should be zero
        np.testing.assert_array_almost_equal(velocities[0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(velocities[1], [0.0, 0.0])
        # Revolute should have non-NaN velocity
        assert not math.isnan(velocities[3, 0])
        assert not math.isnan(velocities[3, 1])

    def test_fixed_joint_velocity(self):
        """Fixed joint velocity is computed."""
        positions = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],  # Will be solved
            ],
            dtype=np.float64,
        )
        velocities = np.zeros((3, 2), dtype=np.float64)
        constraints = np.array([1.0, math.pi / 2], dtype=np.float64)
        joint_types = np.array([JOINT_STATIC, JOINT_STATIC, JOINT_FIXED], dtype=np.int32)
        parent_indices = np.array([[-1, -1, -1], [-1, -1, -1], [0, 1, -1]], dtype=np.int32)
        constraint_offsets = np.array([0, 0, 0], dtype=np.int32)
        solve_order = np.array([0, 1, 2], dtype=np.int32)
        omega_values = np.array([], dtype=np.float64)
        crank_indices = np.array([], dtype=np.int32)

        # Solve positions first
        step_single(
            positions,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            dt=1.0,
        )

        # Solve velocities
        step_single_velocity(
            positions,
            velocities,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            crank_indices,
        )

        # Both parents static -> fixed joint velocity should be zero
        np.testing.assert_array_almost_equal(velocities[2], [0.0, 0.0])

    def test_linear_joint_velocity(self):
        """Linear (prismatic) joint velocity is computed."""
        positions = np.array(
            [
                [0.0, 0.0],  # Circle center
                [0.0, 1.0],  # Line point 1
                [5.0, 1.0],  # Line point 2
                [0.0, 1.0],  # Linear joint (hint)
            ],
            dtype=np.float64,
        )
        velocities = np.zeros((4, 2), dtype=np.float64)
        constraints = np.array([2.0], dtype=np.float64)
        joint_types = np.array(
            [JOINT_STATIC, JOINT_STATIC, JOINT_STATIC, JOINT_LINEAR], dtype=np.int32
        )
        parent_indices = np.array(
            [
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [0, 1, 2],
            ],
            dtype=np.int32,
        )
        constraint_offsets = np.array([0, 0, 0, 0], dtype=np.int32)
        solve_order = np.array([0, 1, 2, 3], dtype=np.int32)
        omega_values = np.array([], dtype=np.float64)
        crank_indices = np.array([], dtype=np.int32)

        # Solve positions first
        step_single(
            positions,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            dt=1.0,
        )

        # Solve velocities
        step_single_velocity(
            positions,
            velocities,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            crank_indices,
        )

        # All parents static -> linear joint velocity should be zero
        np.testing.assert_array_almost_equal(velocities[3], [0.0, 0.0])


class TestStepSingleAcceleration:
    """Tests for step_single_acceleration."""

    def test_static_joint_zero_acceleration(self):
        """Static joints have zero acceleration."""
        positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        velocities = np.zeros((2, 2), dtype=np.float64)
        accelerations = np.zeros((2, 2), dtype=np.float64)
        constraints = np.array([1.0, 0.1], dtype=np.float64)
        joint_types = np.array([JOINT_STATIC, JOINT_CRANK], dtype=np.int32)
        parent_indices = np.array([[-1, -1, -1], [0, -1, -1]], dtype=np.int32)
        constraint_offsets = np.array([0, 0], dtype=np.int32)
        solve_order = np.array([0, 1], dtype=np.int32)
        omega_values = np.array([5.0], dtype=np.float64)
        alpha_values = np.array([0.0], dtype=np.float64)
        crank_indices = np.array([1], dtype=np.int32)

        step_single_acceleration(
            positions,
            velocities,
            accelerations,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            alpha_values,
            crank_indices,
        )

        np.testing.assert_array_almost_equal(accelerations[0], [0.0, 0.0])

    def test_crank_centripetal_acceleration(self):
        """Crank at (1,0) with omega=10 has centripetal accel = -r*omega^2."""
        positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        velocities = np.array([[0.0, 0.0], [0.0, 10.0]], dtype=np.float64)
        accelerations = np.zeros((2, 2), dtype=np.float64)
        constraints = np.array([1.0, 0.1], dtype=np.float64)
        joint_types = np.array([JOINT_STATIC, JOINT_CRANK], dtype=np.int32)
        parent_indices = np.array([[-1, -1, -1], [0, -1, -1]], dtype=np.int32)
        constraint_offsets = np.array([0, 0], dtype=np.int32)
        solve_order = np.array([0, 1], dtype=np.int32)
        omega_values = np.array([10.0], dtype=np.float64)
        alpha_values = np.array([0.0], dtype=np.float64)
        crank_indices = np.array([1], dtype=np.int32)

        step_single_acceleration(
            positions,
            velocities,
            accelerations,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            alpha_values,
            crank_indices,
        )

        # Centripetal: -r*omega^2*(cos0, sin0) = (-100, 0)
        assert accelerations[1, 0] == pytest.approx(-100.0, abs=1e-10)
        assert accelerations[1, 1] == pytest.approx(0.0, abs=1e-10)

    def test_revolute_acceleration(self):
        """Revolute joint acceleration is computed without NaN."""
        positions = np.array(
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [1.0, 0.0],
                [2.0, 1.5],
            ],
            dtype=np.float64,
        )
        velocities = np.zeros((4, 2), dtype=np.float64)
        accelerations = np.zeros((4, 2), dtype=np.float64)
        constraints = np.array([1.0, 0.1, 2.0, 2.0], dtype=np.float64)
        joint_types = np.array(
            [JOINT_STATIC, JOINT_STATIC, JOINT_CRANK, JOINT_REVOLUTE], dtype=np.int32
        )
        parent_indices = np.array(
            [
                [-1, -1, -1],
                [-1, -1, -1],
                [0, -1, -1],
                [2, 1, -1],
            ],
            dtype=np.int32,
        )
        constraint_offsets = np.array([0, 0, 0, 2], dtype=np.int32)
        solve_order = np.array([0, 1, 2, 3], dtype=np.int32)
        omega_values = np.array([5.0], dtype=np.float64)
        alpha_values = np.array([0.0], dtype=np.float64)
        crank_indices = np.array([2], dtype=np.int32)

        # Solve positions
        step_single(
            positions,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            dt=1.0,
        )
        # Solve velocities
        step_single_velocity(
            positions,
            velocities,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            crank_indices,
        )
        # Solve accelerations
        step_single_acceleration(
            positions,
            velocities,
            accelerations,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            alpha_values,
            crank_indices,
        )

        # Should have non-NaN accelerations for all joints
        for j in range(4):
            assert not math.isnan(accelerations[j, 0]), f"Joint {j} ax is NaN"
            assert not math.isnan(accelerations[j, 1]), f"Joint {j} ay is NaN"

    def test_fixed_joint_acceleration(self):
        """Fixed joint acceleration is computed."""
        positions = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        velocities = np.zeros((3, 2), dtype=np.float64)
        accelerations = np.zeros((3, 2), dtype=np.float64)
        constraints = np.array([1.0, math.pi / 2], dtype=np.float64)
        joint_types = np.array([JOINT_STATIC, JOINT_STATIC, JOINT_FIXED], dtype=np.int32)
        parent_indices = np.array([[-1, -1, -1], [-1, -1, -1], [0, 1, -1]], dtype=np.int32)
        constraint_offsets = np.array([0, 0, 0], dtype=np.int32)
        solve_order = np.array([0, 1, 2], dtype=np.int32)
        omega_values = np.array([], dtype=np.float64)
        alpha_values = np.array([], dtype=np.float64)
        crank_indices = np.array([], dtype=np.int32)

        # Solve positions first
        step_single(
            positions,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            dt=1.0,
        )
        step_single_velocity(
            positions,
            velocities,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            crank_indices,
        )
        step_single_acceleration(
            positions,
            velocities,
            accelerations,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            alpha_values,
            crank_indices,
        )

        # All static parents -> zero acceleration
        np.testing.assert_array_almost_equal(accelerations[2], [0.0, 0.0])

    def test_linear_joint_acceleration(self):
        """Linear joint acceleration is computed."""
        positions = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [5.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        velocities = np.zeros((4, 2), dtype=np.float64)
        accelerations = np.zeros((4, 2), dtype=np.float64)
        constraints = np.array([2.0], dtype=np.float64)
        joint_types = np.array(
            [JOINT_STATIC, JOINT_STATIC, JOINT_STATIC, JOINT_LINEAR], dtype=np.int32
        )
        parent_indices = np.array(
            [
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [0, 1, 2],
            ],
            dtype=np.int32,
        )
        constraint_offsets = np.array([0, 0, 0, 0], dtype=np.int32)
        solve_order = np.array([0, 1, 2, 3], dtype=np.int32)
        omega_values = np.array([], dtype=np.float64)
        alpha_values = np.array([], dtype=np.float64)
        crank_indices = np.array([], dtype=np.int32)

        step_single(
            positions,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            dt=1.0,
        )
        step_single_velocity(
            positions,
            velocities,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            crank_indices,
        )
        step_single_acceleration(
            positions,
            velocities,
            accelerations,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            alpha_values,
            crank_indices,
        )

        # All static parents -> zero acceleration
        np.testing.assert_array_almost_equal(accelerations[3], [0.0, 0.0])


class TestSimulateWithKinematics:
    """Tests for simulate_with_kinematics."""

    def test_basic_crank_simulation(self):
        """Simulate a crank and verify kinematics arrays are populated."""
        positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        velocities = np.zeros((2, 2), dtype=np.float64)
        accelerations = np.zeros((2, 2), dtype=np.float64)
        constraints = np.array([1.0, 0.1], dtype=np.float64)
        joint_types = np.array([JOINT_STATIC, JOINT_CRANK], dtype=np.int32)
        parent_indices = np.array([[-1, -1, -1], [0, -1, -1]], dtype=np.int32)
        constraint_offsets = np.array([0, 0], dtype=np.int32)
        solve_order = np.array([0, 1], dtype=np.int32)
        omega_values = np.array([5.0], dtype=np.float64)
        alpha_values = np.array([0.0], dtype=np.float64)
        crank_indices = np.array([1], dtype=np.int32)

        pos_traj, vel_traj, acc_traj = simulate_with_kinematics(
            positions,
            velocities,
            accelerations,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            alpha_values,
            crank_indices,
            iterations=10,
            dt=1.0,
        )

        assert pos_traj.shape == (10, 2, 2)
        assert vel_traj.shape == (10, 2, 2)
        assert acc_traj.shape == (10, 2, 2)

        # Static joint stays at origin
        for step in range(10):
            np.testing.assert_array_almost_equal(pos_traj[step, 0], [0.0, 0.0])

        # Crank moves
        assert not np.allclose(pos_traj[0, 1], pos_traj[9, 1])

        # Crank velocity magnitude should be constant (r * omega = 1 * 5 = 5)
        for step in range(10):
            vel_mag = math.sqrt(vel_traj[step, 1, 0] ** 2 + vel_traj[step, 1, 1] ** 2)
            assert vel_mag == pytest.approx(5.0, abs=1e-5)

        # Centripetal acceleration magnitude should be r * omega^2 = 25
        for step in range(10):
            acc_mag = math.sqrt(acc_traj[step, 1, 0] ** 2 + acc_traj[step, 1, 1] ** 2)
            assert acc_mag == pytest.approx(25.0, abs=1e-5)

    def test_fourbar_simulation_with_kinematics(self):
        """Simulate a four-bar linkage with full kinematics."""
        # NB: the link ratios must be unambiguously Grashof (s + L < p + q).
        # A change-point mechanism (s + L = p + q) passes through singular
        # configurations where the velocity Jacobian is rank-deficient, which
        # non-deterministically produces NaN in velocities/accelerations —
        # this was the cause of a long-standing CI-only flake on this test.
        ground = pl.Static(0, 0, name="ground")
        crank = pl.Crank(1, 0, joint0=ground, distance=1, angle=0.1, omega=5.0, name="crank")
        pin = pl.Revolute(
            2,
            1,
            joint0=crank,
            joint1=(3, 0),
            distance0=2,
            distance1=2.5,
            name="pin",
        )
        linkage = pl.Linkage(
            joints=(ground, crank, pin),
            order=(ground, crank, pin),
            name="test_fourbar",
        )

        positions, velocities, accelerations = linkage.step_fast_with_kinematics(
            iterations=20,
            dt=1.0,
        )

        # Shape: 4 joints (including implicit Static at (3,0))
        assert positions.shape == (20, 4, 2)
        assert velocities.shape == (20, 4, 2)
        assert accelerations.shape == (20, 4, 2)

        # No NaN in trajectory
        assert not np.any(np.isnan(positions))
        assert not np.any(np.isnan(velocities))
        assert not np.any(np.isnan(accelerations))


class TestSimulateEdgeCases:
    """Edge cases for the simulate function."""

    def test_single_iteration(self):
        """Simulation with a single iteration."""
        positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        constraints = np.array([1.0, math.pi / 2], dtype=np.float64)
        joint_types = np.array([JOINT_STATIC, JOINT_CRANK], dtype=np.int32)
        parent_indices = np.array([[-1, -1, -1], [0, -1, -1]], dtype=np.int32)
        constraint_offsets = np.array([0, 0], dtype=np.int32)
        solve_order = np.array([0, 1], dtype=np.int32)

        trajectory = simulate(
            positions,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            iterations=1,
            dt=1.0,
        )

        assert trajectory.shape == (1, 2, 2)
        np.testing.assert_array_almost_equal(trajectory[0, 0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(trajectory[0, 1], [0.0, 1.0])

    def test_static_only_simulation(self):
        """Simulation with only static joints (nothing moves)."""
        positions = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        constraints = np.array([], dtype=np.float64)
        joint_types = np.array([JOINT_STATIC, JOINT_STATIC], dtype=np.int32)
        parent_indices = np.array([[-1, -1, -1], [-1, -1, -1]], dtype=np.int32)
        constraint_offsets = np.array([0, 0], dtype=np.int32)
        solve_order = np.array([0, 1], dtype=np.int32)

        trajectory = simulate(
            positions,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            iterations=5,
            dt=1.0,
        )

        assert trajectory.shape == (5, 2, 2)
        for step in range(5):
            np.testing.assert_array_almost_equal(trajectory[step, 0], [1.0, 2.0])
            np.testing.assert_array_almost_equal(trajectory[step, 1], [3.0, 4.0])

    def test_small_dt(self):
        """Small dt produces small movements per step."""
        positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        constraints = np.array([1.0, 1.0], dtype=np.float64)
        joint_types = np.array([JOINT_STATIC, JOINT_CRANK], dtype=np.int32)
        parent_indices = np.array([[-1, -1, -1], [0, -1, -1]], dtype=np.int32)
        constraint_offsets = np.array([0, 0], dtype=np.int32)
        solve_order = np.array([0, 1], dtype=np.int32)

        trajectory = simulate(
            positions,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            iterations=5,
            dt=0.001,
        )

        # With dt=0.001 and angle_rate=1, angle change per step = 0.001 rad
        # After 5 steps, crank should barely move from (1,0)
        assert trajectory[4, 1, 0] == pytest.approx(1.0, abs=0.01)

    def test_unbuildable_produces_nan(self):
        """Unbuildable revolute should produce NaN in trajectory."""
        # Two static joints far apart, revolute with short radii
        positions = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [50.0, 0.0],  # hint
            ],
            dtype=np.float64,
        )
        constraints = np.array([1.0, 1.0], dtype=np.float64)  # radii too short
        joint_types = np.array([JOINT_STATIC, JOINT_STATIC, JOINT_REVOLUTE], dtype=np.int32)
        parent_indices = np.array(
            [
                [-1, -1, -1],
                [-1, -1, -1],
                [0, 1, -1],
            ],
            dtype=np.int32,
        )
        constraint_offsets = np.array([0, 0, 0], dtype=np.int32)
        solve_order = np.array([0, 1, 2], dtype=np.int32)

        trajectory = simulate(
            positions,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            iterations=3,
            dt=1.0,
        )

        # Revolute joint should be NaN
        assert np.isnan(trajectory[0, 2, 0])
        assert np.isnan(trajectory[0, 2, 1])
