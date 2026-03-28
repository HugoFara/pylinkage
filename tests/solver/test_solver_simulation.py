"""Tests for solver/simulation.py to increase code coverage.

Covers all functions: step_single, simulate, has_nan_positions,
first_nan_step, step_single_velocity, step_single_acceleration,
and simulate_with_kinematics.

Uses a four-bar linkage as the primary test mechanism:
- Joint 0: Static at (0, 0)
- Joint 1: Crank at (2, 0), anchored to joint 0, radius=2, angle_rate=0.1
- Joint 2: Static at (8, 0)
- Joint 3: Revolute connected to joints 1 and 2, distances 7 and 5
"""

import math

import numpy as np
import pytest

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
from pylinkage.solver.simulation import first_nan_step, has_nan_positions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fourbar_arrays():
    """Build arrays for a four-bar linkage.

    Layout:
        Joint 0 (Static):   (0, 0)
        Joint 1 (Crank):    (2, 0), parent=0, radius=2, angle_rate=0.1
        Joint 2 (Static):   (8, 0)
        Joint 3 (Revolute): parents=1,2, r0=7, r1=5, hint=(5, 4)
    """
    positions = np.array(
        [[0.0, 0.0], [2.0, 0.0], [8.0, 0.0], [5.0, 4.0]],
        dtype=np.float64,
    )
    joint_types = np.array(
        [JOINT_STATIC, JOINT_CRANK, JOINT_STATIC, JOINT_REVOLUTE],
        dtype=np.int32,
    )
    parent_indices = np.array(
        [[-1, -1, -1], [0, -1, -1], [-1, -1, -1], [1, 2, -1]],
        dtype=np.int32,
    )
    # Constraints: crank [radius=2, angle_rate=0.1], revolute [r0=7, r1=5]
    constraints = np.array([2.0, 0.1, 7.0, 5.0], dtype=np.float64)
    constraint_offsets = np.array([0, 0, 0, 2], dtype=np.int32)
    solve_order = np.array([1, 3], dtype=np.int32)
    return (
        positions,
        constraints,
        joint_types,
        parent_indices,
        constraint_offsets,
        solve_order,
    )


def _fourbar_kinematics_arrays():
    """Build arrays for a four-bar with kinematics support."""
    (
        positions,
        constraints,
        joint_types,
        parent_indices,
        constraint_offsets,
        solve_order,
    ) = _fourbar_arrays()
    n = positions.shape[0]
    velocities = np.zeros((n, 2), dtype=np.float64)
    accelerations = np.zeros((n, 2), dtype=np.float64)
    omega_values = np.array([0.1], dtype=np.float64)
    alpha_values = np.array([0.0], dtype=np.float64)
    crank_indices = np.array([1], dtype=np.int32)
    return (
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


def _fixed_joint_arrays():
    """Build arrays for a mechanism containing a JOINT_FIXED.

    Layout:
        Joint 0 (Static):  (0, 0)
        Joint 1 (Crank):   (1, 0), parent=0, radius=1, angle_rate=0.2
        Joint 2 (Static):  (4, 0)
        Joint 3 (Fixed):   parents=1,2, radius=2.5, angle=pi/6
    """
    positions = np.array(
        [[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [2.0, 1.0]],
        dtype=np.float64,
    )
    joint_types = np.array(
        [JOINT_STATIC, JOINT_CRANK, JOINT_STATIC, JOINT_FIXED],
        dtype=np.int32,
    )
    parent_indices = np.array(
        [[-1, -1, -1], [0, -1, -1], [-1, -1, -1], [1, 2, -1]],
        dtype=np.int32,
    )
    constraints = np.array(
        [1.0, 0.2, 2.5, math.pi / 6],
        dtype=np.float64,
    )
    constraint_offsets = np.array([0, 0, 0, 2], dtype=np.int32)
    solve_order = np.array([1, 3], dtype=np.int32)
    return (
        positions,
        constraints,
        joint_types,
        parent_indices,
        constraint_offsets,
        solve_order,
    )


def _linear_joint_arrays():
    """Build arrays for a mechanism containing a JOINT_LINEAR.

    Layout:
        Joint 0 (Static):  (0, 0)  -- circle center
        Joint 1 (Crank):   (1, 0), parent=0, radius=1, angle_rate=0.3
        Joint 2 (Static):  (-5, 2) -- line point 1
        Joint 3 (Static):  (5, 2)  -- line point 2
        Joint 4 (Linear):  parents=1,2,3, radius=3
    """
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-5.0, 2.0],
            [5.0, 2.0],
            [1.0, 2.0],  # hint
        ],
        dtype=np.float64,
    )
    joint_types = np.array(
        [JOINT_STATIC, JOINT_CRANK, JOINT_STATIC, JOINT_STATIC, JOINT_LINEAR],
        dtype=np.int32,
    )
    parent_indices = np.array(
        [
            [-1, -1, -1],
            [0, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )
    # Crank: radius=1, angle_rate=0.3; Linear: radius=3
    constraints = np.array([1.0, 0.3, 3.0], dtype=np.float64)
    constraint_offsets = np.array([0, 0, 0, 0, 2], dtype=np.int32)
    solve_order = np.array([1, 4], dtype=np.int32)
    return (
        positions,
        constraints,
        joint_types,
        parent_indices,
        constraint_offsets,
        solve_order,
    )


# ===========================================================================
# step_single tests
# ===========================================================================


class TestStepSingleFourBar:
    """step_single with a four-bar linkage."""

    def test_static_joints_unchanged(self):
        pos, con, jt, pi, co, so = _fourbar_arrays()
        orig0 = pos[0].copy()
        orig2 = pos[2].copy()
        step_single(pos, con, jt, pi, co, so, dt=1.0)
        np.testing.assert_array_equal(pos[0], orig0)
        np.testing.assert_array_equal(pos[2], orig2)

    def test_crank_rotates(self):
        pos, con, jt, pi, co, so = _fourbar_arrays()
        step_single(pos, con, jt, pi, co, so, dt=1.0)
        # Crank should have rotated by 0.1 rad from angle=0 (starting at (2,0))
        expected_angle = 0.1
        np.testing.assert_almost_equal(pos[1, 0], 2.0 * math.cos(expected_angle))
        np.testing.assert_almost_equal(pos[1, 1], 2.0 * math.sin(expected_angle))

    def test_revolute_solved(self):
        pos, con, jt, pi, co, so = _fourbar_arrays()
        step_single(pos, con, jt, pi, co, so, dt=1.0)
        # Revolute at index 3 should satisfy distance constraints
        d_to_p1 = np.linalg.norm(pos[3] - pos[1])
        d_to_p2 = np.linalg.norm(pos[3] - pos[2])
        assert d_to_p1 == pytest.approx(7.0, abs=1e-8)
        assert d_to_p2 == pytest.approx(5.0, abs=1e-8)

    def test_revolute_no_nan(self):
        pos, con, jt, pi, co, so = _fourbar_arrays()
        step_single(pos, con, jt, pi, co, so, dt=1.0)
        assert not np.any(np.isnan(pos))

    def test_multiple_steps_stay_valid(self):
        pos, con, jt, pi, co, so = _fourbar_arrays()
        for _ in range(20):
            step_single(pos, con, jt, pi, co, so, dt=0.05)
        assert not np.any(np.isnan(pos))
        # Distance constraints should still hold
        d_to_p1 = np.linalg.norm(pos[3] - pos[1])
        d_to_p2 = np.linalg.norm(pos[3] - pos[2])
        assert d_to_p1 == pytest.approx(7.0, abs=1e-6)
        assert d_to_p2 == pytest.approx(5.0, abs=1e-6)


class TestStepSingleFixed:
    """step_single with a JOINT_FIXED mechanism."""

    def test_fixed_position_deterministic(self):
        pos, con, jt, pi, co, so = _fixed_joint_arrays()
        step_single(pos, con, jt, pi, co, so, dt=1.0)
        # Fixed joint position is determined from parents and constraints
        # After one step the crank rotates by 0.2 rad
        crank_angle = 0.2
        crank_pos = np.array([math.cos(crank_angle), math.sin(crank_angle)])
        static_pos = np.array([4.0, 0.0])
        base_angle = math.atan2(
            static_pos[1] - crank_pos[1], static_pos[0] - crank_pos[0]
        )
        expected = crank_pos + 2.5 * np.array(
            [math.cos(math.pi / 6 + base_angle), math.sin(math.pi / 6 + base_angle)]
        )
        np.testing.assert_almost_equal(pos[3], expected, decimal=10)

    def test_fixed_no_nan(self):
        pos, con, jt, pi, co, so = _fixed_joint_arrays()
        for _ in range(10):
            step_single(pos, con, jt, pi, co, so, dt=0.1)
        assert not np.any(np.isnan(pos))


class TestStepSingleLinear:
    """step_single with a JOINT_LINEAR mechanism."""

    def test_linear_on_line(self):
        pos, con, jt, pi, co, so = _linear_joint_arrays()
        step_single(pos, con, jt, pi, co, so, dt=1.0)
        # The linear joint should be on the line y=2
        assert pos[4, 1] == pytest.approx(2.0, abs=1e-8)

    def test_linear_at_correct_distance(self):
        pos, con, jt, pi, co, so = _linear_joint_arrays()
        step_single(pos, con, jt, pi, co, so, dt=1.0)
        d = np.linalg.norm(pos[4] - pos[1])
        assert d == pytest.approx(3.0, abs=1e-8)

    def test_linear_multiple_steps(self):
        pos, con, jt, pi, co, so = _linear_joint_arrays()
        for _ in range(15):
            step_single(pos, con, jt, pi, co, so, dt=0.1)
        # Should still be on line y=2 and at distance 3 from crank
        assert pos[4, 1] == pytest.approx(2.0, abs=1e-6)
        d = np.linalg.norm(pos[4] - pos[1])
        assert d == pytest.approx(3.0, abs=1e-6)


# ===========================================================================
# simulate tests
# ===========================================================================


class TestSimulateFourBar:
    """simulate() with the four-bar linkage."""

    def test_trajectory_shape(self):
        pos, con, jt, pi, co, so = _fourbar_arrays()
        traj = simulate(pos, con, jt, pi, co, so, iterations=10, dt=0.1)
        assert traj.shape == (10, 4, 2)

    def test_static_joints_constant(self):
        pos, con, jt, pi, co, so = _fourbar_arrays()
        traj = simulate(pos, con, jt, pi, co, so, iterations=10, dt=0.1)
        for step in range(10):
            np.testing.assert_array_almost_equal(traj[step, 0], [0.0, 0.0])
            np.testing.assert_array_almost_equal(traj[step, 2], [8.0, 0.0])

    def test_crank_moves(self):
        pos, con, jt, pi, co, so = _fourbar_arrays()
        traj = simulate(pos, con, jt, pi, co, so, iterations=10, dt=0.1)
        assert not np.allclose(traj[0, 1], traj[9, 1])

    def test_revolute_distances_maintained(self):
        pos, con, jt, pi, co, so = _fourbar_arrays()
        traj = simulate(pos, con, jt, pi, co, so, iterations=20, dt=0.05)
        for step in range(20):
            d0 = np.linalg.norm(traj[step, 3] - traj[step, 1])
            d1 = np.linalg.norm(traj[step, 3] - traj[step, 2])
            assert d0 == pytest.approx(7.0, abs=1e-6)
            assert d1 == pytest.approx(5.0, abs=1e-6)

    def test_no_nan_in_trajectory(self):
        pos, con, jt, pi, co, so = _fourbar_arrays()
        traj = simulate(pos, con, jt, pi, co, so, iterations=30, dt=0.05)
        assert not np.any(np.isnan(traj))

    def test_positions_updated_in_place(self):
        """After simulate(), positions array holds final state."""
        pos, con, jt, pi, co, so = _fourbar_arrays()
        traj = simulate(pos, con, jt, pi, co, so, iterations=5, dt=0.1)
        np.testing.assert_array_almost_equal(pos, traj[-1])

    def test_single_iteration(self):
        pos, con, jt, pi, co, so = _fourbar_arrays()
        traj = simulate(pos, con, jt, pi, co, so, iterations=1, dt=0.1)
        assert traj.shape == (1, 4, 2)
        assert not np.any(np.isnan(traj))


class TestSimulateFixed:
    """simulate() with a JOINT_FIXED mechanism."""

    def test_fixed_trajectory_valid(self):
        pos, con, jt, pi, co, so = _fixed_joint_arrays()
        traj = simulate(pos, con, jt, pi, co, so, iterations=10, dt=0.1)
        assert traj.shape == (10, 4, 2)
        assert not np.any(np.isnan(traj))


class TestSimulateLinear:
    """simulate() with a JOINT_LINEAR mechanism."""

    def test_linear_trajectory_valid(self):
        pos, con, jt, pi, co, so = _linear_joint_arrays()
        traj = simulate(pos, con, jt, pi, co, so, iterations=10, dt=0.1)
        assert traj.shape == (10, 5, 2)
        assert not np.any(np.isnan(traj))

    def test_linear_stays_on_line(self):
        pos, con, jt, pi, co, so = _linear_joint_arrays()
        traj = simulate(pos, con, jt, pi, co, so, iterations=10, dt=0.1)
        for step in range(10):
            assert traj[step, 4, 1] == pytest.approx(2.0, abs=1e-6)


# ===========================================================================
# has_nan_positions / first_nan_step tests
# ===========================================================================


class TestHasNanPositions:
    """Tests for has_nan_positions()."""

    def test_clean_trajectory(self):
        traj = np.ones((5, 3, 2), dtype=np.float64)
        assert has_nan_positions(traj) is False

    def test_nan_in_x(self):
        traj = np.ones((5, 3, 2), dtype=np.float64)
        traj[2, 1, 0] = np.nan
        assert has_nan_positions(traj) is True

    def test_nan_in_y(self):
        traj = np.ones((5, 3, 2), dtype=np.float64)
        traj[4, 0, 1] = np.nan
        assert has_nan_positions(traj) is True

    def test_all_nan(self):
        traj = np.full((3, 2, 2), np.nan, dtype=np.float64)
        assert has_nan_positions(traj) is True

    def test_empty_like_trajectory(self):
        traj = np.zeros((0, 0, 2), dtype=np.float64)
        assert has_nan_positions(traj) is False

    def test_single_step(self):
        traj = np.zeros((1, 1, 2), dtype=np.float64)
        assert has_nan_positions(traj) is False

    def test_from_fourbar_simulation(self):
        pos, con, jt, pi, co, so = _fourbar_arrays()
        traj = simulate(pos, con, jt, pi, co, so, iterations=10, dt=0.05)
        assert has_nan_positions(traj) is False


class TestFirstNanStep:
    """Tests for first_nan_step()."""

    def test_no_nan_returns_minus_one(self):
        traj = np.ones((10, 4, 2), dtype=np.float64)
        assert first_nan_step(traj) == -1

    def test_nan_at_step_zero(self):
        traj = np.ones((5, 3, 2), dtype=np.float64)
        traj[0, 0, 0] = np.nan
        assert first_nan_step(traj) == 0

    def test_nan_at_last_step(self):
        traj = np.ones((5, 3, 2), dtype=np.float64)
        traj[4, 2, 1] = np.nan
        assert first_nan_step(traj) == 4

    def test_nan_at_middle_step(self):
        traj = np.ones((10, 2, 2), dtype=np.float64)
        traj[6, 0, 0] = np.nan
        assert first_nan_step(traj) == 6

    def test_multiple_nan_returns_first(self):
        traj = np.ones((10, 2, 2), dtype=np.float64)
        traj[3, 0, 0] = np.nan
        traj[7, 1, 1] = np.nan
        assert first_nan_step(traj) == 3

    def test_unbuildable_mechanism(self):
        """Revolute with impossible distances produces NaN early."""
        positions = np.array(
            [[0.0, 0.0], [100.0, 0.0], [50.0, 0.0]],
            dtype=np.float64,
        )
        constraints = np.array([1.0, 1.0], dtype=np.float64)
        joint_types = np.array(
            [JOINT_STATIC, JOINT_STATIC, JOINT_REVOLUTE], dtype=np.int32
        )
        parent_indices = np.array(
            [[-1, -1, -1], [-1, -1, -1], [0, 1, -1]], dtype=np.int32
        )
        constraint_offsets = np.array([0, 0, 0], dtype=np.int32)
        solve_order = np.array([0, 1, 2], dtype=np.int32)

        traj = simulate(
            positions, constraints, joint_types, parent_indices,
            constraint_offsets, solve_order, iterations=5, dt=1.0,
        )
        assert first_nan_step(traj) == 0


# ===========================================================================
# step_single_velocity tests
# ===========================================================================


class TestStepSingleVelocityFourBar:
    """step_single_velocity with the four-bar linkage."""

    def test_static_zero_velocity(self):
        (
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci,
        ) = _fourbar_kinematics_arrays()
        step_single(pos, con, jt, pi, co, so, dt=1.0)
        step_single_velocity(pos, vel, con, jt, pi, co, so, omega, ci)
        np.testing.assert_array_almost_equal(vel[0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(vel[2], [0.0, 0.0])

    def test_crank_velocity_magnitude(self):
        (
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci,
        ) = _fourbar_kinematics_arrays()
        step_single(pos, con, jt, pi, co, so, dt=1.0)
        step_single_velocity(pos, vel, con, jt, pi, co, so, omega, ci)
        # v = r * omega = 2.0 * 0.1 = 0.2
        mag = np.linalg.norm(vel[1])
        assert mag == pytest.approx(0.2, abs=1e-8)

    def test_revolute_velocity_non_nan(self):
        (
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci,
        ) = _fourbar_kinematics_arrays()
        step_single(pos, con, jt, pi, co, so, dt=1.0)
        step_single_velocity(pos, vel, con, jt, pi, co, so, omega, ci)
        assert not np.any(np.isnan(vel[3]))

    def test_revolute_velocity_nonzero(self):
        (
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci,
        ) = _fourbar_kinematics_arrays()
        step_single(pos, con, jt, pi, co, so, dt=1.0)
        step_single_velocity(pos, vel, con, jt, pi, co, so, omega, ci)
        # The revolute should have some velocity because the crank is moving
        assert np.linalg.norm(vel[3]) > 0.0


class TestStepSingleVelocityFixed:
    """step_single_velocity with JOINT_FIXED."""

    def test_fixed_velocity_with_moving_parents(self):
        pos, con, jt, pi, co, so = _fixed_joint_arrays()
        n = pos.shape[0]
        vel = np.zeros((n, 2), dtype=np.float64)
        omega = np.array([0.2], dtype=np.float64)
        ci = np.array([1], dtype=np.int32)

        step_single(pos, con, jt, pi, co, so, dt=1.0)
        step_single_velocity(pos, vel, con, jt, pi, co, so, omega, ci)

        # Fixed joint should have non-NaN velocity
        assert not np.any(np.isnan(vel[3]))
        # Crank is moving so fixed joint inherits some velocity
        assert np.linalg.norm(vel[3]) > 0.0


class TestStepSingleVelocityLinear:
    """step_single_velocity with JOINT_LINEAR."""

    def test_linear_velocity_with_moving_parent(self):
        pos, con, jt, pi, co, so = _linear_joint_arrays()
        n = pos.shape[0]
        vel = np.zeros((n, 2), dtype=np.float64)
        omega = np.array([0.3], dtype=np.float64)
        ci = np.array([1], dtype=np.int32)

        step_single(pos, con, jt, pi, co, so, dt=1.0)
        step_single_velocity(pos, vel, con, jt, pi, co, so, omega, ci)

        assert not np.any(np.isnan(vel[4]))
        # Crank is moving so linear joint should have some velocity
        assert np.linalg.norm(vel[4]) > 0.0


# ===========================================================================
# step_single_acceleration tests
# ===========================================================================


class TestStepSingleAccelerationFourBar:
    """step_single_acceleration with the four-bar linkage."""

    def test_static_zero_acceleration(self):
        (
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci,
        ) = _fourbar_kinematics_arrays()
        step_single(pos, con, jt, pi, co, so, dt=1.0)
        step_single_velocity(pos, vel, con, jt, pi, co, so, omega, ci)
        step_single_acceleration(pos, vel, acc, con, jt, pi, co, so, omega, alpha, ci)
        np.testing.assert_array_almost_equal(acc[0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(acc[2], [0.0, 0.0])

    def test_crank_centripetal_acceleration(self):
        (
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci,
        ) = _fourbar_kinematics_arrays()
        step_single(pos, con, jt, pi, co, so, dt=1.0)
        step_single_velocity(pos, vel, con, jt, pi, co, so, omega, ci)
        step_single_acceleration(pos, vel, acc, con, jt, pi, co, so, omega, alpha, ci)
        # centripetal magnitude = r * omega^2 = 2.0 * 0.01 = 0.02
        mag = np.linalg.norm(acc[1])
        assert mag == pytest.approx(2.0 * 0.1**2, abs=1e-8)

    def test_revolute_acceleration_non_nan(self):
        (
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci,
        ) = _fourbar_kinematics_arrays()
        step_single(pos, con, jt, pi, co, so, dt=1.0)
        step_single_velocity(pos, vel, con, jt, pi, co, so, omega, ci)
        step_single_acceleration(pos, vel, acc, con, jt, pi, co, so, omega, alpha, ci)
        assert not np.any(np.isnan(acc[3]))

    def test_with_angular_acceleration(self):
        (
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci,
        ) = _fourbar_kinematics_arrays()
        alpha[0] = 0.5  # non-zero angular acceleration
        step_single(pos, con, jt, pi, co, so, dt=1.0)
        step_single_velocity(pos, vel, con, jt, pi, co, so, omega, ci)
        step_single_acceleration(pos, vel, acc, con, jt, pi, co, so, omega, alpha, ci)
        # Should still produce valid accelerations
        assert not np.any(np.isnan(acc))


class TestStepSingleAccelerationFixed:
    """step_single_acceleration with JOINT_FIXED."""

    def test_fixed_acceleration_with_moving_parents(self):
        pos, con, jt, pi, co, so = _fixed_joint_arrays()
        n = pos.shape[0]
        vel = np.zeros((n, 2), dtype=np.float64)
        acc = np.zeros((n, 2), dtype=np.float64)
        omega = np.array([0.2], dtype=np.float64)
        alpha = np.array([0.0], dtype=np.float64)
        ci = np.array([1], dtype=np.int32)

        step_single(pos, con, jt, pi, co, so, dt=1.0)
        step_single_velocity(pos, vel, con, jt, pi, co, so, omega, ci)
        step_single_acceleration(pos, vel, acc, con, jt, pi, co, so, omega, alpha, ci)

        assert not np.any(np.isnan(acc[3]))


class TestStepSingleAccelerationLinear:
    """step_single_acceleration with JOINT_LINEAR."""

    def test_linear_acceleration_with_moving_parent(self):
        pos, con, jt, pi, co, so = _linear_joint_arrays()
        n = pos.shape[0]
        vel = np.zeros((n, 2), dtype=np.float64)
        acc = np.zeros((n, 2), dtype=np.float64)
        omega = np.array([0.3], dtype=np.float64)
        alpha = np.array([0.0], dtype=np.float64)
        ci = np.array([1], dtype=np.int32)

        step_single(pos, con, jt, pi, co, so, dt=1.0)
        step_single_velocity(pos, vel, con, jt, pi, co, so, omega, ci)
        step_single_acceleration(pos, vel, acc, con, jt, pi, co, so, omega, alpha, ci)

        assert not np.any(np.isnan(acc[4]))


# ===========================================================================
# simulate_with_kinematics tests
# ===========================================================================


class TestSimulateWithKinematicsFourBar:
    """simulate_with_kinematics with the four-bar linkage."""

    def test_output_shapes(self):
        (
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci,
        ) = _fourbar_kinematics_arrays()
        pt, vt, at = simulate_with_kinematics(
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci, iterations=10, dt=0.1,
        )
        assert pt.shape == (10, 4, 2)
        assert vt.shape == (10, 4, 2)
        assert at.shape == (10, 4, 2)

    def test_no_nan(self):
        (
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci,
        ) = _fourbar_kinematics_arrays()
        pt, vt, at = simulate_with_kinematics(
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci, iterations=20, dt=0.05,
        )
        assert not np.any(np.isnan(pt))
        assert not np.any(np.isnan(vt))
        assert not np.any(np.isnan(at))

    def test_static_joints_zero_kinematics(self):
        (
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci,
        ) = _fourbar_kinematics_arrays()
        pt, vt, at = simulate_with_kinematics(
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci, iterations=5, dt=0.1,
        )
        for step in range(5):
            np.testing.assert_array_almost_equal(pt[step, 0], [0.0, 0.0])
            np.testing.assert_array_almost_equal(pt[step, 2], [8.0, 0.0])

    def test_crank_velocity_constant_magnitude(self):
        (
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci,
        ) = _fourbar_kinematics_arrays()
        pt, vt, at = simulate_with_kinematics(
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci, iterations=10, dt=0.1,
        )
        for step in range(10):
            vmag = np.linalg.norm(vt[step, 1])
            assert vmag == pytest.approx(0.2, abs=1e-5)  # r * omega = 2 * 0.1

    def test_crank_accel_constant_magnitude(self):
        (
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci,
        ) = _fourbar_kinematics_arrays()
        pt, vt, at = simulate_with_kinematics(
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci, iterations=10, dt=0.1,
        )
        for step in range(10):
            amag = np.linalg.norm(at[step, 1])
            # centripetal = r * omega^2 = 2 * 0.01 = 0.02
            assert amag == pytest.approx(0.02, abs=1e-5)

    def test_revolute_distances_all_steps(self):
        (
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci,
        ) = _fourbar_kinematics_arrays()
        pt, vt, at = simulate_with_kinematics(
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci, iterations=15, dt=0.1,
        )
        for step in range(15):
            d0 = np.linalg.norm(pt[step, 3] - pt[step, 1])
            d1 = np.linalg.norm(pt[step, 3] - pt[step, 2])
            assert d0 == pytest.approx(7.0, abs=1e-6)
            assert d1 == pytest.approx(5.0, abs=1e-6)


class TestSimulateWithKinematicsFixed:
    """simulate_with_kinematics with JOINT_FIXED mechanism."""

    def test_fixed_kinematics_valid(self):
        pos, con, jt, pi, co, so = _fixed_joint_arrays()
        n = pos.shape[0]
        vel = np.zeros((n, 2), dtype=np.float64)
        acc = np.zeros((n, 2), dtype=np.float64)
        omega = np.array([0.2], dtype=np.float64)
        alpha = np.array([0.0], dtype=np.float64)
        ci = np.array([1], dtype=np.int32)

        pt, vt, at = simulate_with_kinematics(
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci, iterations=10, dt=0.1,
        )
        assert pt.shape == (10, 4, 2)
        assert not np.any(np.isnan(pt))
        assert not np.any(np.isnan(vt))
        assert not np.any(np.isnan(at))


class TestSimulateWithKinematicsLinear:
    """simulate_with_kinematics with JOINT_LINEAR mechanism."""

    def test_linear_kinematics_valid(self):
        pos, con, jt, pi, co, so = _linear_joint_arrays()
        n = pos.shape[0]
        vel = np.zeros((n, 2), dtype=np.float64)
        acc = np.zeros((n, 2), dtype=np.float64)
        omega = np.array([0.3], dtype=np.float64)
        alpha = np.array([0.0], dtype=np.float64)
        ci = np.array([1], dtype=np.int32)

        pt, vt, at = simulate_with_kinematics(
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci, iterations=10, dt=0.1,
        )
        assert pt.shape == (10, 5, 2)
        assert not np.any(np.isnan(pt))
        assert not np.any(np.isnan(vt))
        assert not np.any(np.isnan(at))

    def test_linear_stays_on_line_during_kinematics(self):
        pos, con, jt, pi, co, so = _linear_joint_arrays()
        n = pos.shape[0]
        vel = np.zeros((n, 2), dtype=np.float64)
        acc = np.zeros((n, 2), dtype=np.float64)
        omega = np.array([0.3], dtype=np.float64)
        alpha = np.array([0.0], dtype=np.float64)
        ci = np.array([1], dtype=np.int32)

        pt, vt, at = simulate_with_kinematics(
            pos, vel, acc, con, jt, pi, co, so,
            omega, alpha, ci, iterations=10, dt=0.1,
        )
        for step in range(10):
            assert pt[step, 4, 1] == pytest.approx(2.0, abs=1e-6)
