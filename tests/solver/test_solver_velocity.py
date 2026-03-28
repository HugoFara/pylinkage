"""Thorough tests for per-joint velocity solvers (velocity.py).

Tests each velocity function with analytically verifiable setups,
edge cases, NaN handling, and singular configurations.
"""

import math

import pytest

from pylinkage.solver.velocity import (
    solve_crank_velocity,
    solve_fixed_velocity,
    solve_prismatic_velocity,
    solve_revolute_velocity,
)


# ---------------------------------------------------------------------------
# solve_crank_velocity
# ---------------------------------------------------------------------------
class TestSolveCrankVelocity:
    """Tests for solve_crank_velocity."""

    def test_at_theta_0(self):
        """At theta=0, velocity is purely tangential (upward)."""
        vx, vy = solve_crank_velocity(5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0)
        # v = r*omega*(-sin(0), cos(0)) = (0, 10)
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy == pytest.approx(10.0, abs=1e-10)

    def test_at_theta_90(self):
        """At theta=pi/2, velocity is purely tangential (leftward)."""
        vx, vy = solve_crank_velocity(0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0)
        # v = r*omega*(-sin(pi/2), cos(pi/2)) = (-10, 0)
        assert vx == pytest.approx(-10.0, abs=1e-10)
        assert vy == pytest.approx(0.0, abs=1e-10)

    def test_at_theta_180(self):
        """At theta=pi, velocity is purely tangential (downward)."""
        vx, vy = solve_crank_velocity(-5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0)
        # v = r*omega*(-sin(pi), cos(pi)) = (0, -10)
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy == pytest.approx(-10.0, abs=1e-10)

    def test_at_theta_270(self):
        """At theta=3pi/2, velocity is purely tangential (rightward)."""
        vx, vy = solve_crank_velocity(0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0)
        # v = r*omega*(-sin(3pi/2), cos(3pi/2)) = (10, 0)
        assert vx == pytest.approx(10.0, abs=1e-10)
        assert vy == pytest.approx(0.0, abs=1e-10)

    def test_velocity_magnitude_equals_r_times_omega(self):
        """Velocity magnitude should always be r*omega (with stationary anchor)."""
        # At arbitrary angle
        r = 3.0
        omega = 7.0
        angle = 1.2
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        vx, vy = solve_crank_velocity(x, y, 0.0, 0.0, 0.0, 0.0, r, omega)
        mag = math.sqrt(vx * vx + vy * vy)
        assert mag == pytest.approx(r * omega, abs=1e-10)

    def test_with_moving_anchor(self):
        """Anchor velocity adds to tangential velocity."""
        vx, vy = solve_crank_velocity(1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 1.0, 10.0)
        # Tangential: (0, 10), plus anchor: (3, 4) -> total (3, 14)
        assert vx == pytest.approx(3.0, abs=1e-10)
        assert vy == pytest.approx(14.0, abs=1e-10)

    def test_zero_omega_zero_velocity(self):
        """Zero angular velocity with stationary anchor -> zero velocity."""
        vx, vy = solve_crank_velocity(5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0)
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy == pytest.approx(0.0, abs=1e-10)

    def test_negative_omega(self):
        """Negative omega reverses tangential velocity direction."""
        vx_pos, vy_pos = solve_crank_velocity(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0)
        vx_neg, vy_neg = solve_crank_velocity(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -5.0)
        assert vx_pos == pytest.approx(-vx_neg, abs=1e-10)
        assert vy_pos == pytest.approx(-vy_neg, abs=1e-10)

    def test_nan_x_returns_nan(self):
        """NaN x position returns NaN velocity."""
        vx, vy = solve_crank_velocity(math.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_nan_y_returns_nan(self):
        """NaN y position returns NaN velocity."""
        vx, vy = solve_crank_velocity(1.0, math.nan, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_offset_anchor_same_relative_position(self):
        """Tangential velocity depends only on relative position."""
        vx1, vy1 = solve_crank_velocity(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 10.0)
        vx2, vy2 = solve_crank_velocity(101.0, 50.0, 100.0, 50.0, 0.0, 0.0, 1.0, 10.0)
        assert vx1 == pytest.approx(vx2, abs=1e-10)
        assert vy1 == pytest.approx(vy2, abs=1e-10)


# ---------------------------------------------------------------------------
# solve_revolute_velocity
# ---------------------------------------------------------------------------
class TestSolveRevoluteVelocity:
    """Tests for solve_revolute_velocity."""

    def test_both_parents_stationary(self):
        """Stationary parents -> stationary joint."""
        jy = math.sqrt(3)
        vx, vy = solve_revolute_velocity(
            0.0, jy, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
        )
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy == pytest.approx(0.0, abs=1e-10)

    def test_rigid_translation(self):
        """Both parents moving identically -> joint translates same."""
        jy = math.sqrt(3)
        vx, vy = solve_revolute_velocity(
            0.0, jy, -1.0, 0.0, 5.0, 3.0, 1.0, 0.0, 5.0, 3.0
        )
        assert vx == pytest.approx(5.0, abs=1e-10)
        assert vy == pytest.approx(3.0, abs=1e-10)

    def test_one_parent_moving(self):
        """Only one parent moving -> non-trivial joint velocity."""
        # Joint at (3, 4) on circles from (0,0) r=5 and (6,0) r=5
        vx, vy = solve_revolute_velocity(
            3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 1.0, 0.0
        )
        # Should produce valid (non-NaN) result
        assert not math.isnan(vx)
        assert not math.isnan(vy)

    def test_collinear_singular_returns_nan(self):
        """Joint collinear with parents is singular."""
        vx, vy = solve_revolute_velocity(
            0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
        )
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_nan_x_returns_nan(self):
        """NaN x returns NaN."""
        vx, vy = solve_revolute_velocity(
            math.nan, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
        )
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_nan_y_returns_nan(self):
        """NaN y returns NaN."""
        vx, vy = solve_revolute_velocity(
            0.0, math.nan, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
        )
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_symmetric_parent_velocities(self):
        """Symmetric parent velocities produce predictable joint velocity.

        Parents at (-1,0) and (1,0), joint at (0, sqrt(3)).
        If p0 moves right at v and p1 moves left at v (squeezing),
        the joint must move upward to maintain distances.
        """
        jy = math.sqrt(3)
        v = 1.0
        vx, vy = solve_revolute_velocity(
            0.0, jy, -1.0, 0.0, v, 0.0, 1.0, 0.0, -v, 0.0
        )
        # By symmetry, vx should be 0, vy should be positive (joint moves up)
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy > 0

    def test_velocity_perpendicular_to_links(self):
        """Joint velocity should be consistent with constraint differentiation.

        For a revolute joint, the velocity projected onto the link direction
        should equal the parent's velocity projected onto that direction.
        """
        # Joint at (3,4), parent0 at (0,0), parent1 at (6,0)
        # r0=5, r1=5
        p0_vx, p0_vy = 2.0, 0.0
        vx, vy = solve_revolute_velocity(
            3.0, 4.0, 0.0, 0.0, p0_vx, p0_vy, 6.0, 0.0, 0.0, 0.0
        )
        # Check constraint: (x-p0_x)*vx + (y-p0_y)*vy = (x-p0_x)*p0_vx + (y-p0_y)*p0_vy
        lhs1 = 3.0 * vx + 4.0 * vy
        rhs1 = 3.0 * p0_vx + 4.0 * p0_vy
        assert lhs1 == pytest.approx(rhs1, abs=1e-10)


# ---------------------------------------------------------------------------
# solve_fixed_velocity
# ---------------------------------------------------------------------------
class TestSolveFixedVelocity:
    """Tests for solve_fixed_velocity."""

    def test_stationary_parents(self):
        """All stationary -> zero velocity."""
        vx, vy = solve_fixed_velocity(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0
        )
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation(self):
        """Both parents moving with same velocity -> joint translates."""
        vx, vy = solve_fixed_velocity(
            1.0, 0.0, 0.0, 0.0, 3.0, 7.0, 2.0, 0.0, 3.0, 7.0, 1.0, 0.0
        )
        # No relative motion -> dα/dt = 0, v = p0_v
        assert vx == pytest.approx(3.0, abs=1e-10)
        assert vy == pytest.approx(7.0, abs=1e-10)

    def test_rotating_reference(self):
        """Reference point rotating creates angular velocity of fixed joint."""
        # p0 at origin, p1 at (2,0) moving upward at 10
        # dα/dt = (dx * rel_vy - dy * rel_vx) / d_sq = (2*10 - 0*0) / 4 = 5
        # Joint at (1,0) with angle=0, radius=1
        # vx = 0 - 1*5*sin(0) = 0
        # vy = 0 + 1*5*cos(0) = 5
        vx, vy = solve_fixed_velocity(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 10.0, 1.0, 0.0
        )
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy == pytest.approx(5.0, abs=1e-10)

    def test_rotating_reference_with_angle_offset(self):
        """Reference rotating with non-zero angle offset."""
        # p0 at origin, p1 at (2,0) moving upward at speed 4
        # dα/dt = (2*4 - 0)/4 = 2
        # total_angle = pi/2 + 0 = pi/2
        # vx = 0 - 1*2*sin(pi/2) = -2
        # vy = 0 + 1*2*cos(pi/2) = 0
        vx, vy = solve_fixed_velocity(
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 1.0, math.pi / 2
        )
        assert vx == pytest.approx(-2.0, abs=1e-10)
        assert vy == pytest.approx(0.0, abs=1e-10)

    def test_coincident_parents_singular(self):
        """Coincident parents (d_sq=0) returns NaN."""
        vx, vy = solve_fixed_velocity(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
        )
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_nan_x_returns_nan(self):
        """NaN x returns NaN."""
        vx, vy = solve_fixed_velocity(
            math.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
        )
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_nan_y_returns_nan(self):
        """NaN y returns NaN."""
        vx, vy = solve_fixed_velocity(
            0.0, math.nan, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
        )
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_p0_moving_only(self):
        """Only p0 moving, p1 stationary -> joint follows p0 plus rotation."""
        # p0 at origin moving right at 1, p1 at (2,0) stationary
        # rel_vx = -1, rel_vy = 0
        # dα/dt = (2*0 - 0*(-1)) / 4 = 0
        # v = p0_v = (1, 0)
        vx, vy = solve_fixed_velocity(
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0
        )
        assert vx == pytest.approx(1.0, abs=1e-10)
        assert vy == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# solve_prismatic_velocity
# ---------------------------------------------------------------------------
class TestSolvePrismaticVelocity:
    """Tests for solve_prismatic_velocity."""

    def test_all_stationary(self):
        """All parents stationary -> zero velocity."""
        jx = math.sqrt(3)
        jy = 1.0
        vx, vy = solve_prismatic_velocity(
            jx, jy, 0.0, 0.0, 0.0, 0.0, 2.0,
            -5.0, 1.0, 0.0, 0.0, 5.0, 1.0, 0.0, 0.0,
        )
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy == pytest.approx(0.0, abs=1e-10)

    def test_circle_center_moving_horizontal(self):
        """Circle center moving right -> joint slides along line."""
        jx = math.sqrt(3)
        jy = 1.0
        vx, vy = solve_prismatic_velocity(
            jx, jy, 0.0, 0.0, 1.0, 0.0, 2.0,
            -5.0, 1.0, 0.0, 0.0, 5.0, 1.0, 0.0, 0.0,
        )
        # Joint must stay on line y=1, so vy=0
        assert vy == pytest.approx(0.0, abs=1e-10)
        assert not math.isnan(vx)

    def test_circle_center_moving_vertical(self):
        """Circle center moving up on horizontal line."""
        jx = math.sqrt(3)
        jy = 1.0
        vx, vy = solve_prismatic_velocity(
            jx, jy, 0.0, 0.0, 0.0, 1.0, 2.0,
            -5.0, 1.0, 0.0, 0.0, 5.0, 1.0, 0.0, 0.0,
        )
        assert not math.isnan(vx)
        assert not math.isnan(vy)

    def test_line_translating_upward(self):
        """Both line points moving up together."""
        jx = math.sqrt(3)
        jy = 1.0
        vx, vy = solve_prismatic_velocity(
            jx, jy, 0.0, 0.0, 0.0, 0.0, 2.0,
            -5.0, 1.0, 0.0, 1.0, 5.0, 1.0, 0.0, 1.0,
        )
        assert not math.isnan(vx)
        assert not math.isnan(vy)

    def test_line_rotating(self):
        """One line point stationary, other moving -> line rotates."""
        jx = math.sqrt(3)
        jy = 1.0
        vx, vy = solve_prismatic_velocity(
            jx, jy, 0.0, 0.0, 0.0, 0.0, 2.0,
            -5.0, 1.0, 0.0, 0.0, 5.0, 1.0, 0.0, 1.0,
        )
        assert not math.isnan(vx)
        assert not math.isnan(vy)

    def test_nan_x_returns_nan(self):
        """NaN x position returns NaN."""
        vx, vy = solve_prismatic_velocity(
            math.nan, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, 0.0, 5.0, 1.0, 0.0, 0.0,
        )
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_nan_y_returns_nan(self):
        """NaN y position returns NaN."""
        vx, vy = solve_prismatic_velocity(
            1.0, math.nan, 0.0, 0.0, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, 0.0, 5.0, 1.0, 0.0, 0.0,
        )
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_singular_joint_at_circle_center(self):
        """Joint coincident with circle center -> singular (det=0)."""
        vx, vy = solve_prismatic_velocity(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        )
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_constraint_consistency(self):
        """Verify velocity satisfies circle constraint differentiation.

        Circle constraint derivative:
        (x - cx)*vx + (y - cy)*vy = (x - cx)*cvx + (y - cy)*cvy
        """
        jx = math.sqrt(3)
        jy = 1.0
        cvx, cvy = 2.0, -1.0
        vx, vy = solve_prismatic_velocity(
            jx, jy, 0.0, 0.0, cvx, cvy, 2.0,
            -5.0, 1.0, 0.0, 0.0, 5.0, 1.0, 0.0, 0.0,
        )
        # Check circle constraint: (x-cx)*vx + (y-cy)*vy = (x-cx)*cvx + (y-cy)*cvy
        lhs = jx * vx + jy * vy
        rhs = jx * cvx + jy * cvy
        assert lhs == pytest.approx(rhs, abs=1e-10)
