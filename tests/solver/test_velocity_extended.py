"""Extended tests for velocity solver functions (velocity.py).

Covers: solve_prismatic_velocity and additional edge cases for
solve_crank_velocity, solve_revolute_velocity, solve_fixed_velocity.
"""

import math

import pytest

from pylinkage.solver.velocity import (
    solve_crank_velocity,
    solve_fixed_velocity,
    solve_prismatic_velocity,
    solve_revolute_velocity,
)


class TestSolvePrismaticVelocity:
    """Tests for solve_prismatic_velocity."""

    def test_stationary_parents(self):
        """All parents stationary -> joint stationary."""
        # Joint on line y=1 at distance 2 from origin
        # Place joint at (sqrt(3), 1) so circle constraint is satisfied
        jx = math.sqrt(3)
        jy = 1.0
        vx, vy = solve_prismatic_velocity(
            x=jx,
            y=jy,
            circle_x=0.0,
            circle_y=0.0,
            circle_vx=0.0,
            circle_vy=0.0,
            radius=2.0,
            line_p1_x=-5.0,
            line_p1_y=1.0,
            line_p1_vx=0.0,
            line_p1_vy=0.0,
            line_p2_x=5.0,
            line_p2_y=1.0,
            line_p2_vx=0.0,
            line_p2_vy=0.0,
        )
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy == pytest.approx(0.0, abs=1e-10)

    def test_circle_center_moving(self):
        """Moving circle center produces non-zero velocity."""
        jx = math.sqrt(3)
        jy = 1.0
        vx, vy = solve_prismatic_velocity(
            x=jx,
            y=jy,
            circle_x=0.0,
            circle_y=0.0,
            circle_vx=1.0,  # Moving right
            circle_vy=0.0,
            radius=2.0,
            line_p1_x=-5.0,
            line_p1_y=1.0,
            line_p1_vx=0.0,
            line_p1_vy=0.0,
            line_p2_x=5.0,
            line_p2_y=1.0,
            line_p2_vx=0.0,
            line_p2_vy=0.0,
        )
        assert not math.isnan(vx)
        assert not math.isnan(vy)
        # Joint must stay on the line, so vy should be 0
        assert vy == pytest.approx(0.0, abs=1e-10)

    def test_nan_input_returns_nan(self):
        """NaN position input returns NaN velocity."""
        vx, vy = solve_prismatic_velocity(
            x=math.nan,
            y=1.0,
            circle_x=0.0,
            circle_y=0.0,
            circle_vx=0.0,
            circle_vy=0.0,
            radius=2.0,
            line_p1_x=0.0,
            line_p1_y=1.0,
            line_p1_vx=0.0,
            line_p1_vy=0.0,
            line_p2_x=5.0,
            line_p2_y=1.0,
            line_p2_vx=0.0,
            line_p2_vy=0.0,
        )
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_singular_configuration(self):
        """Joint at circle center should be singular."""
        # If joint position equals circle center, a11=a12=0 -> singular
        vx, vy = solve_prismatic_velocity(
            x=0.0,
            y=0.0,
            circle_x=0.0,
            circle_y=0.0,
            circle_vx=0.0,
            circle_vy=0.0,
            radius=0.0,
            line_p1_x=-1.0,
            line_p1_y=0.0,
            line_p1_vx=0.0,
            line_p1_vy=0.0,
            line_p2_x=1.0,
            line_p2_y=0.0,
            line_p2_vx=0.0,
            line_p2_vy=0.0,
        )
        # This should be singular (det ~ 0)
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_line_points_moving(self):
        """Moving line points produce non-zero velocity."""
        jx = math.sqrt(3)
        jy = 1.0
        vx, vy = solve_prismatic_velocity(
            x=jx,
            y=jy,
            circle_x=0.0,
            circle_y=0.0,
            circle_vx=0.0,
            circle_vy=0.0,
            radius=2.0,
            line_p1_x=-5.0,
            line_p1_y=1.0,
            line_p1_vx=0.0,
            line_p1_vy=1.0,  # Line moving up
            line_p2_x=5.0,
            line_p2_y=1.0,
            line_p2_vx=0.0,
            line_p2_vy=1.0,  # Both points moving up together
        )
        assert not math.isnan(vx)
        assert not math.isnan(vy)


class TestCrankVelocityAdditional:
    """Additional edge cases for crank velocity."""

    def test_zero_omega(self):
        """Zero angular velocity means no rotation velocity."""
        vx, vy = solve_crank_velocity(
            x=1.0,
            y=0.0,
            anchor_x=0.0,
            anchor_y=0.0,
            anchor_vx=0.0,
            anchor_vy=0.0,
            radius=1.0,
            omega=0.0,
        )
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy == pytest.approx(0.0, abs=1e-10)

    def test_negative_omega(self):
        """Negative omega reverses velocity direction."""
        vx_pos, vy_pos = solve_crank_velocity(
            x=1.0,
            y=0.0,
            anchor_x=0.0,
            anchor_y=0.0,
            anchor_vx=0.0,
            anchor_vy=0.0,
            radius=1.0,
            omega=5.0,
        )
        vx_neg, vy_neg = solve_crank_velocity(
            x=1.0,
            y=0.0,
            anchor_x=0.0,
            anchor_y=0.0,
            anchor_vx=0.0,
            anchor_vy=0.0,
            radius=1.0,
            omega=-5.0,
        )
        assert vx_pos == pytest.approx(-vx_neg, abs=1e-10)
        assert vy_pos == pytest.approx(-vy_neg, abs=1e-10)

    def test_nan_y_returns_nan(self):
        """NaN in y also returns NaN."""
        vx, vy = solve_crank_velocity(
            x=1.0,
            y=math.nan,
            anchor_x=0.0,
            anchor_y=0.0,
            anchor_vx=0.0,
            anchor_vy=0.0,
            radius=1.0,
            omega=5.0,
        )
        assert math.isnan(vx)
        assert math.isnan(vy)


class TestRevoluteVelocityAdditional:
    """Additional edge cases for revolute velocity."""

    def test_nan_input_returns_nan(self):
        """NaN position returns NaN velocity."""
        vx, vy = solve_revolute_velocity(
            x=math.nan,
            y=0.0,
            p0_x=0.0,
            p0_y=0.0,
            p0_vx=0.0,
            p0_vy=0.0,
            p1_x=1.0,
            p1_y=0.0,
            p1_vx=0.0,
            p1_vy=0.0,
        )
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_both_parents_moving_same_direction(self):
        """Both parents moving identically -> joint moves identically."""
        # Joint at (0, sqrt(3)), parents at (-1,0) and (1,0), both r=2
        jy = math.sqrt(3)
        vx, vy = solve_revolute_velocity(
            x=0.0,
            y=jy,
            p0_x=-1.0,
            p0_y=0.0,
            p0_vx=5.0,
            p0_vy=0.0,  # Both moving right
            p1_x=1.0,
            p1_y=0.0,
            p1_vx=5.0,
            p1_vy=0.0,
        )
        # Rigid translation: joint velocity = parent velocity
        assert vx == pytest.approx(5.0, abs=1e-10)
        assert vy == pytest.approx(0.0, abs=1e-10)


class TestFixedVelocityAdditional:
    """Additional edge cases for fixed velocity."""

    def test_nan_input_returns_nan(self):
        """NaN position returns NaN velocity."""
        vx, vy = solve_fixed_velocity(
            x=math.nan,
            y=0.0,
            p0_x=0.0,
            p0_y=0.0,
            p0_vx=0.0,
            p0_vy=0.0,
            p1_x=1.0,
            p1_y=0.0,
            p1_vx=0.0,
            p1_vy=0.0,
            radius=1.0,
            angle=0.0,
        )
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_coincident_parents_singular(self):
        """Coincident parents produce singular result."""
        vx, vy = solve_fixed_velocity(
            x=1.0,
            y=0.0,
            p0_x=0.0,
            p0_y=0.0,
            p0_vx=0.0,
            p0_vy=0.0,
            p1_x=0.0,
            p1_y=0.0,  # Same as p0
            p1_vx=0.0,
            p1_vy=0.0,
            radius=1.0,
            angle=0.0,
        )
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_both_parents_translating(self):
        """Both parents moving with same velocity -> joint follows."""
        vx, vy = solve_fixed_velocity(
            x=1.0,
            y=0.0,
            p0_x=0.0,
            p0_y=0.0,
            p0_vx=3.0,
            p0_vy=7.0,
            p1_x=2.0,
            p1_y=0.0,
            p1_vx=3.0,
            p1_vy=7.0,  # Same velocity = pure translation
            radius=1.0,
            angle=0.0,
        )
        # Pure translation: no rotation, so dα/dt = 0
        # v = p0_v = (3, 7)
        assert vx == pytest.approx(3.0, abs=1e-10)
        assert vy == pytest.approx(7.0, abs=1e-10)
