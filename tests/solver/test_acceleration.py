"""Tests for acceleration solver functions (acceleration.py)."""

import math

import pytest

from pylinkage.solver.acceleration import (
    solve_crank_acceleration,
    solve_fixed_acceleration,
    solve_prismatic_acceleration,
    solve_revolute_acceleration,
)


class TestSolveCrankAcceleration:
    """Tests for solve_crank_acceleration."""

    def test_constant_omega_centripetal_only(self):
        """With constant angular velocity (alpha=0), only centripetal remains."""
        # Crank at (1,0) around origin, omega=10, alpha=0
        ax, ay = solve_crank_acceleration(
            x=1.0,
            y=0.0,
            vx=0.0,
            vy=10.0,  # tangential velocity
            anchor_x=0.0,
            anchor_y=0.0,
            anchor_vx=0.0,
            anchor_vy=0.0,
            anchor_ax=0.0,
            anchor_ay=0.0,
            radius=1.0,
            omega=10.0,
            alpha=0.0,
        )
        # Centripetal: -r*omega^2*(cos,sin) = -100*(1,0) = (-100, 0)
        assert ax == pytest.approx(-100.0, abs=1e-10)
        assert ay == pytest.approx(0.0, abs=1e-10)

    def test_with_angular_acceleration(self):
        """Angular acceleration adds tangential component."""
        ax, ay = solve_crank_acceleration(
            x=1.0,
            y=0.0,
            vx=0.0,
            vy=10.0,
            anchor_x=0.0,
            anchor_y=0.0,
            anchor_vx=0.0,
            anchor_vy=0.0,
            anchor_ax=0.0,
            anchor_ay=0.0,
            radius=1.0,
            omega=10.0,
            alpha=5.0,
        )
        # Tangential: r*alpha*(-sin,cos) = 5*(0,1) = (0, 5)
        # Centripetal: -r*omega^2*(cos,sin) = (-100, 0)
        # Total: (-100, 5)
        assert ax == pytest.approx(-100.0, abs=1e-10)
        assert ay == pytest.approx(5.0, abs=1e-10)

    def test_at_90_degrees(self):
        """Crank at 90 degrees (top)."""
        ax, ay = solve_crank_acceleration(
            x=0.0,
            y=1.0,
            vx=-10.0,
            vy=0.0,
            anchor_x=0.0,
            anchor_y=0.0,
            anchor_vx=0.0,
            anchor_vy=0.0,
            anchor_ax=0.0,
            anchor_ay=0.0,
            radius=1.0,
            omega=10.0,
            alpha=0.0,
        )
        # Centripetal: -r*omega^2*(cos90, sin90) = -100*(0, 1) = (0, -100)
        assert ax == pytest.approx(0.0, abs=1e-10)
        assert ay == pytest.approx(-100.0, abs=1e-10)

    def test_with_anchor_acceleration(self):
        """Anchor acceleration propagates to crank."""
        ax, ay = solve_crank_acceleration(
            x=1.0,
            y=0.0,
            vx=0.0,
            vy=0.0,
            anchor_x=0.0,
            anchor_y=0.0,
            anchor_vx=0.0,
            anchor_vy=0.0,
            anchor_ax=3.0,
            anchor_ay=7.0,
            radius=1.0,
            omega=0.0,
            alpha=0.0,
        )
        # No rotation at all -> only anchor acceleration
        assert ax == pytest.approx(3.0, abs=1e-10)
        assert ay == pytest.approx(7.0, abs=1e-10)

    def test_nan_input_returns_nan(self):
        """NaN position returns NaN acceleration."""
        ax, ay = solve_crank_acceleration(
            x=math.nan,
            y=0.0,
            vx=0.0,
            vy=0.0,
            anchor_x=0.0,
            anchor_y=0.0,
            anchor_vx=0.0,
            anchor_vy=0.0,
            anchor_ax=0.0,
            anchor_ay=0.0,
            radius=1.0,
            omega=10.0,
            alpha=0.0,
        )
        assert math.isnan(ax)
        assert math.isnan(ay)


class TestSolveRevoluteAcceleration:
    """Tests for solve_revolute_acceleration."""

    def test_stationary_parents_stationary_joint(self):
        """All stationary with zero velocities -> zero acceleration."""
        jy = math.sqrt(3)
        ax, ay = solve_revolute_acceleration(
            x=0.0,
            y=jy,
            vx=0.0,
            vy=0.0,
            p0_x=-1.0,
            p0_y=0.0,
            p0_vx=0.0,
            p0_vy=0.0,
            p0_ax=0.0,
            p0_ay=0.0,
            p1_x=1.0,
            p1_y=0.0,
            p1_vx=0.0,
            p1_vy=0.0,
            p1_ax=0.0,
            p1_ay=0.0,
        )
        assert ax == pytest.approx(0.0, abs=1e-10)
        assert ay == pytest.approx(0.0, abs=1e-10)

    def test_with_parent_acceleration(self):
        """Non-zero parent acceleration produces joint acceleration."""
        jy = math.sqrt(3)
        ax, ay = solve_revolute_acceleration(
            x=0.0,
            y=jy,
            vx=0.0,
            vy=0.0,
            p0_x=-1.0,
            p0_y=0.0,
            p0_vx=0.0,
            p0_vy=0.0,
            p0_ax=1.0,
            p0_ay=0.0,  # Parent 0 accelerating right
            p1_x=1.0,
            p1_y=0.0,
            p1_vx=0.0,
            p1_vy=0.0,
            p1_ax=1.0,
            p1_ay=0.0,  # Same acceleration = rigid translation
        )
        # Rigid body translation: joint acc = parent acc
        assert ax == pytest.approx(1.0, abs=1e-10)
        assert ay == pytest.approx(0.0, abs=1e-10)

    def test_nan_position_returns_nan(self):
        """NaN position returns NaN acceleration."""
        ax, ay = solve_revolute_acceleration(
            x=math.nan,
            y=0.0,
            vx=0.0,
            vy=0.0,
            p0_x=0.0,
            p0_y=0.0,
            p0_vx=0.0,
            p0_vy=0.0,
            p0_ax=0.0,
            p0_ay=0.0,
            p1_x=1.0,
            p1_y=0.0,
            p1_vx=0.0,
            p1_vy=0.0,
            p1_ax=0.0,
            p1_ay=0.0,
        )
        assert math.isnan(ax)
        assert math.isnan(ay)

    def test_nan_velocity_returns_nan(self):
        """NaN velocity returns NaN acceleration."""
        ax, ay = solve_revolute_acceleration(
            x=0.0,
            y=1.0,
            vx=math.nan,
            vy=0.0,
            p0_x=-1.0,
            p0_y=0.0,
            p0_vx=0.0,
            p0_vy=0.0,
            p0_ax=0.0,
            p0_ay=0.0,
            p1_x=1.0,
            p1_y=0.0,
            p1_vx=0.0,
            p1_vy=0.0,
            p1_ax=0.0,
            p1_ay=0.0,
        )
        assert math.isnan(ax)
        assert math.isnan(ay)

    def test_singular_collinear(self):
        """Collinear configuration returns NaN (singular)."""
        ax, ay = solve_revolute_acceleration(
            x=0.0,
            y=0.0,
            vx=0.0,
            vy=0.0,
            p0_x=-1.0,
            p0_y=0.0,
            p0_vx=0.0,
            p0_vy=0.0,
            p0_ax=0.0,
            p0_ay=0.0,
            p1_x=1.0,
            p1_y=0.0,
            p1_vx=0.0,
            p1_vy=0.0,
            p1_ax=0.0,
            p1_ay=0.0,
        )
        assert math.isnan(ax)
        assert math.isnan(ay)

    def test_with_joint_velocity(self):
        """Non-zero joint velocity produces centripetal-like terms."""
        jy = math.sqrt(3)
        ax, ay = solve_revolute_acceleration(
            x=0.0,
            y=jy,
            vx=1.0,
            vy=0.0,  # Joint moving
            p0_x=-1.0,
            p0_y=0.0,
            p0_vx=0.0,
            p0_vy=0.0,
            p0_ax=0.0,
            p0_ay=0.0,
            p1_x=1.0,
            p1_y=0.0,
            p1_vx=0.0,
            p1_vy=0.0,
            p1_ax=0.0,
            p1_ay=0.0,
        )
        # Non-zero because of velocity squared terms
        assert not math.isnan(ax)
        assert not math.isnan(ay)


class TestSolveFixedAcceleration:
    """Tests for solve_fixed_acceleration."""

    def test_stationary_parents(self):
        """Stationary parents -> zero acceleration."""
        ax, ay = solve_fixed_acceleration(
            x=1.0,
            y=0.0,
            vx=0.0,
            vy=0.0,
            p0_x=0.0,
            p0_y=0.0,
            p0_vx=0.0,
            p0_vy=0.0,
            p0_ax=0.0,
            p0_ay=0.0,
            p1_x=2.0,
            p1_y=0.0,
            p1_vx=0.0,
            p1_vy=0.0,
            p1_ax=0.0,
            p1_ay=0.0,
            radius=1.0,
            angle=0.0,
        )
        assert ax == pytest.approx(0.0, abs=1e-10)
        assert ay == pytest.approx(0.0, abs=1e-10)

    def test_rigid_translation(self):
        """Both parents accelerating uniformly -> joint follows."""
        ax, ay = solve_fixed_acceleration(
            x=1.0,
            y=0.0,
            vx=3.0,
            vy=7.0,
            p0_x=0.0,
            p0_y=0.0,
            p0_vx=3.0,
            p0_vy=7.0,
            p0_ax=2.0,
            p0_ay=5.0,
            p1_x=2.0,
            p1_y=0.0,
            p1_vx=3.0,
            p1_vy=7.0,
            p1_ax=2.0,
            p1_ay=5.0,  # Same accel = pure translation
            radius=1.0,
            angle=0.0,
        )
        assert ax == pytest.approx(2.0, abs=1e-10)
        assert ay == pytest.approx(5.0, abs=1e-10)

    def test_nan_position_returns_nan(self):
        """NaN position returns NaN acceleration."""
        ax, ay = solve_fixed_acceleration(
            x=math.nan,
            y=0.0,
            vx=0.0,
            vy=0.0,
            p0_x=0.0,
            p0_y=0.0,
            p0_vx=0.0,
            p0_vy=0.0,
            p0_ax=0.0,
            p0_ay=0.0,
            p1_x=1.0,
            p1_y=0.0,
            p1_vx=0.0,
            p1_vy=0.0,
            p1_ax=0.0,
            p1_ay=0.0,
            radius=1.0,
            angle=0.0,
        )
        assert math.isnan(ax)
        assert math.isnan(ay)

    def test_coincident_parents_singular(self):
        """Coincident parents produce singular result."""
        ax, ay = solve_fixed_acceleration(
            x=1.0,
            y=0.0,
            vx=0.0,
            vy=0.0,
            p0_x=0.0,
            p0_y=0.0,
            p0_vx=0.0,
            p0_vy=0.0,
            p0_ax=0.0,
            p0_ay=0.0,
            p1_x=0.0,
            p1_y=0.0,
            p1_vx=0.0,
            p1_vy=0.0,
            p1_ax=0.0,
            p1_ay=0.0,
            radius=1.0,
            angle=0.0,
        )
        assert math.isnan(ax)
        assert math.isnan(ay)


class TestSolvePrismaticAcceleration:
    """Tests for solve_prismatic_acceleration."""

    def test_stationary_all(self):
        """All stationary -> zero acceleration."""
        jx = math.sqrt(3)
        jy = 1.0
        ax, ay = solve_prismatic_acceleration(
            x=jx,
            y=jy,
            vx=0.0,
            vy=0.0,
            circle_x=0.0,
            circle_y=0.0,
            circle_vx=0.0,
            circle_vy=0.0,
            circle_ax=0.0,
            circle_ay=0.0,
            radius=2.0,
            line_p1_x=-5.0,
            line_p1_y=1.0,
            line_p1_vx=0.0,
            line_p1_vy=0.0,
            line_p1_ax=0.0,
            line_p1_ay=0.0,
            line_p2_x=5.0,
            line_p2_y=1.0,
            line_p2_vx=0.0,
            line_p2_vy=0.0,
            line_p2_ax=0.0,
            line_p2_ay=0.0,
        )
        assert ax == pytest.approx(0.0, abs=1e-10)
        assert ay == pytest.approx(0.0, abs=1e-10)

    def test_nan_position_returns_nan(self):
        """NaN position returns NaN acceleration."""
        ax, ay = solve_prismatic_acceleration(
            x=math.nan,
            y=1.0,
            vx=0.0,
            vy=0.0,
            circle_x=0.0,
            circle_y=0.0,
            circle_vx=0.0,
            circle_vy=0.0,
            circle_ax=0.0,
            circle_ay=0.0,
            radius=2.0,
            line_p1_x=0.0,
            line_p1_y=1.0,
            line_p1_vx=0.0,
            line_p1_vy=0.0,
            line_p1_ax=0.0,
            line_p1_ay=0.0,
            line_p2_x=5.0,
            line_p2_y=1.0,
            line_p2_vx=0.0,
            line_p2_vy=0.0,
            line_p2_ax=0.0,
            line_p2_ay=0.0,
        )
        assert math.isnan(ax)
        assert math.isnan(ay)

    def test_nan_velocity_returns_nan(self):
        """NaN velocity returns NaN acceleration."""
        ax, ay = solve_prismatic_acceleration(
            x=1.0,
            y=1.0,
            vx=math.nan,
            vy=0.0,
            circle_x=0.0,
            circle_y=0.0,
            circle_vx=0.0,
            circle_vy=0.0,
            circle_ax=0.0,
            circle_ay=0.0,
            radius=2.0,
            line_p1_x=0.0,
            line_p1_y=1.0,
            line_p1_vx=0.0,
            line_p1_vy=0.0,
            line_p1_ax=0.0,
            line_p1_ay=0.0,
            line_p2_x=5.0,
            line_p2_y=1.0,
            line_p2_vx=0.0,
            line_p2_vy=0.0,
            line_p2_ax=0.0,
            line_p2_ay=0.0,
        )
        assert math.isnan(ax)
        assert math.isnan(ay)

    def test_singular_configuration(self):
        """Singular configuration returns NaN."""
        ax, ay = solve_prismatic_acceleration(
            x=0.0,
            y=0.0,
            vx=0.0,
            vy=0.0,
            circle_x=0.0,
            circle_y=0.0,
            circle_vx=0.0,
            circle_vy=0.0,
            circle_ax=0.0,
            circle_ay=0.0,
            radius=0.0,
            line_p1_x=-1.0,
            line_p1_y=0.0,
            line_p1_vx=0.0,
            line_p1_vy=0.0,
            line_p1_ax=0.0,
            line_p1_ay=0.0,
            line_p2_x=1.0,
            line_p2_y=0.0,
            line_p2_vx=0.0,
            line_p2_vy=0.0,
            line_p2_ax=0.0,
            line_p2_ay=0.0,
        )
        assert math.isnan(ax)
        assert math.isnan(ay)

    def test_with_circle_acceleration(self):
        """Circle center accelerating produces non-zero result."""
        jx = math.sqrt(3)
        jy = 1.0
        ax, ay = solve_prismatic_acceleration(
            x=jx,
            y=jy,
            vx=0.5,
            vy=0.0,
            circle_x=0.0,
            circle_y=0.0,
            circle_vx=0.0,
            circle_vy=0.0,
            circle_ax=1.0,
            circle_ay=0.0,
            radius=2.0,
            line_p1_x=-5.0,
            line_p1_y=1.0,
            line_p1_vx=0.0,
            line_p1_vy=0.0,
            line_p1_ax=0.0,
            line_p1_ay=0.0,
            line_p2_x=5.0,
            line_p2_y=1.0,
            line_p2_vx=0.0,
            line_p2_vy=0.0,
            line_p2_ax=0.0,
            line_p2_ay=0.0,
        )
        assert not math.isnan(ax)
        assert not math.isnan(ay)
