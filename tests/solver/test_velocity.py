"""Tests for velocity solvers and kinematics computation."""

import math

import numpy as np
import pytest

from pylinkage import Linkage
from pylinkage.joints import Crank, Revolute, Static
from pylinkage.solver.velocity import (
    solve_crank_velocity,
    solve_fixed_velocity,
    solve_prismatic_velocity,
    solve_revolute_velocity,
)


class TestSolveCrankVelocity:
    """Test cases for crank velocity computation."""

    def test_crank_velocity_at_zero_angle(self) -> None:
        """At theta=0, velocity should be purely vertical (tangent)."""
        # Position at theta=0: (1, 0) relative to anchor at origin
        vx, vy = solve_crank_velocity(
            x=1.0,
            y=0.0,
            anchor_x=0.0,
            anchor_y=0.0,
            anchor_vx=0.0,
            anchor_vy=0.0,
            radius=1.0,
            omega=10.0,
        )
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy == pytest.approx(10.0, abs=1e-10)  # r * omega

    def test_crank_velocity_at_90_degrees(self) -> None:
        """At theta=90°, velocity should be purely horizontal (tangent)."""
        # Position at theta=90°: (0, 1) relative to anchor at origin
        vx, vy = solve_crank_velocity(
            x=0.0,
            y=1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            anchor_vx=0.0,
            anchor_vy=0.0,
            radius=1.0,
            omega=10.0,
        )
        assert vx == pytest.approx(-10.0, abs=1e-10)  # -r * omega
        assert vy == pytest.approx(0.0, abs=1e-10)

    def test_crank_velocity_with_offset_anchor(self) -> None:
        """Velocity should be same regardless of anchor position."""
        vx1, vy1 = solve_crank_velocity(
            x=1.0,
            y=0.0,
            anchor_x=0.0,
            anchor_y=0.0,
            anchor_vx=0.0,
            anchor_vy=0.0,
            radius=1.0,
            omega=5.0,
        )
        # Same configuration but shifted anchor
        vx2, vy2 = solve_crank_velocity(
            x=11.0,
            y=10.0,
            anchor_x=10.0,
            anchor_y=10.0,
            anchor_vx=0.0,
            anchor_vy=0.0,
            radius=1.0,
            omega=5.0,
        )
        assert vx1 == pytest.approx(vx2, abs=1e-10)
        assert vy1 == pytest.approx(vy2, abs=1e-10)

    def test_crank_velocity_with_moving_anchor(self) -> None:
        """Velocity includes anchor velocity."""
        vx, vy = solve_crank_velocity(
            x=1.0,
            y=0.0,
            anchor_x=0.0,
            anchor_y=0.0,
            anchor_vx=5.0,  # Anchor moving right
            anchor_vy=3.0,  # Anchor moving up
            radius=1.0,
            omega=10.0,
        )
        assert vx == pytest.approx(5.0, abs=1e-10)  # 0 + anchor_vx
        assert vy == pytest.approx(13.0, abs=1e-10)  # 10 + anchor_vy

    def test_crank_velocity_nan_input(self) -> None:
        """NaN position should return NaN velocity."""
        vx, vy = solve_crank_velocity(
            x=math.nan,
            y=0.0,
            anchor_x=0.0,
            anchor_y=0.0,
            anchor_vx=0.0,
            anchor_vy=0.0,
            radius=1.0,
            omega=10.0,
        )
        assert math.isnan(vx)
        assert math.isnan(vy)


class TestSolveRevoluteVelocity:
    """Test cases for revolute joint velocity computation."""

    def test_revolute_velocity_symmetric_parents(self) -> None:
        """Symmetric configuration: joint at top of two circles."""
        # Two parents at (-1, 0) and (1, 0), joint at (0, sqrt(3)) for equilateral
        joint_y = math.sqrt(3)
        vx, vy = solve_revolute_velocity(
            x=0.0,
            y=joint_y,
            p0_x=-1.0,
            p0_y=0.0,
            p0_vx=0.0,  # Parent 0 stationary
            p0_vy=0.0,
            p1_x=1.0,
            p1_y=0.0,
            p1_vx=0.0,  # Parent 1 stationary
            p1_vy=0.0,
        )
        # Both parents stationary -> joint stationary
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy == pytest.approx(0.0, abs=1e-10)

    def test_revolute_velocity_parent_moving(self) -> None:
        """When one parent moves, joint velocity is computed correctly."""
        # Joint at (0, 1), parents at (-1, 0) and (1, 0)
        # Right parent moving right at 10 units/s
        vx, vy = solve_revolute_velocity(
            x=0.0,
            y=1.0,
            p0_x=-1.0,
            p0_y=0.0,
            p0_vx=0.0,
            p0_vy=0.0,
            p1_x=1.0,
            p1_y=0.0,
            p1_vx=10.0,  # Moving right
            p1_vy=0.0,
        )
        # Joint should move to maintain both distances
        assert not math.isnan(vx)
        assert not math.isnan(vy)

    def test_revolute_velocity_singular_configuration(self) -> None:
        """Collinear configuration should return NaN (singular)."""
        # Joint collinear with parents (lock-up position)
        vx, vy = solve_revolute_velocity(
            x=0.0,
            y=0.0,  # Joint between parents
            p0_x=-1.0,
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


class TestSolveFixedVelocity:
    """Test cases for fixed joint velocity computation."""

    def test_fixed_velocity_stationary_parents(self) -> None:
        """Stationary parents -> stationary joint."""
        vx, vy = solve_fixed_velocity(
            x=1.0,
            y=0.0,
            p0_x=0.0,
            p0_y=0.0,
            p0_vx=0.0,
            p0_vy=0.0,
            p1_x=2.0,
            p1_y=0.0,
            p1_vx=0.0,
            p1_vy=0.0,
            radius=1.0,
            angle=0.0,
        )
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy == pytest.approx(0.0, abs=1e-10)

    def test_fixed_velocity_rotating_reference(self) -> None:
        """When reference point rotates, joint follows."""
        # Parent 0 at origin, parent 1 rotating around it
        # Reference point moving perpendicular creates angular velocity
        vx, vy = solve_fixed_velocity(
            x=1.0,
            y=0.0,
            p0_x=0.0,
            p0_y=0.0,
            p0_vx=0.0,
            p0_vy=0.0,
            p1_x=2.0,
            p1_y=0.0,
            p1_vx=0.0,
            p1_vy=10.0,  # Moving up -> creates rotation
            radius=1.0,
            angle=0.0,
        )
        # dα/dt = (dx * rel_vy - dy * rel_vx) / d²
        # = (2 * 10 - 0 * 0) / 4 = 5 rad/s
        # vx = -r * dα/dt * sin(0) = 0
        # vy = r * dα/dt * cos(0) = 5
        assert vx == pytest.approx(0.0, abs=1e-10)
        assert vy == pytest.approx(5.0, abs=1e-10)


class TestLinkageKinematicsAPI:
    """Integration tests for Linkage kinematics API."""

    @pytest.fixture
    def four_bar(self) -> Linkage:
        """Create a simple four-bar linkage."""
        # Ground anchors
        a = Static(0, 0, name="A")
        d = Static(100, 0, name="D")
        # Crank
        b = Crank(30, 0, joint0=a, distance=30, angle=0.1, name="B")
        # Coupler and rocker
        c = Revolute(80, 50, joint0=b, joint1=d, distance0=80, distance1=60, name="C")
        return Linkage(
            joints=(a, b, c, d),
            order=(a, d, b, c),
            name="four-bar",
        )

    def test_set_input_velocity(self, four_bar: Linkage) -> None:
        """Test setting input velocity on crank."""
        crank = four_bar.joints[1]
        assert isinstance(crank, Crank)

        four_bar.set_input_velocity(crank, omega=10.0, alpha=1.0)

        assert crank.omega == 10.0
        assert crank.alpha == 1.0

    def test_set_input_velocity_invalid_joint(self, four_bar: Linkage) -> None:
        """Setting velocity on non-crank joint should raise."""
        static = four_bar.joints[0]
        assert isinstance(static, Static)

        with pytest.raises(ValueError, match="not a crank"):
            four_bar.set_input_velocity(static, omega=10.0)  # type: ignore[arg-type]

    def test_step_fast_with_kinematics(self, four_bar: Linkage) -> None:
        """Test kinematics simulation returns positions, velocities, and accelerations."""
        crank = four_bar.joints[1]
        assert isinstance(crank, Crank)
        four_bar.set_input_velocity(crank, omega=10.0)

        positions, velocities, accelerations = four_bar.step_fast_with_kinematics(
            iterations=10
        )

        assert positions.shape == (10, 4, 2)
        assert velocities.shape == (10, 4, 2)
        assert accelerations.shape == (10, 4, 2)

        # Static joints should have zero velocity
        assert velocities[:, 0, 0] == pytest.approx(0.0, abs=1e-10)
        assert velocities[:, 0, 1] == pytest.approx(0.0, abs=1e-10)
        assert velocities[:, 3, 0] == pytest.approx(0.0, abs=1e-10)
        assert velocities[:, 3, 1] == pytest.approx(0.0, abs=1e-10)

        # Crank velocity magnitude should be r * omega = 30 * 10 = 300
        crank_vel_mag = np.sqrt(velocities[:, 1, 0] ** 2 + velocities[:, 1, 1] ** 2)
        assert crank_vel_mag == pytest.approx(300.0, rel=1e-6)

    def test_joint_velocity_property(self, four_bar: Linkage) -> None:
        """Test that joint.velocity property is updated after simulation."""
        crank = four_bar.joints[1]
        assert isinstance(crank, Crank)
        four_bar.set_input_velocity(crank, omega=10.0)

        # Before simulation, velocity should be None
        assert four_bar.joints[0].velocity is None

        four_bar.step_fast_with_kinematics(iterations=10)

        # After simulation, velocity should be set
        for joint in four_bar.joints:
            assert joint.velocity is not None
            vx, vy = joint.velocity
            assert not math.isnan(vx)
            assert not math.isnan(vy)

    def test_get_velocities(self, four_bar: Linkage) -> None:
        """Test batch velocity query."""
        crank = four_bar.joints[1]
        assert isinstance(crank, Crank)
        four_bar.set_input_velocity(crank, omega=10.0)

        four_bar.step_fast_with_kinematics(iterations=10)

        velocities = four_bar.get_velocities()
        assert len(velocities) == 4
        for v in velocities:
            assert v is not None
            assert len(v) == 2


class TestVelocityNumericValidation:
    """Validate velocity computation by comparing to finite differences."""

    def test_crank_velocity_matches_finite_difference(self) -> None:
        """Crank velocity should match numerical derivative of position."""
        a = Static(0, 0, name="A")
        b = Crank(30, 0, joint0=a, distance=30, angle=0.1, omega=10.0, name="B")
        linkage = Linkage(joints=(a, b), order=(a, b), name="crank-only")

        # Run simulation
        positions, velocities, _ = linkage.step_fast_with_kinematics(iterations=100)

        # Compute numerical derivative using central differences
        dt = 1.0  # Each step is dt=1
        # Convert angular velocity to per-step basis
        omega_per_step = b.omega if b.omega else 0.0  # noqa: F841

        # For numerical validation, use finite differences
        # v ≈ (x[t+1] - x[t-1]) / (2 * dt)
        for i in range(1, 99):
            numerical_vx = (positions[i + 1, 1, 0] - positions[i - 1, 1, 0]) / 2.0
            numerical_vy = (positions[i + 1, 1, 1] - positions[i - 1, 1, 1]) / 2.0

            # Scale by omega (velocity solver uses rad/s, but simulation steps by angle)
            # The angle increment per step is b.angle (0.1 rad)
            # Physical time per step is angle_increment / omega = 0.1 / 10 = 0.01 s
            physical_dt = (b.angle if b.angle else 0.1) / omega_per_step
            scaled_numerical_vx = numerical_vx / physical_dt
            scaled_numerical_vy = numerical_vy / physical_dt

            # The computed velocity should match scaled numerical derivative
            computed_vx = velocities[i, 1, 0]
            computed_vy = velocities[i, 1, 1]

            # Allow some tolerance due to finite difference approximation
            assert computed_vx == pytest.approx(scaled_numerical_vx, rel=0.05)
            assert computed_vy == pytest.approx(scaled_numerical_vy, rel=0.05)
