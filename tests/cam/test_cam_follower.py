"""Tests for cam-follower mechanisms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pylinkage.actuators import Crank
from pylinkage.cam import (
    CycloidalMotionLaw,
    FunctionProfile,
    HarmonicMotionLaw,
    ModifiedTrapezoidalMotionLaw,
    PointArrayProfile,
    PolynomialMotionLaw,
    polynomial_345,
)
from pylinkage.components import Ground
from pylinkage.dyads import (
    Linkage,
    OscillatingCamFollower,
    RRRDyad,
    TranslatingCamFollower,
)


class TestMotionLaws:
    """Tests for motion law implementations."""

    def test_harmonic_boundary_conditions(self) -> None:
        """Test harmonic motion law at boundaries."""
        law = HarmonicMotionLaw()

        # At u=0: displacement = 0
        assert abs(law.displacement(0.0)) < 1e-10

        # At u=0.5: displacement = 0.5
        assert abs(law.displacement(0.5) - 0.5) < 1e-10

        # At u=1: displacement = 1
        assert abs(law.displacement(1.0) - 1.0) < 1e-10

    def test_harmonic_velocity(self) -> None:
        """Test harmonic motion law velocity."""
        law = HarmonicMotionLaw()

        # Maximum velocity at u=0.5
        v_mid = law.velocity(0.5)
        v_start = law.velocity(0.0)
        v_end = law.velocity(1.0)

        assert v_mid > v_start
        assert v_mid > v_end

    def test_cycloidal_boundary_conditions(self) -> None:
        """Test cycloidal motion law at boundaries."""
        law = CycloidalMotionLaw()

        # At u=0: displacement = 0, velocity = 0
        assert abs(law.displacement(0.0)) < 1e-10
        assert abs(law.velocity(0.0)) < 1e-10

        # At u=1: displacement = 1, velocity = 0
        assert abs(law.displacement(1.0) - 1.0) < 1e-10
        assert abs(law.velocity(1.0)) < 1e-10

    def test_cycloidal_velocity(self) -> None:
        """Test cycloidal motion law has zero velocity at endpoints."""
        law = CycloidalMotionLaw()

        # Velocity should be zero at both ends
        assert abs(law.velocity(0.0)) < 1e-10
        assert abs(law.velocity(1.0)) < 1e-10

        # Maximum velocity at u=0.5
        assert law.velocity(0.5) > 0

    def test_polynomial_345_boundary_conditions(self) -> None:
        """Test 3-4-5 polynomial motion law at boundaries."""
        law = polynomial_345()

        # At u=0: displacement = 0, velocity = 0
        assert abs(law.displacement(0.0)) < 1e-10
        assert abs(law.velocity(0.0)) < 1e-10

        # At u=1: displacement = 1, velocity = 0
        assert abs(law.displacement(1.0) - 1.0) < 1e-10
        assert abs(law.velocity(1.0)) < 1e-10

    def test_modified_trapezoidal_boundary_conditions(self) -> None:
        """Test modified trapezoidal motion law at boundaries."""
        law = ModifiedTrapezoidalMotionLaw()

        # At u=0: displacement = 0
        assert abs(law.displacement(0.0)) < 1e-10

        # At u=1: displacement = 1
        assert abs(law.displacement(1.0) - 1.0) < 1e-10


class TestFunctionProfile:
    """Tests for function-based cam profiles."""

    def test_profile_creation(self) -> None:
        """Test creating a function profile."""
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
            rise_start=0.0,
            rise_end=math.pi / 2,
            dwell_high_end=math.pi,
            fall_end=3 * math.pi / 2,
        )

        assert profile.base_radius == 1.0
        assert profile.total_lift == 0.5

    def test_profile_evaluation_dwell_low(self) -> None:
        """Test profile at dwell-low phase."""
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
            rise_start=math.pi / 4,
            rise_end=math.pi / 2,
        )

        # Before rise starts, should be at base radius
        radius = profile.evaluate(0.0)
        assert abs(radius - 1.0) < 1e-10

    def test_profile_evaluation_dwell_high(self) -> None:
        """Test profile at dwell-high phase."""
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
            rise_start=0.0,
            rise_end=math.pi / 2,
            dwell_high_end=math.pi,
            fall_end=3 * math.pi / 2,
        )

        # During dwell-high, should be at base + lift
        radius = profile.evaluate(3 * math.pi / 4)
        assert abs(radius - 1.5) < 1e-10

    def test_profile_evaluation_rise(self) -> None:
        """Test profile during rise phase."""
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
            rise_start=0.0,
            rise_end=math.pi,
        )

        # At midpoint of rise (u=0.5), displacement should be 0.5
        radius = profile.evaluate(math.pi / 2)
        expected = 1.0 + 0.5 * 0.5  # base + lift * 0.5
        assert abs(radius - expected) < 1e-10

    def test_profile_derivative_dwell(self) -> None:
        """Test profile derivative during dwell phases."""
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
            rise_start=math.pi / 4,
            rise_end=math.pi / 2,
        )

        # During dwell, derivative should be zero
        deriv = profile.evaluate_derivative(0.0)
        assert abs(deriv) < 1e-10

    def test_profile_constraints(self) -> None:
        """Test get/set constraints."""
        profile = FunctionProfile(
            base_radius=1.0,
            total_lift=0.5,
        )

        constraints = profile.get_constraints()
        assert constraints == (1.0, 0.5)

        profile.set_constraints(1.5, 0.75)
        assert profile.base_radius == 1.5
        assert profile.total_lift == 0.75


class TestPointArrayProfile:
    """Tests for point array cam profiles."""

    def test_profile_creation(self) -> None:
        """Test creating a point array profile."""
        angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
        radii = [1.0, 1.25, 1.5, 1.25, 1.0]

        profile = PointArrayProfile(angles=angles, radii=radii)

        assert profile.base_radius == 1.0
        assert len(profile.angles) == 5

    def test_profile_evaluation_at_knots(self) -> None:
        """Test profile evaluation at knot points."""
        angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
        radii = [1.0, 1.25, 1.5, 1.25, 1.0]

        profile = PointArrayProfile(angles=angles, radii=radii)

        # At knot points, should match given radii
        assert abs(profile.evaluate(0.0) - 1.0) < 1e-6
        assert abs(profile.evaluate(math.pi / 2) - 1.25) < 1e-6
        assert abs(profile.evaluate(math.pi) - 1.5) < 1e-6

    def test_profile_interpolation(self) -> None:
        """Test profile interpolation between knots."""
        angles = [0, math.pi, 2 * math.pi]
        radii = [1.0, 2.0, 1.0]

        profile = PointArrayProfile(angles=angles, radii=radii)

        # At midpoint, should be between 1.0 and 2.0
        mid_radius = profile.evaluate(math.pi / 2)
        assert 1.0 < mid_radius < 2.0

    def test_profile_too_few_points(self) -> None:
        """Test error with too few points."""
        with pytest.raises(ValueError, match="at least 3 points"):
            PointArrayProfile(angles=[0, math.pi], radii=[1.0, 1.5])


class TestTranslatingCamFollower:
    """Tests for translating cam follower."""

    def test_creation(self) -> None:
        """Test creating a translating cam follower."""
        O = Ground(0.0, 0.0, name="O")
        cam = Crank(anchor=O, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)
        guide = Ground(0.0, 0.0, name="guide")

        follower = TranslatingCamFollower(
            cam_driver=cam,
            profile=profile,
            guide=guide,
            guide_angle=math.pi / 2,
            roller_radius=0.0,
        )

        assert follower.guide_angle == math.pi / 2
        assert follower.roller_radius == 0.0

    def test_knife_edge_vs_roller(self) -> None:
        """Test knife-edge vs roller follower."""
        O = Ground(0.0, 0.0, name="O")
        cam = Crank(anchor=O, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)
        guide = Ground(0.0, 0.0, name="guide")

        # Knife-edge
        knife = TranslatingCamFollower(
            cam_driver=cam,
            profile=profile,
            guide=guide,
            guide_angle=math.pi / 2,
            roller_radius=0.0,
        )

        # Roller
        roller = TranslatingCamFollower(
            cam_driver=cam,
            profile=profile,
            guide=guide,
            guide_angle=math.pi / 2,
            roller_radius=0.1,
        )

        # Roller should be further from guide due to roller radius
        assert roller.y > knife.y

    def test_simulation(self) -> None:
        """Test simulating a translating cam mechanism."""
        O = Ground(0.0, 0.0, name="O")
        cam = Crank(anchor=O, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
            rise_start=0.0,
            rise_end=math.pi,
            dwell_high_end=math.pi,
            fall_end=2 * math.pi,
        )
        guide = Ground(0.0, 0.0, name="guide")

        follower = TranslatingCamFollower(
            cam_driver=cam,
            profile=profile,
            guide=guide,
            guide_angle=math.pi / 2,
        )

        linkage = Linkage([O, guide, cam, follower], name="Cam-Follower")

        # Run for a few steps
        positions = list(linkage.step(iterations=10))
        assert len(positions) == 10

        # Follower should move
        initial_y = positions[0][3][1]
        final_y = positions[-1][3][1]
        assert initial_y != final_y

    def test_constraints(self) -> None:
        """Test get/set constraints."""
        O = Ground(0.0, 0.0, name="O")
        cam = Crank(anchor=O, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)
        guide = Ground(0.0, 0.0, name="guide")

        follower = TranslatingCamFollower(
            cam_driver=cam,
            profile=profile,
            guide=guide,
            roller_radius=0.1,
        )

        constraints = follower.get_constraints()
        assert constraints[0] == 0.1  # roller_radius

        follower.set_constraints(0.2)
        assert follower.roller_radius == 0.2


class TestOscillatingCamFollower:
    """Tests for oscillating cam follower."""

    def test_creation(self) -> None:
        """Test creating an oscillating cam follower."""
        O_cam = Ground(0.0, 0.0, name="cam_center")
        O_pivot = Ground(2.0, 0.0, name="pivot")
        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(
            base_radius=0.0,
            total_lift=math.pi / 4,
        )

        follower = OscillatingCamFollower(
            cam_driver=cam,
            profile=profile,
            pivot_anchor=O_pivot,
            arm_length=1.0,
            initial_angle=math.pi / 2,
        )

        assert follower.arm_length == 1.0
        assert follower.initial_angle == math.pi / 2

    def test_arm_position(self) -> None:
        """Test that arm position is correct."""
        O_cam = Ground(0.0, 0.0, name="cam_center")
        O_pivot = Ground(2.0, 0.0, name="pivot")
        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1, initial_angle=0.0)

        # Profile with zero offset at angle=0
        profile = FunctionProfile(
            base_radius=0.0,
            total_lift=math.pi / 4,
            rise_start=0.0,
            rise_end=math.pi,
        )

        follower = OscillatingCamFollower(
            cam_driver=cam,
            profile=profile,
            pivot_anchor=O_pivot,
            arm_length=1.0,
            initial_angle=0.0,  # Horizontal
        )

        # At initial position, arm should be horizontal
        # Output at (pivot_x + arm_length, pivot_y)
        assert abs(follower.x - 3.0) < 1e-6
        assert abs(follower.y - 0.0) < 1e-6

    def test_simulation(self) -> None:
        """Test simulating an oscillating cam mechanism."""
        O_cam = Ground(0.0, 0.0, name="cam_center")
        O_pivot = Ground(3.0, 0.0, name="pivot")
        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(
            motion_law=CycloidalMotionLaw(),
            base_radius=0.0,
            total_lift=math.pi / 6,  # 30 degree swing
            rise_start=0.0,
            rise_end=math.pi,
            dwell_high_end=math.pi,
            fall_end=2 * math.pi,
        )

        follower = OscillatingCamFollower(
            cam_driver=cam,
            profile=profile,
            pivot_anchor=O_pivot,
            arm_length=1.5,
            initial_angle=math.pi / 2,
        )

        linkage = Linkage([O_cam, O_pivot, cam, follower], name="Oscillating Cam")

        # Run for a few steps
        positions = list(linkage.step(iterations=10))
        assert len(positions) == 10

        # Follower arm angle should change
        initial_x = positions[0][3][0]
        final_x = positions[-1][3][0]
        assert initial_x != final_x


class TestCamFollowerWithLinkage:
    """Integration tests for cam-follower driving other linkage elements."""

    def test_cam_driving_rocker(self) -> None:
        """Test cam-follower driving a rocker via RRRDyad."""
        O_cam = Ground(0.0, 0.0, name="cam_center")
        O_rocker = Ground(3.0, 0.0, name="rocker_pivot")
        guide = Ground(0.0, 0.0, name="guide")

        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.5,
            total_lift=0.5,
        )

        follower = TranslatingCamFollower(
            cam_driver=cam,
            profile=profile,
            guide=guide,
            guide_angle=math.pi / 2,
        )

        # Geometry check: follower at (0, 1.5), pivot at (3, 0)
        # Distance between them: sqrt(9 + 2.25) = sqrt(11.25) ≈ 3.35
        # Need: |d1 - d2| <= 3.35 <= d1 + d2
        rocker = RRRDyad(
            anchor1=follower.output,
            anchor2=O_rocker,
            distance1=2.5,
            distance2=2.0,
        )

        linkage = Linkage(
            [O_cam, O_rocker, guide, cam, follower, rocker],
            name="Cam-Driven Rocker",
        )

        # Run simulation
        positions = list(linkage.step(iterations=20))
        assert len(positions) == 20

        # Rocker should move as follower moves
        rocker_positions = [p[5] for p in positions]
        x_values = [p[0] for p in rocker_positions]

        # Check that rocker moved (not all same position)
        assert len(set(x_values)) > 1


class TestEdgeCases:
    """Edge case tests for cam-follower mechanisms."""

    def test_zero_lift(self) -> None:
        """Test cam with zero lift (constant radius)."""
        profile = FunctionProfile(
            base_radius=1.0,
            total_lift=0.0,
        )

        # Should always return base radius
        for angle in [0, math.pi / 4, math.pi / 2, math.pi]:
            assert abs(profile.evaluate(angle) - 1.0) < 1e-10

    def test_full_rotation_continuity(self) -> None:
        """Test that profile is continuous over full rotation."""
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
        )

        # Sample at many points and check for jumps
        angles = np.linspace(0, 2 * math.pi, 100)
        radii = [profile.evaluate(a) for a in angles]

        # Maximum change between adjacent samples should be small
        max_diff = max(abs(radii[i + 1] - radii[i]) for i in range(len(radii) - 1))
        assert max_diff < 0.1  # Reasonable bound for smooth profile

    def test_negative_angle_evaluation(self) -> None:
        """Test profile evaluation at negative angles."""
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
        )

        # Negative angles should wrap to positive
        r_neg = profile.evaluate(-math.pi / 4)
        r_pos = profile.evaluate(2 * math.pi - math.pi / 4)

        assert abs(r_neg - r_pos) < 1e-10
