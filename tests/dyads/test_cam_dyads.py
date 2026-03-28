"""Tests for cam follower dyads - targeting uncovered lines."""

from __future__ import annotations

import math

import pytest

from pylinkage.actuators import Crank
from pylinkage.cam import (
    CycloidalMotionLaw,
    FunctionProfile,
    HarmonicMotionLaw,
    PointArrayProfile,
)
from pylinkage.components import Ground
from pylinkage.dyads import (
    Linkage,
    OscillatingCamFollower,
    TranslatingCamFollower,
)


class TestOscillatingCamFollowerExtended:
    """Extended tests for OscillatingCamFollower - uncovered lines."""

    def test_output_property(self) -> None:
        O_cam = Ground(0.0, 0.0, name="cam_center")
        O_pivot = Ground(2.0, 0.0, name="pivot")
        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=0.0, total_lift=math.pi / 4)

        follower = OscillatingCamFollower(
            cam_driver=cam,
            profile=profile,
            pivot_anchor=O_pivot,
            arm_length=1.0,
        )
        out = follower.output
        assert out.x == follower.x
        assert out.y == follower.y

    def test_cam_angle_property(self) -> None:
        O_cam = Ground(0.0, 0.0)
        O_pivot = Ground(2.0, 0.0)
        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=0.0, total_lift=math.pi / 4)

        follower = OscillatingCamFollower(
            cam_driver=cam, profile=profile, pivot_anchor=O_pivot, arm_length=1.0
        )
        assert isinstance(follower.cam_angle, float)

    def test_arm_angle_property(self) -> None:
        O_cam = Ground(0.0, 0.0)
        O_pivot = Ground(2.0, 0.0)
        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=0.0, total_lift=math.pi / 4)

        follower = OscillatingCamFollower(
            cam_driver=cam, profile=profile, pivot_anchor=O_pivot, arm_length=1.0
        )
        assert isinstance(follower.arm_angle, float)

    def test_angular_displacement_property(self) -> None:
        O_cam = Ground(0.0, 0.0)
        O_pivot = Ground(2.0, 0.0)
        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1, initial_angle=0.0)
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
            initial_angle=math.pi / 2,
        )
        disp = follower.angular_displacement
        assert isinstance(disp, float)

    def test_anchors_property(self) -> None:
        O_cam = Ground(0.0, 0.0)
        O_pivot = Ground(2.0, 0.0)
        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=0.0, total_lift=math.pi / 4)

        follower = OscillatingCamFollower(
            cam_driver=cam, profile=profile, pivot_anchor=O_pivot, arm_length=1.0
        )
        anchors = follower.anchors
        assert len(anchors) == 2
        assert anchors[0] is cam
        assert anchors[1] is O_pivot

    def test_get_constraints(self) -> None:
        O_cam = Ground(0.0, 0.0)
        O_pivot = Ground(2.0, 0.0)
        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=0.0, total_lift=math.pi / 4)

        follower = OscillatingCamFollower(
            cam_driver=cam,
            profile=profile,
            pivot_anchor=O_pivot,
            arm_length=1.5,
            initial_angle=math.pi / 3,
            roller_radius=0.1,
        )
        constraints = follower.get_constraints()
        assert constraints[0] == 1.5  # arm_length
        assert constraints[1] == pytest.approx(math.pi / 3)  # initial_angle
        assert constraints[2] == 0.1  # roller_radius
        # Remaining are profile constraints
        assert len(constraints) > 3

    def test_set_constraints(self) -> None:
        O_cam = Ground(0.0, 0.0)
        O_pivot = Ground(2.0, 0.0)
        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=0.0, total_lift=math.pi / 4)

        follower = OscillatingCamFollower(
            cam_driver=cam,
            profile=profile,
            pivot_anchor=O_pivot,
            arm_length=1.0,
            roller_radius=0.0,
        )
        follower.set_constraints(2.0, math.pi / 4, 0.05)
        assert follower.arm_length == 2.0
        assert follower.initial_angle == pytest.approx(math.pi / 4)
        assert follower.roller_radius == 0.05

    def test_set_constraints_partial(self) -> None:
        O_cam = Ground(0.0, 0.0)
        O_pivot = Ground(2.0, 0.0)
        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=0.0, total_lift=math.pi / 4)

        follower = OscillatingCamFollower(
            cam_driver=cam,
            profile=profile,
            pivot_anchor=O_pivot,
            arm_length=1.0,
            initial_angle=0.0,
            roller_radius=0.0,
        )
        follower.set_constraints(None, None, None)
        assert follower.arm_length == 1.0
        assert follower.initial_angle == 0.0
        assert follower.roller_radius == 0.0

    def test_set_constraints_with_profile(self) -> None:
        """Test passing profile constraints through set_constraints."""
        O_cam = Ground(0.0, 0.0)
        O_pivot = Ground(2.0, 0.0)
        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)

        follower = OscillatingCamFollower(
            cam_driver=cam, profile=profile, pivot_anchor=O_pivot, arm_length=1.0
        )
        # Pass profile constraints (base_radius, total_lift) after the 3 main ones
        follower.set_constraints(None, None, None, 2.0, 0.75)
        assert profile.base_radius == 2.0
        assert profile.total_lift == 0.75

    def test_invalid_pivot_anchor(self) -> None:
        """Test error when pivot has undefined position."""
        O_cam = Ground(0.0, 0.0)
        O_pivot = Ground(0.0, 0.0, name="bad")
        O_pivot.x = None
        O_pivot.y = None

        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=0.0, total_lift=0.5)

        with pytest.raises(ValueError, match="defined position"):
            OscillatingCamFollower(
                cam_driver=cam,
                profile=profile,
                pivot_anchor=O_pivot,
                arm_length=1.0,
            )

    def test_with_roller_radius(self) -> None:
        O_cam = Ground(0.0, 0.0)
        O_pivot = Ground(2.0, 0.0)
        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=0.0, total_lift=math.pi / 4)

        follower = OscillatingCamFollower(
            cam_driver=cam,
            profile=profile,
            pivot_anchor=O_pivot,
            arm_length=1.0,
            roller_radius=0.2,
        )
        assert follower.roller_radius == 0.2

    def test_reload_with_cam_at_none(self) -> None:
        """Test reload when cam driver has None position."""
        O_cam = Ground(0.0, 0.0)
        O_pivot = Ground(2.0, 0.0)
        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=0.0, total_lift=math.pi / 4)

        follower = OscillatingCamFollower(
            cam_driver=cam, profile=profile, pivot_anchor=O_pivot, arm_length=1.0
        )
        old_x, old_y = follower.x, follower.y

        # Set cam position to None - reload should return early
        cam.x = None
        follower.reload()
        assert follower.x == old_x
        assert follower.y == old_y

    def test_with_point_array_profile(self) -> None:
        """Test oscillating follower with PointArrayProfile."""
        O_cam = Ground(0.0, 0.0)
        O_pivot = Ground(2.0, 0.0)
        cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)

        angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
        radii = [0.0, 0.1, 0.2, 0.1, 0.0]
        profile = PointArrayProfile(angles=angles, radii=radii)

        follower = OscillatingCamFollower(
            cam_driver=cam, profile=profile, pivot_anchor=O_pivot, arm_length=1.0
        )
        assert follower.x is not None
        assert follower.y is not None


class TestTranslatingCamFollowerExtended:
    """Extended tests for TranslatingCamFollower - uncovered lines."""

    def test_output_property(self) -> None:
        origin = Ground(0.0, 0.0)
        cam = Crank(anchor=origin, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)
        guide = Ground(0.0, 0.0)

        follower = TranslatingCamFollower(
            cam_driver=cam, profile=profile, guide=guide
        )
        assert follower.output.x == follower.x
        assert follower.output.y == follower.y

    def test_cam_angle_property(self) -> None:
        origin = Ground(0.0, 0.0)
        cam = Crank(anchor=origin, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)
        guide = Ground(0.0, 0.0)

        follower = TranslatingCamFollower(
            cam_driver=cam, profile=profile, guide=guide
        )
        assert isinstance(follower.cam_angle, float)

    def test_displacement_property(self) -> None:
        origin = Ground(0.0, 0.0)
        cam = Crank(anchor=origin, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)
        guide = Ground(0.0, 0.0)

        follower = TranslatingCamFollower(
            cam_driver=cam, profile=profile, guide=guide
        )
        disp = follower.displacement
        assert isinstance(disp, float)

    def test_anchors_property(self) -> None:
        origin = Ground(0.0, 0.0)
        cam = Crank(anchor=origin, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)
        guide = Ground(1.0, 0.0, name="guide")

        follower = TranslatingCamFollower(
            cam_driver=cam, profile=profile, guide=guide
        )
        anchors = follower.anchors
        assert len(anchors) == 2
        assert anchors[0] is cam
        assert anchors[1] is guide

    def test_get_constraints_with_profile(self) -> None:
        origin = Ground(0.0, 0.0)
        cam = Crank(anchor=origin, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)
        guide = Ground(0.0, 0.0)

        follower = TranslatingCamFollower(
            cam_driver=cam, profile=profile, guide=guide, roller_radius=0.1
        )
        constraints = follower.get_constraints()
        assert constraints[0] == 0.1  # roller_radius
        # Rest are profile constraints
        assert len(constraints) > 1

    def test_set_constraints_with_profile(self) -> None:
        origin = Ground(0.0, 0.0)
        cam = Crank(anchor=origin, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)
        guide = Ground(0.0, 0.0)

        follower = TranslatingCamFollower(
            cam_driver=cam, profile=profile, guide=guide, roller_radius=0.1
        )
        follower.set_constraints(0.2, 2.0, 0.75)
        assert follower.roller_radius == 0.2
        assert profile.base_radius == 2.0
        assert profile.total_lift == 0.75

    def test_set_constraints_partial(self) -> None:
        origin = Ground(0.0, 0.0)
        cam = Crank(anchor=origin, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)
        guide = Ground(0.0, 0.0)

        follower = TranslatingCamFollower(
            cam_driver=cam, profile=profile, guide=guide, roller_radius=0.1
        )
        follower.set_constraints(None)
        assert follower.roller_radius == 0.1

    def test_invalid_guide(self) -> None:
        """Test error when guide has undefined position."""
        origin = Ground(0.0, 0.0)
        cam = Crank(anchor=origin, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)

        guide = Ground(0.0, 0.0, name="bad")
        guide.x = None
        guide.y = None

        with pytest.raises(ValueError, match="defined position"):
            TranslatingCamFollower(
                cam_driver=cam, profile=profile, guide=guide
            )

    def test_roller_follower_reload(self) -> None:
        """Test reload with roller follower (uses pitch radius)."""
        origin = Ground(0.0, 0.0)
        cam = Crank(anchor=origin, radius=0.1, angular_velocity=0.3)
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
            rise_start=0.0,
            rise_end=math.pi,
            dwell_high_end=math.pi,
            fall_end=2 * math.pi,
        )
        guide = Ground(0.0, 0.0)

        follower = TranslatingCamFollower(
            cam_driver=cam, profile=profile, guide=guide,
            guide_angle=math.pi / 2, roller_radius=0.1,
        )

        linkage = Linkage([origin, guide, cam, follower], name="test")
        positions = list(linkage.step(iterations=5))
        assert len(positions) == 5

    def test_reload_with_cam_at_none(self) -> None:
        """Test reload when cam driver has None position."""
        origin = Ground(0.0, 0.0)
        cam = Crank(anchor=origin, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)
        guide = Ground(0.0, 0.0)

        follower = TranslatingCamFollower(
            cam_driver=cam, profile=profile, guide=guide
        )
        old_x, old_y = follower.x, follower.y

        cam.x = None
        follower.reload()
        assert follower.x == old_x
        assert follower.y == old_y

    def test_reload_with_guide_at_none(self) -> None:
        """Test reload when guide has None position."""
        origin = Ground(0.0, 0.0)
        cam = Crank(anchor=origin, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)
        guide = Ground(0.0, 0.0)

        follower = TranslatingCamFollower(
            cam_driver=cam, profile=profile, guide=guide
        )
        old_x, old_y = follower.x, follower.y

        guide.x = None
        follower.reload()
        assert follower.x == old_x
        assert follower.y == old_y

    def test_horizontal_guide(self) -> None:
        """Test follower with horizontal guide direction."""
        origin = Ground(0.0, 0.0)
        cam = Crank(anchor=origin, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)
        guide = Ground(0.0, 0.0)

        follower = TranslatingCamFollower(
            cam_driver=cam, profile=profile, guide=guide, guide_angle=0.0
        )
        # Displacement should be along x
        assert abs(follower.y) < 1e-10

    def test_with_cycloidal_profile(self) -> None:
        origin = Ground(0.0, 0.0)
        cam = Crank(anchor=origin, radius=0.1, angular_velocity=0.1)
        profile = FunctionProfile(
            motion_law=CycloidalMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
        )
        guide = Ground(0.0, 0.0)

        follower = TranslatingCamFollower(
            cam_driver=cam, profile=profile, guide=guide, guide_angle=math.pi / 2
        )
        assert follower.x is not None
        assert follower.y is not None
