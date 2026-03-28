"""Tests for mechanism/link.py — Link, DriverLink, GroundLink, ArcDriverLink."""

import math

import pytest

from pylinkage.mechanism import (
    DriverLink,
    GroundJoint,
    GroundLink,
    Link,
    RevoluteJoint,
)
from pylinkage.mechanism.link import ArcDriverLink, LinkType


class TestLinkProperties:
    """Tests for Link properties."""

    def test_link_type_binary(self):
        j1 = RevoluteJoint("A", position=(0.0, 0.0))
        j2 = RevoluteJoint("B", position=(1.0, 0.0))
        link = Link("AB", joints=[j1, j2])
        assert link.link_type == LinkType.BINARY

    def test_link_type_ternary(self):
        j1 = RevoluteJoint("A", position=(0.0, 0.0))
        j2 = RevoluteJoint("B", position=(1.0, 0.0))
        j3 = RevoluteJoint("C", position=(0.5, 1.0))
        link = Link("ABC", joints=[j1, j2, j3])
        assert link.link_type == LinkType.TERNARY

    def test_link_type_quaternary(self):
        joints = [RevoluteJoint(f"J{i}", position=(float(i), 0.0)) for i in range(4)]
        link = Link("Q", joints=joints)
        assert link.link_type == LinkType.QUATERNARY

    def test_link_type_default_for_single(self):
        j = RevoluteJoint("A", position=(0.0, 0.0))
        link = Link("single", joints=[j])
        assert link.link_type == LinkType.BINARY  # default

    def test_order(self):
        j1 = RevoluteJoint("A", position=(0.0, 0.0))
        j2 = RevoluteJoint("B", position=(1.0, 0.0))
        link = Link("AB", joints=[j1, j2])
        assert link.order == 2

    def test_length_binary(self):
        j1 = RevoluteJoint("A", position=(0.0, 0.0))
        j2 = RevoluteJoint("B", position=(3.0, 4.0))
        link = Link("AB", joints=[j1, j2])
        assert abs(link.length - 5.0) < 1e-10

    def test_length_non_binary(self):
        joints = [RevoluteJoint(f"J{i}", position=(float(i), 0.0)) for i in range(3)]
        link = Link("ternary", joints=joints)
        assert link.length is None

    def test_length_undefined_position(self):
        j1 = RevoluteJoint("A", position=(0.0, 0.0))
        j2 = RevoluteJoint("B", position=(None, None))
        link = Link("AB", joints=[j1, j2])
        assert link.length is None

    def test_name_defaults_to_id(self):
        link = Link("mylink", joints=[])
        assert link.name == "mylink"

    def test_custom_name(self):
        link = Link("AB", joints=[], name="Coupler")
        assert link.name == "Coupler"

    def test_hash_by_id(self):
        l1 = Link("AB", joints=[])
        l2 = Link("AB", joints=[])
        assert hash(l1) == hash(l2)
        assert l1 == l2

    def test_not_equal_different_id(self):
        l1 = Link("AB", joints=[])
        l2 = Link("CD", joints=[])
        assert l1 != l2

    def test_not_equal_non_link(self):
        l1 = Link("AB", joints=[])
        assert l1 != "AB"


class TestLinkDistances:
    """Tests for cache_distances() and get_distance()."""

    def test_cache_distances(self):
        j1 = RevoluteJoint("A", position=(0.0, 0.0))
        j2 = RevoluteJoint("B", position=(3.0, 4.0))
        link = Link("AB", joints=[j1, j2])
        link.cache_distances()
        assert abs(link.get_distance(j1, j2) - 5.0) < 1e-10
        # Symmetric
        assert abs(link.get_distance(j2, j1) - 5.0) < 1e-10

    def test_get_distance_fallback_to_current(self):
        """When cache is empty, compute from current positions."""
        j1 = RevoluteJoint("A", position=(0.0, 0.0))
        j2 = RevoluteJoint("B", position=(1.0, 0.0))
        link = Link("AB", joints=[j1, j2])
        link._cached_distances.clear()
        assert abs(link.get_distance(j1, j2) - 1.0) < 1e-10

    def test_get_distance_undefined(self):
        j1 = RevoluteJoint("A", position=(0.0, 0.0))
        j2 = RevoluteJoint("B", position=(None, None))
        link = Link("AB", joints=[j1, j2])
        link._cached_distances.clear()
        assert link.get_distance(j1, j2) is None

    def test_get_distance_joint_not_in_link(self):
        j1 = RevoluteJoint("A", position=(0.0, 0.0))
        j2 = RevoluteJoint("B", position=(1.0, 0.0))
        j3 = RevoluteJoint("C", position=(2.0, 0.0))
        link = Link("AB", joints=[j1, j2])
        with pytest.raises(ValueError, match="not part of link"):
            link.get_distance(j1, j3)


class TestOtherJoint:
    """Tests for Link.other_joint()."""

    def test_other_joint_binary(self):
        j1 = RevoluteJoint("A", position=(0.0, 0.0))
        j2 = RevoluteJoint("B", position=(1.0, 0.0))
        link = Link("AB", joints=[j1, j2])
        assert link.other_joint(j1) == j2
        assert link.other_joint(j2) == j1

    def test_other_joint_non_binary(self):
        joints = [RevoluteJoint(f"J{i}", position=(float(i), 0.0)) for i in range(3)]
        link = Link("ternary", joints=joints)
        assert link.other_joint(joints[0]) is None

    def test_other_joint_not_in_link(self):
        j1 = RevoluteJoint("A", position=(0.0, 0.0))
        j2 = RevoluteJoint("B", position=(1.0, 0.0))
        j3 = RevoluteJoint("C", position=(2.0, 0.0))
        link = Link("AB", joints=[j1, j2])
        assert link.other_joint(j3) is None


class TestGroundLinkType:
    """Tests for GroundLink.link_type."""

    def test_ground_link_type(self):
        j = GroundJoint("O", position=(0.0, 0.0))
        link = GroundLink("ground", joints=[j])
        assert link.link_type == LinkType.GROUND


class TestDriverLinkProperties:
    """Tests for DriverLink properties and methods."""

    def test_link_type(self):
        origin = GroundJoint("O", position=(0.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        d = DriverLink("crank", joints=[origin, A], motor_joint=origin)
        assert d.link_type == LinkType.DRIVER

    def test_radius(self):
        origin = GroundJoint("O", position=(0.0, 0.0))
        A = RevoluteJoint("A", position=(3.0, 4.0))
        d = DriverLink("crank", joints=[origin, A], motor_joint=origin)
        assert abs(d.radius - 5.0) < 1e-10

    def test_radius_no_motor(self):
        origin = GroundJoint("O", position=(0.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        d = DriverLink("crank", joints=[origin, A], motor_joint=None)
        assert d.radius is None

    def test_radius_non_binary(self):
        origin = GroundJoint("O", position=(0.0, 0.0))
        d = DriverLink("crank", joints=[origin], motor_joint=origin)
        assert d.radius is None

    def test_output_joint(self):
        origin = GroundJoint("O", position=(0.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        d = DriverLink("crank", joints=[origin, A], motor_joint=origin)
        assert d.output_joint == A

    def test_output_joint_no_motor(self):
        d = DriverLink("crank", joints=[], motor_joint=None)
        assert d.output_joint is None

    def test_step_rotates(self):
        origin = GroundJoint("O", position=(0.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        d = DriverLink(
            "crank", joints=[origin, A], motor_joint=origin, angular_velocity=math.pi / 2
        )
        d.step(dt=1.0)
        assert abs(A.x) < 1e-10
        assert abs(A.y - 1.0) < 1e-10

    def test_step_no_output(self):
        """Step with no output joint should not crash."""
        origin = GroundJoint("O", position=(0.0, 0.0))
        d = DriverLink("crank", joints=[origin], motor_joint=None)
        d.step(dt=1.0)  # Should not raise

    def test_reset(self):
        origin = GroundJoint("O", position=(0.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        d = DriverLink(
            "crank", joints=[origin, A], motor_joint=origin, angular_velocity=0.5, initial_angle=0.3
        )
        d.step(1.0)
        d.step(1.0)
        assert d.current_angle != 0.3
        d.reset()
        assert d.current_angle == 0.3


class TestArcDriverLink:
    """Tests for ArcDriverLink."""

    def _make_arc(self, arc_start=0.0, arc_end=math.pi, omega=0.5):
        origin = GroundJoint("O", position=(0.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        return ArcDriverLink(
            "arc",
            joints=[origin, A],
            motor_joint=origin,
            angular_velocity=omega,
            arc_start=arc_start,
            arc_end=arc_end,
        )

    def test_link_type(self):
        arc = self._make_arc()
        assert arc.link_type == LinkType.DRIVER

    def test_initial_angle_defaults_to_arc_start(self):
        arc = self._make_arc(arc_start=0.5)
        assert arc.initial_angle == 0.5
        assert arc.current_angle == 0.5

    def test_radius(self):
        arc = self._make_arc()
        assert abs(arc.radius - 1.0) < 1e-10

    def test_output_joint(self):
        origin = GroundJoint("O", position=(0.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        arc = ArcDriverLink("arc", joints=[origin, A], motor_joint=origin)
        assert arc.output_joint == A

    def test_step_within_bounds(self):
        arc = self._make_arc(arc_start=0.0, arc_end=2.0, omega=0.1)
        arc.step(dt=1.0)
        assert 0.0 <= arc.current_angle <= 2.0

    def test_step_bounces_at_arc_end(self):
        arc = self._make_arc(arc_start=0.0, arc_end=1.0, omega=0.9)
        arc.current_angle = 0.9
        arc.step(dt=1.0)  # Would go to 1.8, should bounce
        assert arc.current_angle <= 1.0
        assert arc._direction == -1.0

    def test_step_bounces_at_arc_start(self):
        arc = self._make_arc(arc_start=0.0, arc_end=1.0, omega=0.9)
        arc._direction = -1.0
        arc.current_angle = 0.1
        arc.step(dt=1.0)  # Would go to -0.8, should bounce
        assert arc.current_angle >= 0.0
        assert arc._direction == 1.0

    def test_reset(self):
        arc = self._make_arc(arc_start=0.5, arc_end=2.0, omega=0.3)
        arc.step(1.0)
        arc.step(1.0)
        arc.reset()
        assert arc.current_angle == 0.5
        assert arc._direction == 1.0

    def test_radius_no_motor(self):
        origin = GroundJoint("O", position=(0.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        arc = ArcDriverLink("arc", joints=[origin, A], motor_joint=None)
        assert arc.radius is None

    def test_radius_non_binary(self):
        origin = GroundJoint("O", position=(0.0, 0.0))
        arc = ArcDriverLink("arc", joints=[origin], motor_joint=origin)
        assert arc.radius is None

    def test_step_no_output(self):
        """Step with degenerate setup should not crash."""
        origin = GroundJoint("O", position=(0.0, 0.0))
        arc = ArcDriverLink("arc", joints=[origin], motor_joint=None)
        arc.step(dt=1.0)  # Should not raise
