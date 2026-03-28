"""Tests for pylinkage.mechanism.joint — Joint classes and their properties.

Covers edge cases and missing lines in joint.py:
- Joint hashing and equality (lines 66, 72)
- Property access: links, x, y setters (lines 83, 93, 103)
- RevoluteJoint.joint_type (line 136)
- PrismaticJoint: joint_type, get_axis_normalized (lines 167, 171-177)
- GroundJoint.joint_type (line 204)
- TrackerJoint.joint_type and update_position (line 240)
"""

import math

import pytest

from pylinkage._types import JointType
from pylinkage.mechanism.joint import (
    GroundJoint,
    Joint,
    PrismaticJoint,
    RevoluteJoint,
    TrackerJoint,
)


class TestJointBasic:
    """Basic Joint creation and properties."""

    def test_default_name_from_id(self):
        j = RevoluteJoint("A", position=(1.0, 2.0))
        assert j.name == "A"

    def test_explicit_name(self):
        j = RevoluteJoint("A", position=(1.0, 2.0), name="custom")
        assert j.name == "custom"

    def test_hash_by_id(self):
        j1 = RevoluteJoint("A", position=(1.0, 2.0))
        j2 = RevoluteJoint("A", position=(5.0, 6.0))
        assert hash(j1) == hash(j2)
        # Different ids should (usually) have different hashes
        j3 = RevoluteJoint("B", position=(1.0, 2.0))
        assert hash(j1) != hash(j3)

    def test_equality_same_id(self):
        j1 = RevoluteJoint("A", position=(1.0, 2.0))
        j2 = RevoluteJoint("A", position=(5.0, 6.0))
        assert j1 == j2

    def test_equality_different_id(self):
        j1 = RevoluteJoint("A")
        j2 = RevoluteJoint("B")
        assert j1 != j2

    def test_equality_non_joint(self):
        j1 = RevoluteJoint("A")
        assert j1 != "A"
        assert j1 != 42

    def test_links_property_empty(self):
        j = RevoluteJoint("A")
        assert j.links == []

    def test_x_property(self):
        j = RevoluteJoint("A", position=(3.0, 4.0))
        assert j.x == 3.0

    def test_y_property(self):
        j = RevoluteJoint("A", position=(3.0, 4.0))
        assert j.y == 4.0

    def test_x_setter(self):
        j = RevoluteJoint("A", position=(3.0, 4.0))
        j.x = 10.0
        assert j.position == (10.0, 4.0)

    def test_y_setter(self):
        j = RevoluteJoint("A", position=(3.0, 4.0))
        j.y = 20.0
        assert j.position == (3.0, 20.0)

    def test_coord(self):
        j = RevoluteJoint("A", position=(1.0, 2.0))
        assert j.coord() == (1.0, 2.0)

    def test_set_coord(self):
        j = RevoluteJoint("A")
        j.set_coord(5.0, 6.0)
        assert j.position == (5.0, 6.0)

    def test_is_defined_true(self):
        j = RevoluteJoint("A", position=(1.0, 2.0))
        assert j.is_defined() is True

    def test_is_defined_false_none(self):
        j = RevoluteJoint("A", position=(None, None))
        assert j.is_defined() is False

    def test_is_defined_partial_none(self):
        j = RevoluteJoint("A", position=(1.0, None))
        assert j.is_defined() is False

    def test_x_none_property(self):
        j = RevoluteJoint("A", position=(None, None))
        assert j.x is None

    def test_x_setter_to_none(self):
        j = RevoluteJoint("A", position=(1.0, 2.0))
        j.x = None
        assert j.position == (None, 2.0)

    def test_y_setter_to_none(self):
        j = RevoluteJoint("A", position=(1.0, 2.0))
        j.y = None
        assert j.position == (1.0, None)


class TestRevoluteJoint:
    def test_joint_type(self):
        j = RevoluteJoint("A", position=(0.0, 0.0))
        assert j.joint_type == JointType.REVOLUTE


class TestGroundJoint:
    def test_joint_type(self):
        j = GroundJoint("O", position=(0.0, 0.0))
        assert j.joint_type == JointType.GROUND

    def test_is_ground_flag(self):
        j = GroundJoint("O", position=(0.0, 0.0))
        assert j.is_ground is True

    def test_inherits_revolute(self):
        j = GroundJoint("O", position=(0.0, 0.0))
        assert isinstance(j, RevoluteJoint)


class TestPrismaticJoint:
    def test_joint_type(self):
        j = PrismaticJoint("S", position=(0.0, 0.0))
        assert j.joint_type == JointType.PRISMATIC

    def test_default_axis(self):
        j = PrismaticJoint("S")
        assert j.axis == (1.0, 0.0)

    def test_get_axis_normalized_unit(self):
        j = PrismaticJoint("S", axis=(1.0, 0.0))
        assert j.get_axis_normalized() == (1.0, 0.0)

    def test_get_axis_normalized_non_unit(self):
        j = PrismaticJoint("S", axis=(3.0, 4.0))
        nx, ny = j.get_axis_normalized()
        assert abs(nx - 0.6) < 1e-10
        assert abs(ny - 0.8) < 1e-10

    def test_get_axis_normalized_degenerate(self):
        """Zero-length axis should default to horizontal."""
        j = PrismaticJoint("S", axis=(0.0, 0.0))
        assert j.get_axis_normalized() == (1.0, 0.0)

    def test_get_axis_normalized_near_zero(self):
        j = PrismaticJoint("S", axis=(1e-12, 1e-12))
        assert j.get_axis_normalized() == (1.0, 0.0)

    def test_slide_distance(self):
        j = PrismaticJoint("S", slide_distance=5.0)
        assert j.slide_distance == 5.0

    def test_line_point(self):
        j = PrismaticJoint("S", line_point=(1.0, 2.0))
        assert j.line_point == (1.0, 2.0)


class TestTrackerJoint:
    def test_joint_type(self):
        j = TrackerJoint("T")
        assert j.joint_type == JointType.TRACKER

    def test_default_attributes(self):
        j = TrackerJoint("T")
        assert j.ref_joint1_id == ""
        assert j.ref_joint2_id == ""
        assert j.distance == 0.0
        assert j.angle == 0.0

    def test_update_position_zero_angle(self):
        """Tracker at same direction as ref1->ref2."""
        j = TrackerJoint("T", ref_joint1_id="A", ref_joint2_id="B", distance=2.0, angle=0.0)
        j.update_position((0.0, 0.0), (1.0, 0.0))
        # Should be 2.0 along the x-axis from ref1
        assert abs(j.x - 2.0) < 1e-10
        assert abs(j.y - 0.0) < 1e-10

    def test_update_position_with_angle(self):
        """Tracker at 90 degrees from ref1->ref2 direction."""
        j = TrackerJoint("T", ref_joint1_id="A", ref_joint2_id="B", distance=1.0, angle=math.pi / 2)
        j.update_position((0.0, 0.0), (1.0, 0.0))
        # 90 degrees from horizontal => straight up
        assert abs(j.x - 0.0) < 1e-10
        assert abs(j.y - 1.0) < 1e-10

    def test_update_position_diagonal_reference(self):
        """Tracker with diagonal reference direction."""
        j = TrackerJoint("T", ref_joint1_id="A", ref_joint2_id="B", distance=1.0, angle=0.0)
        j.update_position((0.0, 0.0), (1.0, 1.0))
        # base_angle = pi/4, so position at (cos(pi/4), sin(pi/4))
        expected_x = math.cos(math.pi / 4)
        expected_y = math.sin(math.pi / 4)
        assert abs(j.x - expected_x) < 1e-10
        assert abs(j.y - expected_y) < 1e-10


class TestJointUsedInSets:
    """Joint should work correctly in sets and as dict keys."""

    def test_in_set(self):
        j1 = RevoluteJoint("A")
        j2 = RevoluteJoint("A")
        s = {j1, j2}
        assert len(s) == 1

    def test_as_dict_key(self):
        j = RevoluteJoint("A")
        d = {j: "value"}
        j2 = RevoluteJoint("A")
        assert d[j2] == "value"
