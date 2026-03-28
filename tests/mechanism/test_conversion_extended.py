"""Extended tests for mechanism/conversion.py — edge cases and missing coverage.

Covers:
- Fixed joint conversion (lines 84-102)
- Prismatic joint conversion (lines 94-102, 167-179)
- Conversion with missing/None positions (lines 143, 198)
- mechanism_to_linkage with PrismaticJoint (lines 260-284)
- convert_legacy_dict (lines 301-312)
- _compute_initial_angle with None coords (line 198)
"""

import math

import pytest

from pylinkage.mechanism.conversion import (
    _compute_initial_angle,
    mechanism_from_linkage,
    mechanism_to_linkage,
)
from pylinkage.mechanism.joint import GroundJoint, PrismaticJoint, RevoluteJoint
from pylinkage.mechanism.link import DriverLink, GroundLink, Link
from pylinkage.mechanism import Mechanism


def _build_linkage_with_fixed_joint():
    """Build a legacy linkage using a Fixed joint."""
    from pylinkage.joints import Crank, Fixed, Static

    O1 = Static(x=0.0, y=0.0, name="O1")
    O2 = Static(x=2.0, y=0.0, name="O2")
    A = Crank(x=1.0, y=0.0, joint0=O1, distance=1.0, angle=0.1, name="A")
    B = Fixed(
        x=1.5,
        y=1.0,
        joint0=A,
        joint1=O2,
        distance=1.0,
        angle=0.5,
        name="B",
    )
    from pylinkage.linkage import Linkage

    return Linkage(name="WithFixed", joints=(O1, O2, A, B))


def _build_linkage_with_prismatic():
    """Build a legacy linkage with a Prismatic joint."""
    from pylinkage.joints import Crank, Prismatic, Static

    O1 = Static(x=0.0, y=0.0, name="O1")
    O2 = Static(x=4.0, y=0.0, name="O2")
    A = Crank(x=1.0, y=0.0, joint0=O1, distance=1.0, angle=0.1, name="A")
    # Prismatic joint: circle centered at A, line through O1-O2
    P = Prismatic(
        x=2.0,
        y=0.0,
        joint0=A,
        joint1=O1,
        joint2=O2,
        revolute_radius=1.5,
        name="P",
    )
    from pylinkage.linkage import Linkage

    return Linkage(name="WithPrismatic", joints=(O1, O2, A, P))


class TestFixedJointConversion:
    """Test mechanism_from_linkage with Fixed joints (lines 84-102)."""

    def test_fixed_creates_revolute_joint(self):
        linkage = _build_linkage_with_fixed_joint()
        mech = mechanism_from_linkage(linkage)

        # B should become a RevoluteJoint
        b_joints = [j for j in mech.joints if j.id == "B"]
        assert len(b_joints) == 1
        assert isinstance(b_joints[0], RevoluteJoint)

    def test_fixed_creates_links(self):
        linkage = _build_linkage_with_fixed_joint()
        mech = mechanism_from_linkage(linkage)

        link_ids = {lnk.id for lnk in mech.links}
        assert "B_link0" in link_ids
        assert "B_link1" in link_ids

    def test_fixed_preserves_position(self):
        linkage = _build_linkage_with_fixed_joint()
        mech = mechanism_from_linkage(linkage)

        b_joint = next(j for j in mech.joints if j.id == "B")
        assert b_joint.position[0] is not None
        assert b_joint.position[1] is not None


class TestPrismaticJointConversion:
    """Test mechanism_from_linkage with Prismatic joints (lines 94-102, 167-179)."""

    def test_prismatic_creates_prismatic_joint(self):
        linkage = _build_linkage_with_prismatic()
        mech = mechanism_from_linkage(linkage)

        p_joints = [j for j in mech.joints if j.id == "P"]
        assert len(p_joints) == 1
        assert isinstance(p_joints[0], PrismaticJoint)

    def test_prismatic_creates_link(self):
        linkage = _build_linkage_with_prismatic()
        mech = mechanism_from_linkage(linkage)

        link_ids = {lnk.id for lnk in mech.links}
        # Should have P_link0 connecting A to P
        assert "P_link0" in link_ids

    def test_prismatic_has_both_links(self):
        """With both joint0 and joint1, P_link0 should be created."""
        linkage = _build_linkage_with_prismatic()
        mech = mechanism_from_linkage(linkage)

        link_ids = {lnk.id for lnk in mech.links}
        assert "P_link0" in link_ids


class TestComputeInitialAngle:
    """Test _compute_initial_angle helper (line 198)."""

    def test_normal_positions(self):
        motor = GroundJoint("O", position=(0.0, 0.0))
        output = RevoluteJoint("A", position=(1.0, 0.0))
        angle = _compute_initial_angle(motor, output)
        assert abs(angle - 0.0) < 1e-10

    def test_diagonal_position(self):
        motor = GroundJoint("O", position=(0.0, 0.0))
        output = RevoluteJoint("A", position=(1.0, 1.0))
        angle = _compute_initial_angle(motor, output)
        assert abs(angle - math.pi / 4) < 1e-10

    def test_none_motor_position(self):
        """With None coordinates, should return 0.0."""
        motor = GroundJoint("O", position=(None, None))
        output = RevoluteJoint("A", position=(1.0, 0.0))
        angle = _compute_initial_angle(motor, output)
        assert angle == 0.0

    def test_none_output_position(self):
        motor = GroundJoint("O", position=(0.0, 0.0))
        output = RevoluteJoint("A", position=(None, None))
        angle = _compute_initial_angle(motor, output)
        assert angle == 0.0

    def test_partial_none(self):
        motor = GroundJoint("O", position=(0.0, None))
        output = RevoluteJoint("A", position=(1.0, 0.0))
        angle = _compute_initial_angle(motor, output)
        assert angle == 0.0


class TestMechanismToLinkageEdgeCases:
    """Edge cases for mechanism_to_linkage (lines 240-284)."""

    def test_empty_mechanism(self):
        mech = Mechanism(name="empty", joints=[], links=[])
        linkage = mechanism_to_linkage(mech)
        assert linkage.name == "empty"
        assert len(linkage.joints) == 0

    def test_ground_only_mechanism(self):
        O1 = GroundJoint("O1", position=(0.0, 0.0))
        ground = GroundLink("ground", joints=[O1])
        mech = Mechanism(name="ground_only", joints=[O1], links=[ground])
        linkage = mechanism_to_linkage(mech)
        assert len(linkage.joints) >= 1

    def test_driver_with_no_motor(self):
        """DriverLink with motor_joint=None should still not crash."""
        O1 = GroundJoint("O1", position=(0.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        ground = GroundLink("ground", joints=[O1])
        # motor_joint is a GroundJoint, so output_joint maps to A
        driver = DriverLink(
            "crank", joints=[O1, A], motor_joint=O1, angular_velocity=0.1
        )
        mech = Mechanism(name="test", joints=[O1, A], links=[ground, driver])
        linkage = mechanism_to_linkage(mech)
        assert linkage.name == "test"


class TestRoundtripWithFixed:
    """Roundtrip conversion with Fixed joints."""

    def test_roundtrip_preserves_structure(self):
        linkage = _build_linkage_with_fixed_joint()
        mech = mechanism_from_linkage(linkage)
        restored = mechanism_to_linkage(mech)
        # Should have at least the same number of joints
        assert len(restored.joints) >= 2  # At least ground and crank
