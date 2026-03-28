"""Tests for mechanism/conversion.py — converting between Mechanism and Linkage."""

import math

from pylinkage.mechanism import (
    DriverLink,
    GroundJoint,
    GroundLink,
    Mechanism,
    RevoluteJoint,
    mechanism_from_linkage,
    mechanism_to_linkage,
)


def _build_legacy_four_bar():
    """Build a four-bar linkage using the legacy joints API."""
    from pylinkage.joints import Crank, Revolute, Static

    O1 = Static(x=0.0, y=0.0, name="O1")
    O2 = Static(x=2.0, y=0.0, name="O2")
    A = Crank(x=1.0, y=0.0, joint0=O1, distance=1.0, angle=0.1, name="A")
    B = Revolute(
        x=2.5,
        y=1.0,
        joint0=A,
        joint1=O2,
        distance0=2.0,
        distance1=1.5,
        name="B",
    )
    from pylinkage.linkage import Linkage

    return Linkage(name="FourBar", joints=(O1, O2, A, B))


class TestMechanismFromLinkage:
    """Tests for mechanism_from_linkage()."""

    def test_converts_static_to_ground_joint(self):
        linkage = _build_legacy_four_bar()
        mech = mechanism_from_linkage(linkage)

        ground_joints = [j for j in mech.joints if isinstance(j, GroundJoint)]
        assert len(ground_joints) == 2
        ids = {j.id for j in ground_joints}
        assert "O1" in ids
        assert "O2" in ids

    def test_converts_crank_to_driver_link(self):
        linkage = _build_legacy_four_bar()
        mech = mechanism_from_linkage(linkage)

        drivers = [lnk for lnk in mech.links if isinstance(lnk, DriverLink)]
        assert len(drivers) == 1
        driver = drivers[0]
        assert driver.motor_joint is not None
        assert driver.motor_joint.id == "O1"
        assert driver.output_joint is not None
        assert driver.output_joint.id == "A"

    def test_creates_ground_link(self):
        linkage = _build_legacy_four_bar()
        mech = mechanism_from_linkage(linkage)

        assert mech.ground is not None
        assert isinstance(mech.ground, GroundLink)
        assert len(mech.ground.joints) == 2

    def test_creates_links_for_revolute(self):
        linkage = _build_legacy_four_bar()
        mech = mechanism_from_linkage(linkage)

        # Revolute B depends on A and O2, so should produce link0 and link1
        link_ids = {lnk.id for lnk in mech.links}
        assert "B_link0" in link_ids
        assert "B_link1" in link_ids

    def test_preserves_name(self):
        linkage = _build_legacy_four_bar()
        mech = mechanism_from_linkage(linkage)
        assert mech.name == "FourBar"

    def test_joint_count(self):
        linkage = _build_legacy_four_bar()
        mech = mechanism_from_linkage(linkage)
        # O1, O2, A, B
        assert len(mech.joints) == 4

    def test_initial_angle_computation(self):
        """Driver initial angle should match atan2 from motor to output."""
        linkage = _build_legacy_four_bar()
        mech = mechanism_from_linkage(linkage)

        drivers = [lnk for lnk in mech.links if isinstance(lnk, DriverLink)]
        driver = drivers[0]
        motor = driver.motor_joint
        output = driver.output_joint
        expected = math.atan2(
            output.position[1] - motor.position[1],
            output.position[0] - motor.position[0],
        )
        assert abs(driver.initial_angle - expected) < 1e-10


class TestMechanismToLinkage:
    """Tests for mechanism_to_linkage()."""

    def test_roundtrip_preserves_name(self):
        linkage = _build_legacy_four_bar()
        mech = mechanism_from_linkage(linkage)
        restored = mechanism_to_linkage(mech)
        assert restored.name == "FourBar"

    def test_roundtrip_preserves_joint_count(self):
        linkage = _build_legacy_four_bar()
        mech = mechanism_from_linkage(linkage)
        restored = mechanism_to_linkage(mech)
        assert len(restored.joints) == len(linkage.joints)

    def test_roundtrip_creates_static_joints(self):
        from pylinkage.joints import Static

        linkage = _build_legacy_four_bar()
        mech = mechanism_from_linkage(linkage)
        restored = mechanism_to_linkage(mech)

        statics = [j for j in restored.joints if isinstance(j, Static)]
        assert len(statics) == 2

    def test_roundtrip_creates_crank_joint(self):
        from pylinkage.joints import Crank

        linkage = _build_legacy_four_bar()
        mech = mechanism_from_linkage(linkage)
        restored = mechanism_to_linkage(mech)

        cranks = [j for j in restored.joints if isinstance(j, Crank)]
        assert len(cranks) == 1
        assert cranks[0].name == "A"

    def test_simple_mechanism_to_linkage(self):
        """Convert a hand-built Mechanism to a Linkage."""
        O1 = GroundJoint("O1", position=(0.0, 0.0))
        O2 = GroundJoint("O2", position=(2.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))

        ground = GroundLink("ground", joints=[O1, O2])
        crank = DriverLink(
            "crank",
            joints=[O1, A],
            motor_joint=O1,
            angular_velocity=0.1,
        )

        mech = Mechanism(
            name="Simple",
            joints=[O1, O2, A],
            links=[ground, crank],
        )

        linkage = mechanism_to_linkage(mech)
        assert linkage.name == "Simple"
        assert len(linkage.joints) >= 2  # At least the ground + crank joints
