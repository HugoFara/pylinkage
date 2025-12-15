"""Tests for the mechanism module.

These tests verify the new Links + Joints model works correctly.
"""

import math

import pytest

from pylinkage.mechanism import (
    DriverLink,
    GroundJoint,
    GroundLink,
    Joint,
    Link,
    Mechanism,
    PrismaticJoint,
    RevoluteJoint,
    create_crank,
    create_fixed_dyad,
    create_rrp_dyad,
    create_rrr_dyad,
    mechanism_from_dict,
    mechanism_to_dict,
)


class TestJoints:
    """Tests for Joint classes."""

    def test_revolute_joint_creation(self):
        """Test creating a revolute joint."""
        joint = RevoluteJoint(id="A", position=(1.0, 2.0))
        assert joint.id == "A"
        assert joint.position == (1.0, 2.0)
        assert joint.x == 1.0
        assert joint.y == 2.0
        assert joint.name == "A"

    def test_ground_joint_creation(self):
        """Test creating a ground joint."""
        joint = GroundJoint(id="O", position=(0.0, 0.0))
        assert joint.id == "O"
        assert joint.is_ground is True

    def test_prismatic_joint_creation(self):
        """Test creating a prismatic joint."""
        joint = PrismaticJoint(id="S", position=(1.0, 0.0), axis=(1.0, 0.0))
        assert joint.id == "S"
        assert joint.axis == (1.0, 0.0)

    def test_joint_equality(self):
        """Test that joints are equal by ID."""
        j1 = RevoluteJoint(id="A", position=(1.0, 2.0))
        j2 = RevoluteJoint(id="A", position=(3.0, 4.0))
        j3 = RevoluteJoint(id="B", position=(1.0, 2.0))
        assert j1 == j2  # Same ID
        assert j1 != j3  # Different ID

    def test_joint_is_defined(self):
        """Test checking if joint position is defined."""
        j1 = RevoluteJoint(id="A", position=(1.0, 2.0))
        j2 = RevoluteJoint(id="B", position=(None, None))
        assert j1.is_defined() is True
        assert j2.is_defined() is False


class TestLinks:
    """Tests for Link classes."""

    def test_link_creation(self):
        """Test creating a basic link."""
        j1 = RevoluteJoint(id="A", position=(0.0, 0.0))
        j2 = RevoluteJoint(id="B", position=(1.0, 0.0))
        link = Link(id="AB", joints=[j1, j2])
        assert link.id == "AB"
        assert len(link.joints) == 2
        assert link.length == 1.0

    def test_ground_link_creation(self):
        """Test creating a ground link."""
        O1 = GroundJoint(id="O1", position=(0.0, 0.0))
        O2 = GroundJoint(id="O2", position=(2.0, 0.0))
        ground = GroundLink(id="ground", joints=[O1, O2])
        assert ground.is_ground is True
        assert ground.length == 2.0

    def test_driver_link_creation(self):
        """Test creating a driver link."""
        O = GroundJoint(id="O", position=(0.0, 0.0))
        A = RevoluteJoint(id="A", position=(1.0, 0.0))
        driver = DriverLink(
            id="crank",
            joints=[O, A],
            motor_joint=O,
            angular_velocity=0.1,
        )
        assert driver.motor_joint == O
        assert driver.angular_velocity == 0.1
        assert driver.radius == 1.0
        assert driver.output_joint == A

    def test_driver_link_step(self):
        """Test that driver link rotates correctly."""
        O = GroundJoint(id="O", position=(0.0, 0.0))
        A = RevoluteJoint(id="A", position=(1.0, 0.0))
        driver = DriverLink(
            id="crank",
            joints=[O, A],
            motor_joint=O,
            angular_velocity=math.pi / 2,  # 90 degrees per step
            initial_angle=0.0,
        )
        driver.step(dt=1.0)

        # Should now be at 90 degrees
        assert abs(A.x - 0.0) < 1e-10
        assert abs(A.y - 1.0) < 1e-10


class TestDyads:
    """Tests for dyad factory functions."""

    def test_create_crank(self):
        """Test creating a crank."""
        O = GroundJoint(id="O", position=(0.0, 0.0))
        driver, output = create_crank(
            O, radius=2.0, angular_velocity=0.1, initial_angle=0.0
        )

        assert isinstance(driver, DriverLink)
        assert isinstance(output, RevoluteJoint)
        assert driver.radius == 2.0
        assert output.position == (2.0, 0.0)

    def test_create_rrr_dyad(self):
        """Test creating an RRR dyad."""
        A = RevoluteJoint(id="A", position=(0.0, 0.0))
        B = RevoluteJoint(id="B", position=(4.0, 0.0))

        link1, link2, C = create_rrr_dyad(
            A, B, distance1=3.0, distance2=3.0, name="dyad"
        )

        assert isinstance(C, RevoluteJoint)
        # C should be at intersection of circles
        # Two circles: center (0,0) r=3, center (4,0) r=3
        # Intersection at (2, sqrt(5)) or (2, -sqrt(5))
        assert abs(C.x - 2.0) < 1e-10
        assert abs(abs(C.y) - math.sqrt(5)) < 1e-10

    def test_create_rrr_dyad_unbuildable(self):
        """Test that unbuildable RRR dyad raises error."""
        A = RevoluteJoint(id="A", position=(0.0, 0.0))
        B = RevoluteJoint(id="B", position=(10.0, 0.0))

        # Circles too far apart to intersect
        with pytest.raises(ValueError, match="unbuildable"):
            create_rrr_dyad(A, B, distance1=1.0, distance2=1.0)

    def test_create_fixed_dyad(self):
        """Test creating a fixed angular dyad."""
        A = RevoluteJoint(id="A", position=(0.0, 0.0))
        B = RevoluteJoint(id="B", position=(1.0, 0.0))

        link1, link2, C = create_fixed_dyad(
            A, B, distance=1.0, angle=math.pi / 2, name="fixed"
        )

        # C should be perpendicular to AB at distance 1 from A
        assert abs(C.x - 0.0) < 1e-10
        assert abs(C.y - 1.0) < 1e-10


class TestMechanism:
    """Tests for the Mechanism class."""

    def test_simple_mechanism_creation(self):
        """Test creating a simple mechanism."""
        O1 = GroundJoint(id="O1", position=(0.0, 0.0))
        O2 = GroundJoint(id="O2", position=(2.0, 0.0))
        ground = GroundLink(id="ground", joints=[O1, O2])

        mechanism = Mechanism(
            name="Test",
            joints=[O1, O2],
            links=[ground],
            ground=ground,
        )

        assert mechanism.name == "Test"
        assert len(mechanism.joints) == 2
        assert len(mechanism.links) == 1
        assert mechanism.ground == ground

    def test_four_bar_creation(self):
        """Test creating a four-bar linkage."""
        # Ground joints
        O1 = GroundJoint(id="O1", position=(0.0, 0.0))
        O2 = GroundJoint(id="O2", position=(2.0, 0.0))
        ground = GroundLink(id="ground", joints=[O1, O2])

        # Create crank
        crank, A = create_crank(O1, radius=1.0, angular_velocity=0.1)

        # Create coupler via RRR dyad
        link1, link2, B = create_rrr_dyad(A, O2, distance1=2.0, distance2=1.5)

        mechanism = Mechanism(
            name="Four-Bar",
            joints=[O1, O2, A, B],
            links=[ground, crank, link1, link2],
        )

        assert len(mechanism.joints) == 4
        assert len(mechanism.links) == 4

    def test_get_joint_positions(self):
        """Test getting joint positions."""
        O = GroundJoint(id="O", position=(0.0, 0.0))
        A = RevoluteJoint(id="A", position=(1.0, 2.0))

        mechanism = Mechanism(joints=[O, A], links=[])

        positions = mechanism.get_joint_positions()
        assert positions == [(0.0, 0.0), (1.0, 2.0)]


class TestSerialization:
    """Tests for mechanism serialization."""

    def test_mechanism_to_dict(self):
        """Test serializing a mechanism."""
        O = GroundJoint(id="O", position=(0.0, 0.0))
        A = RevoluteJoint(id="A", position=(1.0, 0.0))
        ground = GroundLink(id="ground", joints=[O])
        link = Link(id="OA", joints=[O, A])

        mechanism = Mechanism(
            name="Test",
            joints=[O, A],
            links=[ground, link],
            ground=ground,
        )

        data = mechanism_to_dict(mechanism)

        assert data["name"] == "Test"
        assert len(data["joints"]) == 2
        assert len(data["links"]) == 2

    def test_mechanism_from_dict(self):
        """Test deserializing a mechanism."""
        data = {
            "name": "Test",
            "joints": [
                {"id": "O", "type": "ground", "position": [0.0, 0.0]},
                {"id": "A", "type": "revolute", "position": [1.0, 0.0]},
            ],
            "links": [
                {"id": "ground", "type": "ground", "joints": ["O"]},
                {"id": "OA", "type": "link", "joints": ["O", "A"]},
            ],
            "ground": "ground",
        }

        mechanism = mechanism_from_dict(data)

        assert mechanism.name == "Test"
        assert len(mechanism.joints) == 2
        assert isinstance(mechanism.joints[0], GroundJoint)
        assert isinstance(mechanism.joints[1], RevoluteJoint)

    def test_roundtrip_serialization(self):
        """Test that serialization round-trips correctly."""
        O = GroundJoint(id="O", position=(0.0, 0.0))
        A = RevoluteJoint(id="A", position=(1.0, 0.0))
        ground = GroundLink(id="ground", joints=[O])
        link = Link(id="OA", joints=[O, A])

        original = Mechanism(
            name="Test",
            joints=[O, A],
            links=[ground, link],
            ground=ground,
        )

        data = mechanism_to_dict(original)
        restored = mechanism_from_dict(data)

        assert restored.name == original.name
        assert len(restored.joints) == len(original.joints)
        assert len(restored.links) == len(original.links)
