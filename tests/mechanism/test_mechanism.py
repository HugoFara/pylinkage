"""Tests for the mechanism module.

These tests verify the low-level Links + Joints model works correctly.

Note: For dyad factory function tests, see tests/dyads/test_dyads.py.
"""

import math

from pylinkage.mechanism import (
    DriverLink,
    GroundJoint,
    GroundLink,
    Link,
    Mechanism,
    PrismaticJoint,
    RevoluteJoint,
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
        """Test creating a four-bar linkage with low-level API."""
        # Ground joints
        O1 = GroundJoint(id="O1", position=(0.0, 0.0))
        O2 = GroundJoint(id="O2", position=(2.0, 0.0))
        ground = GroundLink(id="ground", joints=[O1, O2])

        # Create crank manually
        A = RevoluteJoint(id="A", position=(1.0, 0.0))
        crank = DriverLink(
            id="crank",
            joints=[O1, A],
            motor_joint=O1,
            angular_velocity=0.1,
        )

        # Create coupler joint and links manually
        B = RevoluteJoint(id="B", position=(2.5, 1.0))  # Approximate position
        link1 = Link(id="link1", joints=[A, B])
        link2 = Link(id="link2", joints=[O2, B])

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
