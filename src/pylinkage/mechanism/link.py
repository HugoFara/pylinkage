"""Link classes for the mechanism module.

This module defines rigid body types used in planar mechanisms:
- Link: A rigid body connecting two or more joints
- GroundLink: The stationary frame (ground) of the mechanism
- DriverLink: An input link driven by a motor

Links are the structural members of a mechanism that transmit
forces and motion between joints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .joint import GroundJoint, Joint


class LinkType(IntEnum):
    """Enumeration of link types."""

    BINARY = 2  # Connects exactly 2 joints
    TERNARY = 3  # Connects exactly 3 joints
    QUATERNARY = 4  # Connects exactly 4 joints
    GROUND = 0  # Special: the stationary frame
    DRIVER = 1  # Special: motor-driven input link


@dataclass(eq=False)
class Link:
    """A rigid body connecting two or more joints.

    A link is a rigid member that maintains fixed distances between
    all joints attached to it. Most common are binary links (2 joints),
    but ternary (3 joints) and higher-order links also exist.

    Attributes:
        id: Unique identifier for this link.
        joints: List of joints connected by this link.
        name: Human-readable name for display (defaults to id).

    Example:
        >>> from pylinkage.mechanism.joint import RevoluteJoint
        >>> j1 = RevoluteJoint("A", position=(0.0, 0.0))
        >>> j2 = RevoluteJoint("B", position=(1.0, 0.0))
        >>> link = Link("AB", joints=[j1, j2])
        >>> link.length
        1.0
    """

    id: str
    joints: list[Joint] = field(default_factory=list)
    name: str | None = None

    def __post_init__(self) -> None:
        """Set default name to id if not provided."""
        if self.name is None:
            self.name = self.id

    def __hash__(self) -> int:
        """Hash by id for use in sets and dict keys."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality by id."""
        if isinstance(other, Link):
            return self.id == other.id
        return False

    @property
    def link_type(self) -> LinkType:
        """Return the type based on number of joints."""
        n = len(self.joints)
        if n == 2:
            return LinkType.BINARY
        if n == 3:
            return LinkType.TERNARY
        if n == 4:
            return LinkType.QUATERNARY
        return LinkType.BINARY  # Default

    @property
    def order(self) -> int:
        """Return the order (number of joints) of this link."""
        return len(self.joints)

    @property
    def length(self) -> float | None:
        """Return the length of a binary link.

        Only meaningful for binary links (2 joints). Returns the
        Euclidean distance between the two joint positions.

        Returns:
            Distance between joints, or None if not a binary link
            or if positions are undefined.
        """
        if len(self.joints) != 2:
            return None

        j1, j2 = self.joints
        if not j1.is_defined() or not j2.is_defined():
            return None

        x1, y1 = j1.position
        x2, y2 = j2.position
        # Type narrowing: is_defined() guarantees non-None
        assert x1 is not None and y1 is not None
        assert x2 is not None and y2 is not None

        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def get_distance(self, joint1: Joint, joint2: Joint) -> float | None:
        """Get the distance constraint between two joints on this link.

        Args:
            joint1: First joint (must be in this link).
            joint2: Second joint (must be in this link).

        Returns:
            Distance between joints, or None if undefined.

        Raises:
            ValueError: If either joint is not part of this link.
        """
        if joint1 not in self.joints:
            raise ValueError(f"Joint {joint1.id} is not part of link {self.id}")
        if joint2 not in self.joints:
            raise ValueError(f"Joint {joint2.id} is not part of link {self.id}")

        if not joint1.is_defined() or not joint2.is_defined():
            return None

        x1, y1 = joint1.position
        x2, y2 = joint2.position
        assert x1 is not None and y1 is not None
        assert x2 is not None and y2 is not None

        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def other_joint(self, joint: Joint) -> Joint | None:
        """Get the other joint in a binary link.

        Args:
            joint: One of the joints in this binary link.

        Returns:
            The other joint, or None if not a binary link
            or if joint is not in this link.
        """
        if len(self.joints) != 2:
            return None
        if joint not in self.joints:
            return None

        return self.joints[1] if self.joints[0] == joint else self.joints[0]


@dataclass(eq=False)
class GroundLink(Link):
    """The stationary frame (ground) of a mechanism.

    Every mechanism has exactly one ground link, which represents
    the fixed reference frame. All ground joints are attached to
    this link.

    The ground link is special because:
    - Its joints never move during simulation
    - It establishes the global coordinate system
    - All motion is measured relative to it

    Example:
        >>> from pylinkage.mechanism.joint import GroundJoint
        >>> O1 = GroundJoint("O1", position=(0.0, 0.0))
        >>> O2 = GroundJoint("O2", position=(2.0, 0.0))
        >>> ground = GroundLink("ground", joints=[O1, O2])
    """

    is_ground: bool = field(default=True, repr=False)

    @property
    def link_type(self) -> LinkType:
        """Return GROUND type."""
        return LinkType.GROUND


@dataclass(eq=False)
class DriverLink(Link):
    """An input link driven by a motor.

    A driver link is connected to the ground at one joint (the motor
    joint) and rotates at a specified angular velocity. It is the
    source of motion in the mechanism.

    Attributes:
        motor_joint: The ground joint where the motor is attached.
        angular_velocity: Rotation rate in radians per simulation step.
        initial_angle: Starting angle in radians (from positive x-axis).
        current_angle: Current angle during simulation.

    Example:
        >>> from pylinkage.mechanism.joint import GroundJoint, RevoluteJoint
        >>> O = GroundJoint("O", position=(0.0, 0.0))
        >>> A = RevoluteJoint("A", position=(1.0, 0.0))
        >>> crank = DriverLink("crank", joints=[O, A], motor_joint=O,
        ...                    angular_velocity=0.1)
    """

    motor_joint: GroundJoint | None = None
    angular_velocity: float = 0.1  # radians per step
    initial_angle: float = 0.0  # radians
    current_angle: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        """Initialize current angle from initial angle."""
        super().__post_init__()
        self.current_angle = self.initial_angle

    @property
    def link_type(self) -> LinkType:
        """Return DRIVER type."""
        return LinkType.DRIVER

    @property
    def radius(self) -> float | None:
        """Return the crank radius (distance from motor to output joint).

        Only valid for binary driver links. Returns the distance from
        the motor joint to the other joint.
        """
        if len(self.joints) != 2 or self.motor_joint is None:
            return None

        output_joint = self.other_joint(self.motor_joint)
        if output_joint is None:
            return None

        return self.get_distance(self.motor_joint, output_joint)

    @property
    def output_joint(self) -> Joint | None:
        """Return the non-motor joint (output of the crank)."""
        if self.motor_joint is None:
            return None
        return self.other_joint(self.motor_joint)

    def step(self, dt: float = 1.0) -> None:
        """Advance the crank by one time step.

        Updates current_angle and repositions the output joint
        based on the angular velocity.

        Args:
            dt: Time step multiplier (default 1.0 = full step).
        """
        self.current_angle += self.angular_velocity * dt

        # Update output joint position
        output = self.output_joint
        if output is None or self.motor_joint is None:
            return

        radius = self.radius
        if radius is None:
            return

        mx, my = self.motor_joint.position
        if mx is None or my is None:
            return

        # Compute new position from angle
        new_x = mx + radius * math.cos(self.current_angle)
        new_y = my + radius * math.sin(self.current_angle)
        output.set_coord(new_x, new_y)

    def reset(self) -> None:
        """Reset the crank to its initial angle."""
        self.current_angle = self.initial_angle


# Type alias for any link
AnyLink = Link | GroundLink | DriverLink
