"""Joint classes for the mechanism module.

This module defines the fundamental joint types used in planar mechanisms:
- RevoluteJoint: Pin joint allowing rotation (1 DOF)
- PrismaticJoint: Slider joint allowing translation (1 DOF)
- GroundJoint: Revolute joint fixed to the frame

These are actual mechanical joints, unlike the 'joints' module which
defines Assur groups (combinations of joints and links).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .._types import Coord, JointType, MaybeCoord

# Re-export JointType for backward compatibility
__all__ = [
    "JointType",
    "Joint",
    "RevoluteJoint",
    "PrismaticJoint",
    "GroundJoint",
    "TrackerJoint",
    "AnyJoint",
]

if TYPE_CHECKING:
    from .link import Link


@dataclass(eq=False)
class Joint(ABC):
    """Base class for mechanical joints.

    A joint is a connection point between rigid links that allows
    relative motion. Each joint type permits specific degrees of freedom.

    Attributes:
        id: Unique identifier for this joint.
        position: Current (x, y) coordinates in the global frame.
        name: Human-readable name for display (defaults to id).

    Note:
        The `links` attribute is populated when the joint is added
        to a Mechanism, establishing the connectivity graph.
    """

    id: str
    position: MaybeCoord = (None, None)
    name: str | None = None

    # Links connected at this joint (populated by Mechanism)
    _links: list[Link] = field(default_factory=list, repr=False, compare=False)

    # Kinematic state — populated by Mechanism.step_with_derivatives()
    velocity: Coord | None = field(default=None, repr=False, compare=False)
    acceleration: Coord | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Set default name to id if not provided."""
        if self.name is None:
            self.name = self.id

    def __hash__(self) -> int:
        """Hash by id for use in sets and dict keys."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality by id."""
        if isinstance(other, Joint):
            return self.id == other.id
        return False

    @property
    @abstractmethod
    def joint_type(self) -> JointType:
        """Return the type of this joint."""
        ...

    @property
    def links(self) -> list[Link]:
        """Return the links connected at this joint."""
        return self._links

    @property
    def x(self) -> float | None:
        """Return the x coordinate."""
        return self.position[0]

    @x.setter
    def x(self, value: float | None) -> None:
        """Set the x coordinate."""
        self.position = (value, self.position[1])

    @property
    def y(self) -> float | None:
        """Return the y coordinate."""
        return self.position[1]

    @y.setter
    def y(self, value: float | None) -> None:
        """Set the y coordinate."""
        self.position = (self.position[0], value)

    def coord(self) -> MaybeCoord:
        """Return the current coordinates."""
        return self.position

    def set_coord(self, x: float | None, y: float | None) -> None:
        """Set the coordinates."""
        self.position = (x, y)

    def is_defined(self) -> bool:
        """Return True if position is fully defined (no None values)."""
        return self.position[0] is not None and self.position[1] is not None


@dataclass(eq=False)
class RevoluteJoint(Joint):
    """Pin joint allowing rotation between two links.

    A revolute joint permits one degree of freedom: rotation about
    the joint axis (perpendicular to the plane for planar mechanisms).

    This is the most common joint type in planar linkages.

    Example:
        >>> joint = RevoluteJoint("A", position=(1.0, 2.0))
        >>> joint.joint_type
        <JointType.REVOLUTE: 1>
    """

    @property
    def joint_type(self) -> JointType:
        """Return REVOLUTE type."""
        return JointType.REVOLUTE


@dataclass(eq=False)
class PrismaticJoint(Joint):
    """Slider joint allowing translation along an axis.

    A prismatic joint permits one degree of freedom: translation
    along the joint axis.

    Attributes:
        axis: Direction of allowed translation as (dx, dy).
              Should be normalized for consistency.
        line_point: A fixed point (x, y) on the slide line.
                    The joint is constrained to move along the line
                    passing through this point in the axis direction.
        slide_distance: Current displacement along the axis from origin.

    Example:
        >>> joint = PrismaticJoint("S", position=(0.0, 0.0), axis=(1.0, 0.0))
        >>> joint.joint_type
        <JointType.PRISMATIC: 2>
    """

    axis: Coord = (1.0, 0.0)  # Default: horizontal sliding
    line_point: Coord = (0.0, 0.0)  # Fixed point on the slide line
    slide_distance: float = 0.0

    @property
    def joint_type(self) -> JointType:
        """Return PRISMATIC type."""
        return JointType.PRISMATIC

    def get_axis_normalized(self) -> Coord:
        """Return the normalized axis direction."""
        import math

        dx, dy = self.axis
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-10:
            return (1.0, 0.0)  # Default to horizontal if degenerate
        return (dx / length, dy / length)


@dataclass(eq=False)
class GroundJoint(RevoluteJoint):
    """Revolute joint fixed to the frame (ground link).

    A ground joint is a revolute joint whose position is fixed
    in the global coordinate frame. It connects a moving link
    to the stationary frame.

    Ground joints are typically:
    - The base of a crank (motor attachment point)
    - The pivot point of a rocker
    - Any fixed pivot in the mechanism

    Example:
        >>> ground = GroundJoint("O", position=(0.0, 0.0))
        >>> ground.is_ground
        True
    """

    is_ground: bool = field(default=True, repr=False)

    @property
    def joint_type(self) -> JointType:
        """Return GROUND type."""
        return JointType.GROUND


@dataclass(eq=False)
class TrackerJoint(Joint):
    """Observer joint that tracks a position relative to two reference joints.

    A tracker joint is a sensor/observer that computes its position as a
    point at a fixed distance and angle from a reference joint, with the
    angle measured relative to the line connecting the two reference joints.

    This is useful for:
    - Tracking coupler points on a link (e.g., Chebyshev straight-line mechanism)
    - Observing positions without affecting the kinematic chain
    - Adding tracer points for visualization

    Attributes:
        ref_joint1_id: ID of the first reference joint (origin for polar coords).
        ref_joint2_id: ID of the second reference joint (defines reference direction).
        distance: Distance from ref_joint1 to this tracker.
        angle: Angle offset from ref_joint1->ref_joint2 direction (radians).

    Example:
        >>> # Track the midpoint of a link between joints A and B
        >>> tracker = TrackerJoint("midpoint", ref_joint1_id="A", ref_joint2_id="B",
        ...                        distance=0.5, angle=0.0)
    """

    ref_joint1_id: str = ""
    ref_joint2_id: str = ""
    distance: float = 0.0  # Distance from ref_joint1
    angle: float = 0.0  # Angle relative to ref_joint1->ref_joint2 direction

    @property
    def joint_type(self) -> JointType:
        """Return TRACKER type."""
        return JointType.TRACKER

    def update_position(self, ref1_pos: tuple[float, float], ref2_pos: tuple[float, float]) -> None:
        """Compute position from reference joint positions.

        Args:
            ref1_pos: Position of first reference joint (x, y).
            ref2_pos: Position of second reference joint (x, y).
        """
        import math

        x1, y1 = ref1_pos
        x2, y2 = ref2_pos

        # Compute angle from ref1 to ref2
        base_angle = math.atan2(y2 - y1, x2 - x1)

        # Add the offset angle
        total_angle = base_angle + self.angle

        # Compute position
        new_x = x1 + self.distance * math.cos(total_angle)
        new_y = y1 + self.distance * math.sin(total_angle)
        self.set_coord(new_x, new_y)


# Type alias for any joint
AnyJoint = RevoluteJoint | PrismaticJoint | GroundJoint | TrackerJoint
