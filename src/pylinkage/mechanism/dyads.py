"""Factory functions for creating common Assur groups (dyads).

This module provides convenient factory functions for creating
standard kinematic substructures used in planar mechanisms.

In mechanism design, dyads are minimal Assur groups - kinematic
substructures with DOF=0 when attached to the frame. They are
the building blocks for constructing linkages.

Factory Functions:
    create_crank: Create a driver link with output joint
    create_rrr_dyad: Create two links meeting at a computed revolute joint
    create_rrp_dyad: Create a slider mechanism (circle-line intersection)
    create_fixed_dyad: Create a fixed angular relationship
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..solver.joints import solve_linear, solve_revolute
from .joint import GroundJoint, PrismaticJoint, RevoluteJoint
from .link import DriverLink, Link

if TYPE_CHECKING:
    from .joint import Joint


def create_crank(
    ground_joint: GroundJoint,
    radius: float,
    angular_velocity: float = 0.1,
    initial_angle: float = 0.0,
    name: str = "crank",
    output_name: str | None = None,
) -> tuple[DriverLink, RevoluteJoint]:
    """Create a driver crank (motor link + output joint).

    A crank is a driver link that rotates around a ground joint
    at a constant angular velocity. This is the primary input
    for most linkage mechanisms.

    Args:
        ground_joint: The fixed pivot point (motor attachment).
        radius: Distance from ground_joint to output joint.
        angular_velocity: Rotation rate in radians per step.
        initial_angle: Starting angle in radians (from +x axis).
        name: Name for the driver link.
        output_name: Name for the output joint (defaults to "{name}_output").

    Returns:
        A tuple of (DriverLink, RevoluteJoint) where:
        - DriverLink: The rotating input link
        - RevoluteJoint: The joint at the end of the crank

    Example:
        >>> O = GroundJoint("O", position=(0.0, 0.0))
        >>> crank, A = create_crank(O, radius=1.0, angular_velocity=0.1)
        >>> A.position
        (1.0, 0.0)
    """
    if output_name is None:
        output_name = f"{name}_output"

    # Compute initial position of output joint
    gx, gy = ground_joint.position
    if gx is None or gy is None:
        raise ValueError(f"Ground joint {ground_joint.id} must have defined position")

    output_x = gx + radius * math.cos(initial_angle)
    output_y = gy + radius * math.sin(initial_angle)

    output_joint = RevoluteJoint(
        id=output_name,
        position=(output_x, output_y),
        name=output_name,
    )

    driver = DriverLink(
        id=name,
        joints=[ground_joint, output_joint],
        name=name,
        motor_joint=ground_joint,
        angular_velocity=angular_velocity,
        initial_angle=initial_angle,
    )

    return driver, output_joint


def create_rrr_dyad(
    anchor1: Joint,
    anchor2: Joint,
    distance1: float,
    distance2: float,
    name: str = "dyad",
    joint_name: str | None = None,
    initial_position: tuple[float, float] | None = None,
) -> tuple[Link, Link, RevoluteJoint]:
    """Create an RRR dyad (two links + one computed joint).

    An RRR dyad consists of:
    - Two rigid links connecting to existing anchor joints
    - One new revolute joint at their intersection

    The new joint position is computed as the intersection of
    two circles centered at the anchor joints.

    Args:
        anchor1: First anchor joint (already positioned).
        anchor2: Second anchor joint (already positioned).
        distance1: Length of link from anchor1 to new joint.
        distance2: Length of link from anchor2 to new joint.
        name: Base name for the dyad components.
        joint_name: Name for the new joint (defaults to "{name}_joint").
        initial_position: Hint for which intersection to choose.
                         If None, uses the anchor1 position as hint.

    Returns:
        A tuple of (Link, Link, RevoluteJoint) where:
        - Link 1: Connects anchor1 to the new joint
        - Link 2: Connects anchor2 to the new joint
        - RevoluteJoint: The computed intersection joint

    Raises:
        ValueError: If anchor positions are undefined or circles don't intersect.

    Example:
        >>> A = RevoluteJoint("A", position=(0.0, 1.0))
        >>> O2 = GroundJoint("O2", position=(2.0, 0.0))
        >>> link1, link2, B = create_rrr_dyad(A, O2, distance1=2.0, distance2=1.5)
    """
    if joint_name is None:
        joint_name = f"{name}_joint"

    # Validate anchor positions
    if not anchor1.is_defined():
        raise ValueError(f"Anchor {anchor1.id} must have defined position")
    if not anchor2.is_defined():
        raise ValueError(f"Anchor {anchor2.id} must have defined position")

    a1x, a1y = anchor1.position
    a2x, a2y = anchor2.position
    assert a1x is not None and a1y is not None
    assert a2x is not None and a2y is not None

    # Determine initial guess for circle-circle intersection
    if initial_position is not None:
        curr_x, curr_y = initial_position
    else:
        # Use midpoint as initial guess
        curr_x = (a1x + a2x) / 2
        curr_y = (a1y + a2y) / 2 + 0.1  # Slight offset to avoid degenerate cases

    # Solve for intersection using existing solver
    new_x, new_y = solve_revolute(
        curr_x, curr_y,
        a1x, a1y, distance1,
        a2x, a2y, distance2,
    )

    if math.isnan(new_x):
        raise ValueError(
            f"RRR dyad unbuildable: circles centered at {anchor1.id} "
            f"(r={distance1}) and {anchor2.id} (r={distance2}) don't intersect"
        )

    # Create the new joint
    new_joint = RevoluteJoint(
        id=joint_name,
        position=(new_x, new_y),
        name=joint_name,
    )

    # Create the two links
    link1 = Link(
        id=f"{name}_link1",
        joints=[anchor1, new_joint],
        name=f"{name}_link1",
    )
    link2 = Link(
        id=f"{name}_link2",
        joints=[anchor2, new_joint],
        name=f"{name}_link2",
    )

    return link1, link2, new_joint


def create_rrp_dyad(
    revolute_anchor: Joint,
    line_joint1: Joint,
    line_joint2: Joint,
    distance: float,
    name: str = "slider",
    joint_name: str | None = None,
    initial_position: tuple[float, float] | None = None,
) -> tuple[Link, Link, PrismaticJoint]:
    """Create an RRP dyad (circle-line intersection, slider mechanism).

    An RRP dyad consists of:
    - A revolute connection to an anchor joint
    - A prismatic (sliding) connection along a line
    - The new joint slides along the line while maintaining
      fixed distance from the revolute anchor

    Args:
        revolute_anchor: Joint connected by revolute pair.
        line_joint1: First joint defining the sliding line.
        line_joint2: Second joint defining the sliding line.
        distance: Distance from revolute_anchor to new joint.
        name: Base name for the dyad components.
        joint_name: Name for the new joint (defaults to "{name}_joint").
        initial_position: Hint for which intersection to choose.

    Returns:
        A tuple of (Link, Link, PrismaticJoint) where:
        - Link 1: Connects revolute_anchor to the new joint
        - Link 2: The sliding guide (connects line joints)
        - PrismaticJoint: The computed slider joint

    Raises:
        ValueError: If positions are undefined or no intersection exists.

    Example:
        >>> A = RevoluteJoint("A", position=(0.0, 1.0))
        >>> L1 = GroundJoint("L1", position=(0.0, 0.0))
        >>> L2 = GroundJoint("L2", position=(2.0, 0.0))
        >>> link1, link2, S = create_rrp_dyad(A, L1, L2, distance=1.5)
    """
    if joint_name is None:
        joint_name = f"{name}_joint"

    # Validate positions
    if not revolute_anchor.is_defined():
        raise ValueError(f"Revolute anchor {revolute_anchor.id} must have defined position")
    if not line_joint1.is_defined():
        raise ValueError(f"Line joint {line_joint1.id} must have defined position")
    if not line_joint2.is_defined():
        raise ValueError(f"Line joint {line_joint2.id} must have defined position")

    ax, ay = revolute_anchor.position
    l1x, l1y = line_joint1.position
    l2x, l2y = line_joint2.position
    assert ax is not None and ay is not None
    assert l1x is not None and l1y is not None
    assert l2x is not None and l2y is not None

    # Initial guess
    if initial_position is not None:
        curr_x, curr_y = initial_position
    else:
        curr_x, curr_y = ax, ay

    # Solve circle-line intersection
    new_x, new_y = solve_linear(
        curr_x, curr_y,
        ax, ay, distance,
        l1x, l1y,
        l2x, l2y,
    )

    if math.isnan(new_x):
        raise ValueError(
            f"RRP dyad unbuildable: circle at {revolute_anchor.id} (r={distance}) "
            f"doesn't intersect line through {line_joint1.id} and {line_joint2.id}"
        )

    # Compute sliding axis (along the line)
    axis = (l2x - l1x, l2y - l1y)

    # Create the prismatic joint
    new_joint = PrismaticJoint(
        id=joint_name,
        position=(new_x, new_y),
        name=joint_name,
        axis=axis,
    )

    # Create links
    link1 = Link(
        id=f"{name}_link1",
        joints=[revolute_anchor, new_joint],
        name=f"{name}_link1",
    )
    link2 = Link(
        id=f"{name}_guide",
        joints=[line_joint1, line_joint2, new_joint],
        name=f"{name}_guide",
    )

    return link1, link2, new_joint


def create_fixed_dyad(
    anchor1: Joint,
    anchor2: Joint,
    distance: float,
    angle: float,
    name: str = "fixed",
    joint_name: str | None = None,
) -> tuple[Link, Link, RevoluteJoint]:
    """Create a fixed angular dyad (deterministic position).

    A fixed dyad positions a joint at a fixed distance and angle
    relative to two anchor joints. Unlike RRR dyad, the position
    is deterministic (no ambiguity).

    The angle is measured from the anchor1→anchor2 direction.

    Args:
        anchor1: First anchor joint (center of rotation for angle).
        anchor2: Second anchor joint (defines reference direction).
        distance: Distance from anchor1 to new joint.
        angle: Angle in radians from anchor1→anchor2 direction.
        name: Base name for the dyad components.
        joint_name: Name for the new joint.

    Returns:
        A tuple of (Link, Link, RevoluteJoint) where:
        - Link 1: Connects anchor1 to the new joint
        - Link 2: Connects anchor2 to the new joint
        - RevoluteJoint: The deterministically positioned joint

    Example:
        >>> A = RevoluteJoint("A", position=(0.0, 0.0))
        >>> B = RevoluteJoint("B", position=(1.0, 0.0))
        >>> link1, link2, C = create_fixed_dyad(A, B, distance=1.0, angle=math.pi/2)
        >>> C.position  # (0.0, 1.0) - perpendicular to AB
    """
    if joint_name is None:
        joint_name = f"{name}_joint"

    if not anchor1.is_defined():
        raise ValueError(f"Anchor {anchor1.id} must have defined position")
    if not anchor2.is_defined():
        raise ValueError(f"Anchor {anchor2.id} must have defined position")

    a1x, a1y = anchor1.position
    a2x, a2y = anchor2.position
    assert a1x is not None and a1y is not None
    assert a2x is not None and a2y is not None

    # Compute reference angle (from anchor1 to anchor2)
    ref_angle = math.atan2(a2y - a1y, a2x - a1x)

    # Compute new joint position
    total_angle = ref_angle + angle
    new_x = a1x + distance * math.cos(total_angle)
    new_y = a1y + distance * math.sin(total_angle)

    # Create joint and links
    new_joint = RevoluteJoint(
        id=joint_name,
        position=(new_x, new_y),
        name=joint_name,
    )

    link1 = Link(
        id=f"{name}_link1",
        joints=[anchor1, new_joint],
        name=f"{name}_link1",
    )

    # For fixed dyad, link2 connects both anchors to the new joint
    # (ternary link representing the rigid triangle)
    link2 = Link(
        id=f"{name}_link2",
        joints=[anchor2, new_joint],
        name=f"{name}_link2",
    )

    return link1, link2, new_joint
