"""Conversion utilities between Mechanism and legacy Linkage formats.

This module provides bidirectional conversion between:
- The new Mechanism (Links + Joints) model
- The legacy Linkage (Assur group "joints") model

This enables backward compatibility with existing code and files.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from .joint import GroundJoint, Joint, PrismaticJoint, RevoluteJoint
from .link import DriverLink, GroundLink, Link
from .mechanism import Mechanism

if TYPE_CHECKING:
    from ..linkage import Linkage


def mechanism_from_linkage(linkage: Linkage) -> Mechanism:
    """Convert a legacy Linkage to a Mechanism.

    Maps the old Assur-group-based representation to the new
    Links + Joints model.

    Args:
        linkage: The legacy Linkage object.

    Returns:
        A new Mechanism with equivalent structure.

    Mapping:
        - Static -> GroundJoint
        - Crank -> DriverLink + RevoluteJoint (output)
        - Revolute -> RevoluteJoint + 2 Links (RRR dyad)
        - Fixed -> RevoluteJoint + 2 Links (fixed dyad)
        - Prismatic -> PrismaticJoint + Links (RRP dyad)
    """
    from ..joints import Crank, Fixed, Prismatic, Revolute, Static

    joints: list[Joint] = []
    links: list[Link] = []
    joint_map: dict[str, Joint] = {}
    ground_joints: list[GroundJoint] = []

    # First pass: create joints from legacy joints
    for old_joint in linkage.joints:
        new_joint: Joint

        if isinstance(old_joint, Static):
            # Static -> GroundJoint
            ground = GroundJoint(
                id=old_joint.name,
                position=old_joint.coord(),
                name=old_joint.name,
            )
            joints.append(ground)
            joint_map[old_joint.name] = ground
            ground_joints.append(ground)

        elif isinstance(old_joint, Crank):
            # Crank -> RevoluteJoint (the crank output)
            new_joint = RevoluteJoint(
                id=old_joint.name,
                position=old_joint.coord(),
                name=old_joint.name,
            )
            joints.append(new_joint)
            joint_map[old_joint.name] = new_joint

        elif isinstance(old_joint, Revolute):
            # Revolute -> RevoluteJoint
            new_joint = RevoluteJoint(
                id=old_joint.name,
                position=old_joint.coord(),
                name=old_joint.name,
            )
            joints.append(new_joint)
            joint_map[old_joint.name] = new_joint

        elif isinstance(old_joint, Fixed):
            # Fixed -> RevoluteJoint
            new_joint = RevoluteJoint(
                id=old_joint.name,
                position=old_joint.coord(),
                name=old_joint.name,
            )
            joints.append(new_joint)
            joint_map[old_joint.name] = new_joint

        elif isinstance(old_joint, Prismatic):
            # Prismatic -> PrismaticJoint
            new_joint = PrismaticJoint(
                id=old_joint.name,
                position=old_joint.coord(),
                name=old_joint.name,
            )
            joints.append(new_joint)
            joint_map[old_joint.name] = new_joint

    # Create ground link from ground joints
    ground_link = None
    if ground_joints:
        ground_link = GroundLink(
            id="ground",
            joints=list(ground_joints),
            name="ground",
        )
        links.append(ground_link)

    # Second pass: create links from joint relationships
    for old_joint in linkage.joints:
        if isinstance(old_joint, Crank):
            # Crank creates a DriverLink
            output_joint = joint_map.get(old_joint.name)
            motor_joint = None

            # Find the motor joint (joint0)
            if old_joint.joint0 is not None:
                parent_name = old_joint.joint0.name
                parent = joint_map.get(parent_name)
                if isinstance(parent, GroundJoint):
                    motor_joint = parent

            if output_joint and motor_joint:
                driver = DriverLink(
                    id=f"{old_joint.name}_crank",
                    joints=[motor_joint, output_joint],
                    name=f"{old_joint.name}_crank",
                    motor_joint=motor_joint,
                    angular_velocity=old_joint.angle or 0.1,
                    initial_angle=_compute_initial_angle(motor_joint, output_joint),
                )
                links.append(driver)

        elif isinstance(old_joint, (Revolute, Fixed)):
            # Create links to parent joints
            current_joint = joint_map.get(old_joint.name)
            if current_joint is None:
                continue

            if old_joint.joint0 is not None:
                parent0 = joint_map.get(old_joint.joint0.name)
                if parent0 and parent0 != current_joint:
                    link = Link(
                        id=f"{old_joint.name}_link0",
                        joints=[parent0, current_joint],
                        name=f"{old_joint.name}_link0",
                    )
                    links.append(link)

            if old_joint.joint1 is not None:
                parent1 = joint_map.get(old_joint.joint1.name)
                if parent1 and parent1 != current_joint:
                    link = Link(
                        id=f"{old_joint.name}_link1",
                        joints=[parent1, current_joint],
                        name=f"{old_joint.name}_link1",
                    )
                    links.append(link)

        elif isinstance(old_joint, Prismatic):
            # Create links for prismatic joint
            current_joint = joint_map.get(old_joint.name)
            if current_joint is None:
                continue

            if old_joint.joint0 is not None:
                parent0 = joint_map.get(old_joint.joint0.name)
                if parent0:
                    link = Link(
                        id=f"{old_joint.name}_link0",
                        joints=[parent0, current_joint],
                        name=f"{old_joint.name}_link0",
                    )
                    links.append(link)

    return Mechanism(
        name=linkage.name,
        joints=joints,
        links=links,
        ground=ground_link,
    )


def _compute_initial_angle(
    motor: GroundJoint,
    output: Joint,
) -> float:
    """Compute the initial angle from motor to output joint."""
    mx, my = motor.position
    ox, oy = output.position

    if mx is None or my is None or ox is None or oy is None:
        return 0.0

    return math.atan2(oy - my, ox - mx)


def mechanism_to_linkage(mechanism: Mechanism) -> Linkage:
    """Convert a Mechanism back to a legacy Linkage.

    This is useful for using the new mechanism API with
    existing code that expects Linkage objects.

    Args:
        mechanism: The Mechanism to convert.

    Returns:
        A legacy Linkage with equivalent behavior.
    """
    from ..joints import Crank, Revolute, Static
    from ..joints.joint import Joint as OldJoint
    from ..linkage import Linkage

    old_joints: list[OldJoint] = []
    joint_map: dict[str, OldJoint] = {}

    # Create Static joints for ground joints
    for joint in mechanism.joints:
        if isinstance(joint, GroundJoint):
            x, y = joint.position
            static_joint = Static(
                x=x or 0.0,
                y=y or 0.0,
                name=joint.id,
            )
            old_joints.append(static_joint)
            joint_map[joint.id] = static_joint

    # Create Crank joints for driver outputs
    for link in mechanism.links:
        if isinstance(link, DriverLink):
            output = link.output_joint
            motor = link.motor_joint

            if output and motor:
                motor_old = joint_map.get(motor.id)
                if motor_old:
                    x, y = output.position
                    crank_joint = Crank(
                        x=x,
                        y=y,
                        joint0=motor_old,
                        distance=link.radius,
                        angle=link.angular_velocity,
                        name=output.id,
                    )
                    old_joints.append(crank_joint)
                    joint_map[output.id] = crank_joint

    # Create Revolute joints for remaining joints
    for joint in mechanism.joints:
        if joint.id in joint_map:
            continue  # Already created

        if isinstance(joint, (RevoluteJoint, PrismaticJoint)):
            # Find parent joints through links
            parents: list[OldJoint] = []
            distances: list[float] = []

            for link in joint._links:
                for other in link.joints:
                    if other.id != joint.id and other.id in joint_map:
                        parents.append(joint_map[other.id])
                        dist = link.get_distance(joint, other)
                        distances.append(dist or 1.0)

            x, y = joint.position
            if len(parents) >= 2:
                revolute_joint = Revolute(
                    x=x or 0.0,
                    y=y or 0.0,
                    joint0=parents[0],
                    joint1=parents[1],
                    distance0=distances[0],
                    distance1=distances[1],
                    name=joint.id,
                )
                old_joints.append(revolute_joint)
                joint_map[joint.id] = revolute_joint

    return Linkage(
        name=mechanism.name,
        joints=old_joints,
    )


def convert_legacy_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy linkage dict to mechanism format.

    Args:
        data: Dictionary in legacy format.

    Returns:
        Dictionary in new mechanism format.
    """
    from ..linkage.serialization import linkage_from_dict

    # Parse as legacy linkage
    linkage = linkage_from_dict(data)

    # Convert to mechanism
    mechanism = mechanism_from_linkage(linkage)

    # Serialize as mechanism
    from .serialization import mechanism_to_dict

    return mechanism_to_dict(mechanism)
