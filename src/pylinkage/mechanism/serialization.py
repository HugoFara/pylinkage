"""Serialization for the mechanism module.

This module provides functions to serialize and deserialize
Mechanism objects to/from JSON-compatible dictionaries.

The serialization format separates joints and links clearly:
- Joints have types, positions, and properties
- Links reference joints by ID and include constraints
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .joint import GroundJoint, Joint, PrismaticJoint, RevoluteJoint, TrackerJoint
from .link import ArcDriverLink, DriverLink, GroundLink, Link
from .mechanism import Mechanism


def joint_to_dict(joint: Joint) -> dict[str, Any]:
    """Serialize a joint to a dictionary.

    Args:
        joint: The joint to serialize.

    Returns:
        Dictionary representation of the joint.
    """
    data: dict[str, Any] = {
        "id": joint.id,
        "type": _joint_type_name(joint),
        "position": list(joint.position),
    }

    if joint.name and joint.name != joint.id:
        data["name"] = joint.name

    # Type-specific attributes
    if isinstance(joint, PrismaticJoint):
        data["axis"] = list(joint.axis)
        data["line_point"] = list(joint.line_point)
        data["slide_distance"] = joint.slide_distance
    elif isinstance(joint, TrackerJoint):
        data["ref_joint1_id"] = joint.ref_joint1_id
        data["ref_joint2_id"] = joint.ref_joint2_id
        data["distance"] = joint.distance
        data["angle"] = joint.angle

    return data


def _joint_type_name(joint: Joint) -> str:
    """Get the type name for a joint."""
    if isinstance(joint, GroundJoint):
        return "ground"
    if isinstance(joint, PrismaticJoint):
        return "prismatic"
    if isinstance(joint, TrackerJoint):
        return "tracker"
    if isinstance(joint, RevoluteJoint):
        return "revolute"
    return "revolute"  # Default


def joint_from_dict(data: dict[str, Any]) -> Joint:
    """Deserialize a joint from a dictionary.

    Args:
        data: Dictionary representation of a joint.

    Returns:
        The deserialized Joint object.
    """
    joint_type = data.get("type", "revolute")
    joint_id = data["id"]
    position = tuple(data.get("position", [None, None]))
    name = data.get("name")

    pos = (position[0], position[1])

    if joint_type == "ground":
        return GroundJoint(
            id=joint_id,
            position=pos,
            name=name,
        )
    if joint_type == "prismatic":
        axis_data = data.get("axis", [1.0, 0.0])
        axis = (float(axis_data[0]), float(axis_data[1]))
        line_point_data = data.get("line_point", [0.0, 0.0])
        line_point = (float(line_point_data[0]), float(line_point_data[1]))
        slide_distance = data.get("slide_distance", 0.0)
        return PrismaticJoint(
            id=joint_id,
            position=pos,
            name=name,
            axis=axis,
            line_point=line_point,
            slide_distance=slide_distance,
        )
    if joint_type == "tracker":
        return TrackerJoint(
            id=joint_id,
            position=pos,
            name=name,
            ref_joint1_id=data.get("ref_joint1_id", ""),
            ref_joint2_id=data.get("ref_joint2_id", ""),
            distance=data.get("distance", 0.0),
            angle=data.get("angle", 0.0),
        )
    # Default: revolute
    return RevoluteJoint(
        id=joint_id,
        position=pos,
        name=name,
    )


def link_to_dict(link: Link) -> dict[str, Any]:
    """Serialize a link to a dictionary.

    Args:
        link: The link to serialize.

    Returns:
        Dictionary representation of the link.
    """
    data: dict[str, Any] = {
        "id": link.id,
        "type": _link_type_name(link),
        "joints": [j.id for j in link.joints],
    }

    if link.name and link.name != link.id:
        data["name"] = link.name

    # Type-specific attributes
    if isinstance(link, ArcDriverLink):
        # ArcDriverLink must come before DriverLink (it's a subclass check order issue)
        data["angular_velocity"] = link.angular_velocity
        data["arc_start"] = link.arc_start
        data["arc_end"] = link.arc_end
        data["initial_angle"] = link.initial_angle
        if link.motor_joint:
            data["motor_joint"] = link.motor_joint.id
    elif isinstance(link, DriverLink):
        data["angular_velocity"] = link.angular_velocity
        data["initial_angle"] = link.initial_angle
        if link.motor_joint:
            data["motor_joint"] = link.motor_joint.id

    return data


def _link_type_name(link: Link) -> str:
    """Get the type name for a link."""
    if isinstance(link, GroundLink):
        return "ground"
    if isinstance(link, ArcDriverLink):
        return "arc_driver"
    if isinstance(link, DriverLink):
        return "driver"
    return "link"


def link_from_dict(
    data: dict[str, Any],
    joint_map: dict[str, Joint],
) -> Link:
    """Deserialize a link from a dictionary.

    Args:
        data: Dictionary representation of a link.
        joint_map: Map of joint IDs to Joint objects.

    Returns:
        The deserialized Link object.
    """
    link_type = data.get("type", "link")
    link_id = data["id"]
    joint_ids = data.get("joints", [])
    joints = [joint_map[jid] for jid in joint_ids if jid in joint_map]
    name = data.get("name")

    if link_type == "ground":
        return GroundLink(
            id=link_id,
            joints=joints,
            name=name,
        )
    if link_type == "arc_driver":
        motor_joint_id = data.get("motor_joint")
        motor_joint = None
        if motor_joint_id and motor_joint_id in joint_map:
            mj = joint_map[motor_joint_id]
            if isinstance(mj, GroundJoint):
                motor_joint = mj

        return ArcDriverLink(
            id=link_id,
            joints=joints,
            name=name,
            motor_joint=motor_joint,
            angular_velocity=data.get("angular_velocity", 0.1),
            arc_start=data.get("arc_start", 0.0),
            arc_end=data.get("arc_end", 3.14159),
            initial_angle=data.get("initial_angle"),
        )
    if link_type == "driver":
        motor_joint_id = data.get("motor_joint")
        motor_joint = None
        if motor_joint_id and motor_joint_id in joint_map:
            mj = joint_map[motor_joint_id]
            if isinstance(mj, GroundJoint):
                motor_joint = mj

        return DriverLink(
            id=link_id,
            joints=joints,
            name=name,
            motor_joint=motor_joint,
            angular_velocity=data.get("angular_velocity", 0.1),
            initial_angle=data.get("initial_angle", 0.0),
        )
    # Default: regular link
    return Link(
        id=link_id,
        joints=joints,
        name=name,
    )


def mechanism_to_dict(mechanism: Mechanism) -> dict[str, Any]:
    """Serialize a mechanism to a dictionary.

    Args:
        mechanism: The mechanism to serialize.

    Returns:
        Dictionary representation of the mechanism.

    Example:
        >>> data = mechanism_to_dict(mechanism)
        >>> json.dumps(data, indent=2)
    """
    data: dict[str, Any] = {
        "name": mechanism.name,
        "joints": [joint_to_dict(j) for j in mechanism.joints],
        "links": [link_to_dict(link) for link in mechanism.links],
    }

    if mechanism.ground:
        data["ground"] = mechanism.ground.id

    return data


def mechanism_from_dict(data: dict[str, Any]) -> Mechanism:
    """Deserialize a mechanism from a dictionary.

    Args:
        data: Dictionary representation of a mechanism.

    Returns:
        The deserialized Mechanism object.

    Example:
        >>> mechanism = mechanism_from_dict(data)
    """
    name = data.get("name", "")

    # First pass: create all joints
    joints: list[Joint] = []
    joint_map: dict[str, Joint] = {}

    for jdata in data.get("joints", []):
        joint = joint_from_dict(jdata)
        joints.append(joint)
        joint_map[joint.id] = joint

    # Second pass: create all links
    links: list[Link] = []
    ground: GroundLink | None = None
    ground_id = data.get("ground")

    for ldata in data.get("links", []):
        link = link_from_dict(ldata, joint_map)
        links.append(link)

        if isinstance(link, GroundLink) and (ground_id and link.id == ground_id or ground is None):
            ground = link

    return Mechanism(
        name=name,
        joints=joints,
        links=links,
        ground=ground,
    )


def mechanism_to_json(mechanism: Mechanism, path: str | Path) -> None:
    """Save a mechanism to a JSON file.

    Args:
        mechanism: The mechanism to save.
        path: Path to the output JSON file.
    """
    data = mechanism_to_dict(mechanism)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def mechanism_from_json(path: str | Path) -> Mechanism:
    """Load a mechanism from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The loaded Mechanism object.
    """
    with open(path) as f:
        data = json.load(f)
    return mechanism_from_dict(data)


def is_legacy_format(data: dict[str, Any]) -> bool:
    """Check if data is in the legacy (joints/) format.

    The legacy format has a 'joints' key where each joint has
    a 'type' like 'Static', 'Crank', 'Revolute', etc.

    Args:
        data: Dictionary to check.

    Returns:
        True if this appears to be the legacy format.
    """
    joints = data.get("joints", [])
    if not joints:
        return False

    # Check for legacy type names
    legacy_types = {"Static", "Crank", "Revolute", "Fixed", "Prismatic"}
    for jdata in joints:
        jtype = jdata.get("type", "")
        if jtype in legacy_types:
            return True

    return False
