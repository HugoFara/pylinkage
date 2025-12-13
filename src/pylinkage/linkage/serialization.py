"""
Serialization support for linkages.

Provides JSON serialization and deserialization for Linkage objects.
"""


import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..joints import Crank, Fixed, Prismatic, Revolute, Static
from ..joints.joint import Joint

if TYPE_CHECKING:
    from .linkage import Linkage


# Mapping from joint type names to classes
JOINT_TYPES: dict[str, type[Joint]] = {
    "Static": Static,
    "Crank": Crank,
    "Fixed": Fixed,
    "Linear": Prismatic,  # Backward compatible alias
    "Prismatic": Prismatic,
    "Revolute": Revolute,
}


def _serialize_joint_ref(
    joint_ref: Joint | None, linkage_joints: tuple[Joint, ...]
) -> dict[str, Any] | None:
    """Serialize a joint reference.

    If the joint is in the linkage's joints list, reference by name.
    Otherwise (e.g., an implicit Static from tuple conversion), serialize inline.

    Args:
        joint_ref: The joint to serialize.
        linkage_joints: The joints in the parent linkage.

    Returns:
        Either a name string, inline joint dict, or None.
    """
    if joint_ref is None:
        return None

    # If it's in the linkage's joints list, just store the name
    if joint_ref in linkage_joints:
        return {"ref": joint_ref.name}

    # Otherwise, it's likely an implicit Static joint from tuple conversion
    # Serialize it inline with its coordinates
    if isinstance(joint_ref, Static):
        return {
            "inline": True,
            "type": "Static",
            "x": joint_ref.x,
            "y": joint_ref.y,
            "name": joint_ref.name,
        }

    # For other cases, try to reference by name (may fail on load)
    return {"ref": joint_ref.name}


def joint_to_dict(joint: Joint, linkage_joints: tuple[Joint, ...] | None = None) -> dict[str, Any]:
    """Convert a joint to a dictionary representation.

    Args:
        joint: The joint to serialize.
        linkage_joints: Optional tuple of joints in the parent linkage (for reference resolution).

    Returns:
        Dictionary containing the joint's data.
    """
    if linkage_joints is None:
        linkage_joints = ()

    data: dict[str, Any] = {
        "type": joint.__class__.__name__,
        "name": joint.name,
        "x": joint.x,
        "y": joint.y,
    }

    # Serialize parent joint references
    if hasattr(joint, "joint0") and joint.joint0 is not None:
        data["joint0"] = _serialize_joint_ref(joint.joint0, linkage_joints)
    if hasattr(joint, "joint1") and joint.joint1 is not None:
        data["joint1"] = _serialize_joint_ref(joint.joint1, linkage_joints)

    # Type-specific attributes
    if isinstance(joint, (Crank, Fixed)):
        data["distance"] = joint.r
        data["angle"] = joint.angle
    elif isinstance(joint, Revolute):
        data["distance0"] = joint.r0
        data["distance1"] = joint.r1
    elif isinstance(joint, Prismatic):
        data["revolute_radius"] = joint.revolute_radius
        if hasattr(joint, "joint2") and joint.joint2 is not None:
            data["joint2"] = _serialize_joint_ref(joint.joint2, linkage_joints)

    return data


def _resolve_joint_ref(
    ref_data: dict[str, Any] | None,
    joints_by_name: dict[str, Joint],
) -> Joint | None:
    """Resolve a joint reference from serialized data.

    Args:
        ref_data: The serialized reference (name string or inline dict).
        joints_by_name: Map of joint names to already-created joints.

    Returns:
        The resolved joint or None.
    """
    if ref_data is None:
        return None

    # If it's an inline joint definition, create it
    if ref_data.get("inline"):
        return Static(
            x=ref_data.get("x", 0),
            y=ref_data.get("y", 0),
            name=ref_data.get("name"),
        )

    # Otherwise, it's a reference by name
    ref_name = ref_data.get("ref")
    if ref_name:
        return joints_by_name.get(ref_name)

    return None


def joint_from_dict(
    data: dict[str, Any],
    joints_by_name: dict[str, Joint],
) -> Joint:
    """Create a joint from a dictionary representation.

    Args:
        data: Dictionary containing the joint's data.
        joints_by_name: Map of joint names to already-created joints for reference resolution.

    Returns:
        The reconstructed joint.

    Raises:
        ValueError: If the joint type is unknown.
    """
    joint_type = data["type"]
    if joint_type not in JOINT_TYPES:
        raise ValueError(f"Unknown joint type: {joint_type}")

    cls = JOINT_TYPES[joint_type]

    # Resolve parent joint references (now handles inline definitions)
    joint0 = _resolve_joint_ref(data.get("joint0"), joints_by_name)
    joint1 = _resolve_joint_ref(data.get("joint1"), joints_by_name)

    result: Joint
    if cls == Static:
        result = Static(
            x=data.get("x", 0),
            y=data.get("y", 0),
            name=data.get("name"),
        )
    elif cls == Crank:
        result = Crank(
            x=data.get("x"),
            y=data.get("y"),
            joint0=joint0,
            distance=data.get("distance"),
            angle=data.get("angle"),
            name=data.get("name"),
        )
    elif cls == Fixed:
        result = Fixed(
            x=data.get("x"),
            y=data.get("y"),
            joint0=joint0,
            joint1=joint1,
            distance=data.get("distance"),
            angle=data.get("angle"),
            name=data.get("name"),
        )
    elif cls == Revolute:
        result = Revolute(
            x=data.get("x", 0),
            y=data.get("y", 0),
            joint0=joint0,
            joint1=joint1,
            distance0=data.get("distance0"),
            distance1=data.get("distance1"),
            name=data.get("name"),
        )
    elif cls == Prismatic:
        joint2 = _resolve_joint_ref(data.get("joint2"), joints_by_name)
        result = Prismatic(
            x=data.get("x", 0),
            y=data.get("y", 0),
            joint0=joint0,
            joint1=joint1,
            joint2=joint2,
            revolute_radius=data.get("revolute_radius"),
            name=data.get("name"),
        )
    else:
        raise ValueError(f"Unsupported joint type: {joint_type}")

    return result


def linkage_to_dict(linkage: "Linkage") -> dict[str, Any]:
    """Convert a linkage to a dictionary representation.

    Args:
        linkage: The linkage to serialize.

    Returns:
        Dictionary containing the linkage's data.
    """
    return {
        "name": linkage.name,
        "joints": [joint_to_dict(j, linkage.joints) for j in linkage.joints],
        "solve_order": (
            [j.name for j in linkage._solve_order]
            if hasattr(linkage, "_solve_order")
            else None
        ),
    }


def _is_dependency_satisfied(
    ref_data: dict[str, Any] | None,
    joints_by_name: dict[str, Joint],
) -> bool:
    """Check if a joint reference dependency is satisfied.

    Args:
        ref_data: The serialized reference data.
        joints_by_name: Map of joint names to already-created joints.

    Returns:
        True if the dependency is satisfied (or None), False otherwise.
    """
    if ref_data is None:
        return True

    # Inline references are always satisfied (will be created on the fly)
    if ref_data.get("inline"):
        return True

    # Named references need to exist in joints_by_name
    ref_name = ref_data.get("ref")
    return ref_name is None or ref_name in joints_by_name


def linkage_from_dict(data: dict[str, Any]) -> "Linkage":
    """Create a linkage from a dictionary representation.

    This performs a two-pass reconstruction:
    1. First pass: Create all joints with their intrinsic properties
    2. Second pass: Resolve joint references

    Args:
        data: Dictionary containing the linkage's data.

    Returns:
        The reconstructed linkage.
    """
    # Import here to avoid circular imports
    from .linkage import Linkage

    # First pass: create joints without references to resolve names
    joints_data = data.get("joints", [])
    joints_by_name: dict[str, Joint] = {}

    # Create Static joints first (they have no dependencies)
    for joint_data in joints_data:
        if joint_data["type"] == "Static":
            joint = joint_from_dict(joint_data, joints_by_name)
            joints_by_name[joint.name] = joint

    # Create remaining joints in order (may need multiple passes for complex dependencies)
    remaining = [jd for jd in joints_data if jd["type"] != "Static"]
    max_iterations = len(remaining) + 1  # Safety limit

    for _ in range(max_iterations):
        if not remaining:
            break

        still_remaining = []
        for joint_data in remaining:
            # Check if all dependencies are satisfied (handles both refs and inline)
            deps_satisfied = all(
                _is_dependency_satisfied(joint_data.get(dep_key), joints_by_name)
                for dep_key in ("joint0", "joint1", "joint2")
            )

            if deps_satisfied:
                joint = joint_from_dict(joint_data, joints_by_name)
                joints_by_name[joint.name] = joint
            else:
                still_remaining.append(joint_data)

        remaining = still_remaining

    if remaining:
        unresolved = [jd["name"] for jd in remaining]
        raise ValueError(f"Could not resolve dependencies for joints: {unresolved}")

    # Reconstruct joint list in original order
    joints = [joints_by_name[jd["name"]] for jd in joints_data]

    # Reconstruct solve order if present
    order_names = data.get("solve_order")
    order = None
    if order_names:
        order = [joints_by_name[name] for name in order_names if name in joints_by_name]

    return Linkage(
        joints=joints,
        order=order,
        name=data.get("name"),
    )


def save_to_json(linkage: "Linkage", path: str | Path) -> None:
    """Save a linkage to a JSON file.

    Args:
        linkage: The linkage to save.
        path: Path to the output JSON file.
    """
    data = linkage_to_dict(linkage)
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_from_json(path: str | Path) -> "Linkage":
    """Load a linkage from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The reconstructed linkage.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return linkage_from_dict(data)
