"""Linkage management operations for the web editor."""

from typing import Any

import pylinkage as pl
from pylinkage.exceptions import UnbuildableError


def try_simulate(
    linkage: pl.Linkage, iterations: int | None = None
) -> tuple[list[tuple[tuple[float, float], ...]] | None, str | None]:
    """Attempt to simulate the linkage.

    Args:
        linkage: The linkage to simulate.
        iterations: Number of iterations. If None, uses rotation period.

    Returns:
        Tuple of (loci, error_message). If successful, error_message is None.
    """
    try:
        if iterations is None:
            iterations = linkage.get_rotation_period()
        loci = list(linkage.step(iterations=iterations))
        return loci, None
    except UnbuildableError as e:
        return None, f"Cannot build: {e}"
    except Exception as e:
        return None, f"Simulation error: {e}"


def add_joint_to_linkage(
    linkage: pl.Linkage | None, joint_data: dict[str, Any]
) -> tuple[pl.Linkage | None, str | None]:
    """Add a new joint to an existing linkage.

    Args:
        linkage: The current linkage (can be None to start fresh).
        joint_data: Dictionary with joint parameters.

    Returns:
        Tuple of (new_linkage, error_message).
    """
    try:
        # Get existing joints data
        if linkage is not None:
            data = linkage.to_dict()
            joints_data = data.get("joints", [])
        else:
            joints_data = []
            data = {"name": "User Linkage", "joints": [], "solve_order": None}

        # Add new joint
        joints_data.append(joint_data)
        data["joints"] = joints_data

        # Update solve order to include new joint
        if data.get("solve_order") is None:
            data["solve_order"] = [jd["name"] for jd in joints_data]
        else:
            data["solve_order"].append(joint_data["name"])

        # Rebuild linkage
        new_linkage = pl.Linkage.from_dict(data)
        return new_linkage, None
    except Exception as e:
        return None, f"Failed to add joint: {e}"


def delete_joint_from_linkage(
    linkage: pl.Linkage, joint_name: str
) -> tuple[pl.Linkage | None, str | None]:
    """Remove a joint from a linkage.

    Args:
        linkage: The current linkage.
        joint_name: Name of the joint to remove.

    Returns:
        Tuple of (new_linkage, error_message).
    """
    try:
        data = linkage.to_dict()
        joints_data = data.get("joints", [])

        # Remove the joint
        joints_data = [jd for jd in joints_data if jd.get("name") != joint_name]

        if not joints_data:
            # No joints left
            return None, None

        data["joints"] = joints_data

        # Update solve order
        if data.get("solve_order"):
            data["solve_order"] = [n for n in data["solve_order"] if n != joint_name]

        # Rebuild linkage
        new_linkage = pl.Linkage.from_dict(data)
        return new_linkage, None
    except Exception as e:
        return None, f"Failed to delete joint: {e}"


def update_joint_in_linkage(
    linkage: pl.Linkage, joint_name: str, updated_data: dict[str, Any]
) -> tuple[pl.Linkage | None, str | None]:
    """Update a joint's parameters in a linkage.

    Args:
        linkage: The current linkage.
        joint_name: Name of the joint to update.
        updated_data: New joint data (should include all fields).

    Returns:
        Tuple of (new_linkage, error_message).
    """
    try:
        data = linkage.to_dict()
        joints_data = data.get("joints", [])

        # Find and replace the joint
        for i, jd in enumerate(joints_data):
            if jd.get("name") == joint_name:
                joints_data[i] = updated_data
                break
        else:
            return None, f"Joint '{joint_name}' not found"

        data["joints"] = joints_data

        # Rebuild linkage
        new_linkage = pl.Linkage.from_dict(data)
        return new_linkage, None
    except Exception as e:
        return None, f"Failed to update joint: {e}"


def get_joint_names(linkage: pl.Linkage | None) -> list[str]:
    """Get list of joint names in a linkage.

    Args:
        linkage: The linkage to query.

    Returns:
        List of joint names.
    """
    if linkage is None:
        return []
    return [j.name for j in linkage.joints if j.name]


def get_joint_data(linkage: pl.Linkage, joint_name: str) -> dict[str, Any] | None:
    """Get the serialized data for a specific joint.

    Args:
        linkage: The linkage to query.
        joint_name: Name of the joint.

    Returns:
        Joint data dictionary or None if not found.
    """
    data = linkage.to_dict()
    for jd in data.get("joints", []):
        if jd.get("name") == joint_name:
            return jd
    return None


def create_empty_linkage() -> pl.Linkage:
    """Create an empty linkage with no joints.

    Returns:
        An empty linkage.
    """
    return pl.Linkage(joints=[], order=[], name="User Linkage")


def validate_joint_name(linkage: pl.Linkage | None, name: str) -> str | None:
    """Validate that a joint name is unique.

    Args:
        linkage: The current linkage.
        name: The proposed joint name.

    Returns:
        Error message if invalid, None if valid.
    """
    if not name or not name.strip():
        return "Joint name cannot be empty"

    existing_names = get_joint_names(linkage)
    if name in existing_names:
        return f"Joint name '{name}' already exists"

    return None
