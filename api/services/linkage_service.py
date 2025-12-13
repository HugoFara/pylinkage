"""Business logic for linkage operations."""

from typing import Any

import pylinkage as pl
from pylinkage.exceptions import UnbuildableError


def validate_and_build(linkage_dict: dict[str, Any]) -> tuple[pl.Linkage | None, bool, str]:
    """Validate linkage data and attempt to build.

    Args:
        linkage_dict: Dictionary representation of the linkage.

    Returns:
        Tuple of (linkage, is_buildable, error_message).
    """
    try:
        linkage = pl.Linkage.from_dict(linkage_dict)
        # Try one step to validate buildability
        list(linkage.step(iterations=1))
        return linkage, True, ""
    except UnbuildableError as e:
        return None, False, f"Unbuildable: {e}"
    except Exception as e:
        return None, False, f"Error: {e}"


def get_rotation_period(linkage: pl.Linkage) -> int:
    """Get the rotation period for the linkage."""
    return linkage.get_rotation_period()


def run_simulation(
    linkage: pl.Linkage,
    iterations: int | None = None,
    dt: float = 1.0,
) -> list[list[tuple[float, float]]]:
    """Run simulation and return trajectory data.

    Args:
        linkage: The linkage to simulate.
        iterations: Number of iterations. If None, uses rotation_period.
        dt: Time step (angle increment for cranks).

    Returns:
        List of frames, each frame is a list of (x, y) positions.
    """
    if iterations is None:
        iterations = linkage.get_rotation_period()

    frames: list[list[tuple[float, float]]] = []
    for positions in linkage.step(iterations=iterations, dt=dt):
        frame = [(pos[0], pos[1]) for pos in positions]
        frames.append(frame)
    return frames


def get_joint_names(linkage: pl.Linkage) -> list[str]:
    """Get list of joint names in order."""
    return [joint.name for joint in linkage.joints]


def linkage_to_response_dict(
    linkage_id: str,
    stored_data: dict[str, Any],
    linkage: pl.Linkage | None,
    is_buildable: bool,
    error: str = "",
) -> dict[str, Any]:
    """Convert stored linkage data to response format.

    Args:
        linkage_id: The linkage ID.
        stored_data: Data from storage.
        linkage: The built Linkage object (if buildable).
        is_buildable: Whether the linkage is buildable.
        error: Error message if not buildable.

    Returns:
        Dictionary suitable for LinkageResponse.
    """
    result = {
        "id": linkage_id,
        "name": stored_data.get("name", "Unnamed"),
        "joints": stored_data.get("joints", []),
        "solve_order": stored_data.get("solve_order"),
        "created_at": stored_data["created_at"],
        "updated_at": stored_data["updated_at"],
        "is_buildable": is_buildable,
        "rotation_period": None,
        "error": error if error else None,
    }

    if linkage is not None and is_buildable:
        result["rotation_period"] = get_rotation_period(linkage)

    return result
