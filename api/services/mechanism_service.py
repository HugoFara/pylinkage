"""Business logic for mechanism operations."""

from typing import Any

from pylinkage.exceptions import UnbuildableError
from pylinkage.mechanism import Mechanism
from pylinkage.mechanism.serialization import mechanism_from_dict


def validate_and_build(
    mechanism_dict: dict[str, Any],
) -> tuple[Mechanism | None, bool, str]:
    """Validate mechanism data and attempt to build.

    Args:
        mechanism_dict: Dictionary representation of the mechanism.

    Returns:
        Tuple of (mechanism, is_buildable, error_message).
    """
    try:
        mechanism = mechanism_from_dict(mechanism_dict)
        # Try one step to validate buildability
        for _ in mechanism.step(dt=1.0):
            break  # Just need to test first step
        return mechanism, True, ""
    except UnbuildableError as e:
        return None, False, f"Unbuildable: {e}"
    except Exception as e:
        return None, False, f"Error: {e}"


def get_rotation_period(mechanism: Mechanism) -> int:
    """Get the rotation period for the mechanism."""
    return mechanism.get_rotation_period()


def run_simulation(
    mechanism: Mechanism,
    iterations: int | None = None,
    dt: float = 1.0,
) -> list[list[tuple[float, float]]]:
    """Run simulation and return trajectory data.

    Args:
        mechanism: The mechanism to simulate.
        iterations: Number of iterations. If None, uses rotation_period.
        dt: Time step.

    Returns:
        List of frames, each frame is a list of (x, y) positions.
    """
    if iterations is None:
        iterations = mechanism.get_rotation_period()

    # Reset mechanism to initial state
    mechanism.reset()

    frames: list[list[tuple[float, float]]] = []
    for step_count, positions in enumerate(mechanism.step(dt=dt)):
        frame = [
            (pos[0] if pos[0] is not None else 0.0, pos[1] if pos[1] is not None else 0.0)
            for pos in positions
        ]
        frames.append(frame)
        if step_count + 1 >= iterations:
            break
    return frames


def get_joint_names(mechanism: Mechanism) -> list[str]:
    """Get list of joint names in order."""
    return [joint.name or joint.id for joint in mechanism.joints]


def mechanism_to_response_dict(
    mechanism_id: str,
    stored_data: dict[str, Any],
    mechanism: Mechanism | None,
    is_buildable: bool,
    error: str = "",
) -> dict[str, Any]:
    """Convert stored mechanism data to response format.

    Args:
        mechanism_id: The mechanism ID.
        stored_data: Data from storage.
        mechanism: The built Mechanism object (if buildable).
        is_buildable: Whether the mechanism is buildable.
        error: Error message if not buildable.

    Returns:
        Dictionary suitable for MechanismResponse.
    """
    result = {
        "id": mechanism_id,
        "name": stored_data.get("name", "Unnamed"),
        "joints": stored_data.get("joints", []),
        "links": stored_data.get("links", []),
        "ground": stored_data.get("ground"),
        "created_at": stored_data["created_at"],
        "updated_at": stored_data["updated_at"],
        "is_buildable": is_buildable,
        "rotation_period": None,
        "error": error if error else None,
    }

    if mechanism is not None and is_buildable:
        result["rotation_period"] = get_rotation_period(mechanism)

    return result
