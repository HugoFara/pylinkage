"""Business logic for linkage operations.

The legacy ``pylinkage.Linkage.from_dict`` pathway this service was
built around has been removed. The endpoints below now operate on
``pylinkage.mechanism.Mechanism`` via
``pylinkage.mechanism.serialization.mechanism_from_dict``; any residual
references to ``pl.Linkage`` raise at call time until the full
Mechanism-native rewrite lands.
"""

import logging
from typing import Any

from pylinkage.exceptions import UnbuildableError
from pylinkage.mechanism import Mechanism
from pylinkage.mechanism.serialization import mechanism_from_dict

logger = logging.getLogger(__name__)


def validate_and_build(
    linkage_dict: dict[str, Any],
) -> tuple[Mechanism | None, bool, str]:
    """Validate mechanism data and attempt to build.

    Args:
        linkage_dict: Dictionary representation of the mechanism
            (``pylinkage.mechanism.serialization`` format).

    Returns:
        Tuple of ``(mechanism, is_buildable, error_message)``. The
        error message is safe to surface to API clients: for known-
        domain failures (``UnbuildableError``) it echoes the domain
        message, for any other exception it returns a generic string
        and logs the real traceback server-side.
    """
    try:
        mechanism = mechanism_from_dict(linkage_dict)
        list(mechanism.step(iterations=1))
        return mechanism, True, ""
    except UnbuildableError as e:
        return None, False, f"Unbuildable: {e}"
    except Exception:
        logger.exception("Unexpected error building mechanism")
        return None, False, "Internal error: could not build mechanism"


def get_rotation_period(linkage: Mechanism) -> int:
    """Get the rotation period for the mechanism."""
    return linkage.get_rotation_period()


def run_simulation(
    linkage: Mechanism,
    iterations: int | None = None,
    dt: float = 1.0,
) -> list[list[tuple[float, float]]]:
    """Run simulation and return trajectory data."""
    if iterations is None:
        iterations = linkage.get_rotation_period()

    frames: list[list[tuple[float, float]]] = []
    for positions in linkage.step(iterations=iterations, dt=dt):
        frame = [(pos[0], pos[1]) for pos in positions]
        frames.append(frame)
    return frames


def get_joint_names(linkage: Mechanism) -> list[str]:
    """Get list of joint names in order."""
    return [joint.name for joint in linkage.joints if joint.name is not None]


def linkage_to_response_dict(
    linkage_id: str,
    stored_data: dict[str, Any],
    linkage: Mechanism | None,
    is_buildable: bool,
    error: str = "",
) -> dict[str, Any]:
    """Convert stored mechanism data to response format."""
    result: dict[str, Any] = {
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
