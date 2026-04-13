"""Compatibility utilities for legacy and modern linkage APIs.

Provides type-agnostic accessors so library code can work with both
the legacy ``pylinkage.linkage.Linkage`` (joints API) and the modern
``pylinkage.simulation.Linkage`` (components API) without importing
either directly.
"""

from __future__ import annotations

from typing import Any


def get_parts(linkage: Any) -> list[Any]:
    """Return the ordered list of joints/components from any linkage type.

    Works with:
    - Legacy ``Linkage`` → ``.joints``
    - Modern ``SimLinkage`` → ``.components``
    """
    if hasattr(linkage, "joints"):
        return list(linkage.joints)
    return list(linkage.components)


def is_ground(part: Any) -> bool:
    """Check if a joint/component is a ground (fixed frame) element."""
    name = type(part).__name__
    if name == "Ground":
        return True
    if name in ("Static", "_StaticBase"):
        return getattr(part, "joint0", None) is None
    return False


def is_driver(part: Any) -> bool:
    """Check if a joint/component is a driver (motor input)."""
    name = type(part).__name__
    return name in ("Crank", "ArcCrank", "LinearActuator")


def is_dyad(part: Any) -> bool:
    """Check if a joint/component is a constrained dyad."""
    name = type(part).__name__
    return name in (
        "RRRDyad", "RRPDyad", "PPDyad", "FixedDyad", "BinaryDyad",
        "Revolute", "Pivot", "Fixed", "Prismatic", "Linear",
        "TranslatingCamFollower", "OscillatingCamFollower",
    )


def get_coord(part: Any) -> tuple[float | None, float | None]:
    """Get (x, y) position of a joint/component."""
    if hasattr(part, "coord"):
        return part.coord()  # type: ignore[no-any-return]
    return (getattr(part, "x", None), getattr(part, "y", None))
