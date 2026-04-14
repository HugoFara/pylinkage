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
    if name in ("Ground", "GroundJoint"):
        return True
    if name in ("Static", "_StaticBase"):
        return getattr(part, "joint0", None) is None
    return False


def is_driver(part: Any) -> bool:
    """Check if a joint/component is a driver (motor input).

    Recognises:

    - the modern component/actuator API by class name
      (``Crank`` / ``ArcCrank`` / ``LinearActuator``);
    - a ``Mechanism`` joint that sits as the output of a ``DriverLink`` /
      ``ArcDriverLink`` — looked up via the joint's ``_links`` list.
    """
    name = type(part).__name__
    if name in ("Crank", "ArcCrank", "LinearActuator"):
        return True
    links = getattr(part, "_links", None)
    if links:
        for link in links:
            link_name = type(link).__name__
            if link_name in ("DriverLink", "ArcDriverLink") and (
                getattr(link, "output_joint", None) is part
            ):
                return True
    return False


def is_dyad(part: Any) -> bool:
    """Check if a joint/component is a constrained dyad.

    Recognises the modern dyad classes by name plus a ``Mechanism``
    ``RevoluteJoint`` / ``PrismaticJoint`` that is neither ground nor a
    driver output (those are dependent joints whose position is solved
    from neighbouring links — i.e. dyad-equivalent).
    """
    name = type(part).__name__
    if name in (
        "RRRDyad",
        "RRPDyad",
        "PPDyad",
        "FixedDyad",
        "BinaryDyad",
        "Revolute",
        "Pivot",
        "Fixed",
        "Prismatic",
        "Linear",
        "TranslatingCamFollower",
        "OscillatingCamFollower",
    ):
        return True
    if name in ("RevoluteJoint", "PrismaticJoint") and not is_ground(part):
        return not is_driver(part)
    return False


def get_coord(part: Any) -> tuple[float | None, float | None]:
    """Get (x, y) position of a joint/component."""
    if hasattr(part, "coord"):
        return part.coord()  # type: ignore[no-any-return]
    return (getattr(part, "x", None), getattr(part, "y", None))
