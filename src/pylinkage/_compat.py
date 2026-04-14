"""Container-agnostic accessors for linkage-like objects.

The modern linkage surface exposes two container shapes:

- :class:`pylinkage.simulation.Linkage` — a flat tuple of components
  accessible as ``linkage.components``.
- :class:`pylinkage.mechanism.Mechanism` — a links+joints model with
  ``mechanism.joints`` and ``mechanism.links``.

The helpers here let library code (analysis, bridges, visualizers,
optimizers) classify parts and read positions without depending on a
specific container. Classification is by ``type(...).__name__`` so no
import dependency on the concrete classes is required.
"""

from __future__ import annotations

from typing import Any


def get_parts(linkage: Any) -> list[Any]:
    """Return the ordered list of parts from any linkage container.

    - :class:`~pylinkage.simulation.Linkage` → ``.components``.
    - :class:`~pylinkage.mechanism.Mechanism` → ``.joints``.

    The Mechanism branch runs first because a Mechanism has *both*
    ``.joints`` and ``.links``, and the joints list is the one that
    matches the simulation-linkage iteration order.
    """
    if hasattr(linkage, "joints"):
        return list(linkage.joints)
    return list(linkage.components)


def is_ground(part: Any) -> bool:
    """True for a fixed-frame element (``Ground`` or ``GroundJoint``)."""
    return type(part).__name__ in ("Ground", "GroundJoint")


def is_driver(part: Any) -> bool:
    """True for a motor-driven input.

    Recognises:

    - the actuator API by class name
      (``Crank`` / ``ArcCrank`` / ``LinearActuator``);
    - a ``Mechanism`` ``RevoluteJoint`` that is the output joint of a
      ``DriverLink`` / ``ArcDriverLink`` — looked up through the
      joint's ``_links`` list.
    """
    name = type(part).__name__
    if name in ("Crank", "ArcCrank", "LinearActuator"):
        return True
    for link in getattr(part, "_links", None) or ():
        if type(link).__name__ in ("DriverLink", "ArcDriverLink") and (
            getattr(link, "output_joint", None) is part
        ):
            return True
    return False


def is_dyad(part: Any) -> bool:
    """True for a constrained dyad.

    Recognises the modern dyad classes by name
    (``RRRDyad`` / ``RRPDyad`` / ``PPDyad`` / ``FixedDyad`` /
    ``BinaryDyad`` / cam-follower variants) plus a ``Mechanism``
    ``RevoluteJoint`` or ``PrismaticJoint`` that is neither a ground
    anchor nor a driver output (those are the dependent joints solved
    from neighbouring links — dyad-equivalent).
    """
    name = type(part).__name__
    if name in (
        "RRRDyad",
        "RRPDyad",
        "PPDyad",
        "FixedDyad",
        "BinaryDyad",
        "TranslatingCamFollower",
        "OscillatingCamFollower",
    ):
        return True
    if name in ("RevoluteJoint", "PrismaticJoint"):
        return not is_ground(part) and not is_driver(part)
    return False


def get_coord(part: Any) -> tuple[float | None, float | None]:
    """Return the ``(x, y)`` position of a part.

    Prefers ``part.coord()`` when available (the canonical accessor on
    both :class:`~pylinkage.components.Component` and
    :class:`~pylinkage.mechanism.joint.Joint`); falls back to plain
    ``.x`` / ``.y`` attribute lookup.
    """
    if hasattr(part, "coord"):
        return part.coord()  # type: ignore[no-any-return]
    return (getattr(part, "x", None), getattr(part, "y", None))
