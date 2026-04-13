"""
Core features for visualization.

This module provides shared utilities used by all visualization backends
(matplotlib, plotly, drawsvg). It contains linkage introspection helpers
that work with both the legacy ``pylinkage.linkage.Linkage`` (joints API)
and the modern ``pylinkage.simulation.Linkage`` (components API).

For symbol definitions see :mod:`symbols`.
"""

from __future__ import annotations

from typing import Any

from ..components._base import Component
from ..joints.crank import Crank
from ..joints.fixed import Fixed
from ..joints.joint import Joint
from ..joints.joint import _StaticBase as Static
from ..joints.prismatic import Prismatic
from ..joints.revolute import Pivot, Revolute

# Colors to use for matplotlib plotting (backwards compatible)
COLOR_SWITCHER: dict[type[Joint], str] = {
    Static: "k",
    Crank: "g",
    Fixed: "r",
    Pivot: "b",
    Revolute: "b",
    Prismatic: "orange",
}


def _get_color(joint: Joint | Component) -> str:
    """Search in COLOR_SWITCHER for the corresponding color.

    Args:
        joint: The joint to get the color for.

    Returns:
        The color string for matplotlib.
    """
    for joint_type, color in COLOR_SWITCHER.items():
        if isinstance(joint, joint_type):
            return color
    return ""


# ------------------------------------------------------------------
# Shared helpers for both legacy and modern linkage APIs
# ------------------------------------------------------------------


def get_components(linkage: Any) -> list[Any]:
    """Return the ordered list of joints/components from either API.

    Works with both legacy ``Linkage.joints`` and modern
    ``SimLinkage.components``.
    """
    from .._compat import get_parts

    return get_parts(linkage)


def get_parent_pairs(component: Any) -> list[Any]:
    """Return the parent components that should draw links to *component*.

    Works with both legacy joints (``joint0``, ``joint1``) and modern
    components (``anchor``, ``anchor1``, ``anchor2``).
    """
    parents: list[Any] = []

    # Modern API: Crank.anchor, BinaryDyad.anchor1/anchor2
    anchor = getattr(component, "anchor", None)
    if anchor is not None:
        parents.append(anchor)
        return parents  # Crank has exactly one parent

    anchor1 = getattr(component, "anchor1", None)
    anchor2 = getattr(component, "anchor2", None)
    if anchor1 is not None:
        parents.append(anchor1)
    if anchor2 is not None:
        parents.append(anchor2)
    if parents:
        return parents

    # Legacy API: joint0, joint1
    joint0 = getattr(component, "joint0", None)
    if joint0 is not None:
        parents.append(joint0)

    joint1 = getattr(component, "joint1", None)
    if joint1 is not None and (
        isinstance(component, (Fixed, Pivot))
        or type(component).__name__ == "Revolute"
    ):
        parents.append(joint1)

    return parents


def resolve_component(
    parent: Any, components: list[Any],
) -> int | None:
    """Resolve a parent reference to its index in the component list.

    Handles direct membership, ``_AnchorProxy._parent``, and legacy
    ``joint0``/``joint1`` references.

    Returns:
        Index into *components*, or ``None`` if not found.
    """
    if parent in components:
        return components.index(parent)
    # AnchorProxy (from Crank.output)
    actual = getattr(parent, "_parent", None)
    if actual is not None and actual in components:
        return components.index(actual)
    return None
