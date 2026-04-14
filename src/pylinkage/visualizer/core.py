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

# Type-name sets covering both the legacy joints API (Static/Revolute/Pivot/
# Fixed/Prismatic) and the modern component/actuator/dyad API
# (Ground/RRRDyad/FixedDyad/RRPDyad). Keyed by ``type(...).__name__`` so
# the visualizer doesn't take a hard import on the legacy ``joints``
# module — which is scheduled for removal in 1.0.
_STATIC_TYPE_NAMES = frozenset({"Static", "_StaticBase", "Ground"})
_REVOLUTE_TYPE_NAMES = frozenset({"Revolute", "Pivot", "Fixed", "RRRDyad", "FixedDyad"})
_PRISMATIC_TYPE_NAMES = frozenset({"Prismatic", "RRPDyad"})

# Colors per type name, for matplotlib plotting.
COLOR_SWITCHER: dict[str, str] = {
    "Static": "k",
    "_StaticBase": "k",
    "Ground": "k",
    "Crank": "g",
    "Fixed": "r",
    "FixedDyad": "r",
    "Pivot": "b",
    "Revolute": "b",
    "RRRDyad": "b",
    "Prismatic": "orange",
    "RRPDyad": "orange",
}


def is_static_like(component: Any) -> bool:
    """True if the component is a fixed-frame joint (Static/Ground)."""
    return type(component).__name__ in _STATIC_TYPE_NAMES


def is_revolute_like(component: Any) -> bool:
    """True if the component draws as a pin joint with two anchored parents.

    Matches legacy ``Revolute``/``Pivot``/``Fixed`` and modern
    ``RRRDyad``/``FixedDyad``.
    """
    return type(component).__name__ in _REVOLUTE_TYPE_NAMES


def is_prismatic_like(component: Any) -> bool:
    """True if the component is a slider (legacy ``Prismatic`` or ``RRPDyad``)."""
    return type(component).__name__ in _PRISMATIC_TYPE_NAMES


def _get_color(joint: Any) -> str:
    """Return the matplotlib color for *joint* based on its class name."""
    return COLOR_SWITCHER.get(type(joint).__name__, "")


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
    if joint1 is not None and is_revolute_like(component):
        parents.append(joint1)

    return parents


def resolve_component(
    parent: Any,
    components: list[Any],
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


def build_connections(linkage: Any, components: list[Any]) -> list[tuple[int, int]]:
    """Return ``(parent_idx, child_idx)`` pairs for every bar to draw.

    Three cases:

    - ``Mechanism`` (has ``.links``): iterate each link and emit all pairwise
      combinations of its joint indices. A ternary link with three joints
      yields three bars (the full triangle); a simple binary link yields one.
    - Legacy ``Linkage`` / modern ``SimLinkage`` (joint-/component-centric):
      fall back to per-component :func:`get_parent_pairs`.
    """
    from itertools import combinations

    # Mechanism: derive bars from Link.joints
    links = getattr(linkage, "links", None)
    if links is not None:
        pairs: list[tuple[int, int]] = []
        for link in links:
            link_joints = getattr(link, "joints", None)
            if not link_joints or len(link_joints) < 2:
                continue
            idxs = [components.index(j) for j in link_joints if j in components]
            for a, b in combinations(idxs, 2):
                pairs.append((a, b))
        return pairs

    # Legacy / SimLinkage path
    pairs = []
    for j, comp in enumerate(components):
        for parent in get_parent_pairs(comp):
            p = resolve_component(parent, components)
            if p is not None:
                pairs.append((p, j))
    return pairs
