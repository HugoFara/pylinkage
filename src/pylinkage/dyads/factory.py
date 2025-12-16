"""Factory function for creating dyads from isomer signatures.

This module provides a unified interface for creating any of the 12 dyadic
isomers using a signature string.

Notation:
    R = Revolute joint (pin)
    T = Translating element (slider block)
    _ = Guide (rail/slot that slider moves along)

The 12 isomers map to 3 geometry types:
    - RRR: circle-circle intersection
    - Circle-line isomers (RR_T, RRT_, RT_R, R_T_T, RT_T_, R_TT_, RT__T): RRPDyad
    - Line-line isomers (T_R_T, T_RT_, _TRT_): PPDyad
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._base import BinaryDyad, Dyad, _AnchorProxy
from .pp import PPDyad
from .rrp import RRPDyad
from .rrr import RRRDyad

if TYPE_CHECKING:
    from ..components import Component

# Map isomer signatures to their geometry type
_ISOMER_TO_GEOMETRY: dict[str, str] = {
    # Circle-circle (RRR)
    "RRR": "circle_circle",
    # Circle-line (RRP variants)
    "RR_T": "circle_line",
    "RRT_": "circle_line",
    "RT_R": "circle_line",
    "R_T_T": "circle_line",
    "RT_T_": "circle_line",
    "R_TT_": "circle_line",
    "RT__T": "circle_line",
    # Canonical aliases
    "RRP": "circle_line",
    "RPR": "circle_line",
    "PRR": "circle_line",
    # Line-line (PP variants)
    "T_R_T": "line_line",
    "T_RT_": "line_line",
    "_TRT_": "line_line",
    # Canonical aliases
    "PP": "line_line",
    "PPR": "line_line",
    "PRP": "line_line",
    "RPP": "line_line",
}


def _parse_isomer_signature(signature: str) -> tuple[str, str]:
    """Parse an isomer signature and return (normalized, geometry_type).

    Args:
        signature: The isomer signature (e.g., "RRR", "RT_R", "T_R_T").

    Returns:
        Tuple of (normalized_signature, geometry_type).

    Raises:
        ValueError: If signature is not recognized.
    """
    # Normalize: uppercase, preserve underscores
    normalized = signature.upper()

    if normalized in _ISOMER_TO_GEOMETRY:
        return (normalized, _ISOMER_TO_GEOMETRY[normalized])

    # Try without separators for some canonical forms
    cleaned = normalized.replace("_", "").replace(" ", "")
    if cleaned in _ISOMER_TO_GEOMETRY:
        return (cleaned, _ISOMER_TO_GEOMETRY[cleaned])

    valid = ", ".join(sorted(_ISOMER_TO_GEOMETRY.keys()))
    msg = f"Unknown isomer signature '{signature}'. Valid signatures: {valid}"
    raise ValueError(msg)


def create_dyad(
    signature: str,
    anchors: dict[str, Dyad | _AnchorProxy | Component],
    constraints: dict[str, float] | None = None,
    x: float | None = None,
    y: float | None = None,
    name: str | None = None,
) -> RRRDyad | RRPDyad | PPDyad:
    """Create a dyad from an isomer signature.

    This factory function provides a unified interface for creating any of
    the 12 dyadic isomers. It determines the appropriate geometry type from
    the signature and creates the corresponding dyad class.

    Args:
        signature: Isomer signature (e.g., "RRR", "RT_R", "T_R_T").
        anchors: Dict mapping anchor roles to components. Required keys depend
            on geometry type:
            - circle_circle (RRR): "anchor1", "anchor2"
            - circle_line (RRP variants): "revolute", "line1", "line2"
            - line_line (PP variants): "line1_p1", "line1_p2", "line2_p1", "line2_p2"
        constraints: Dict of constraints. Required keys depend on geometry type:
            - circle_circle (RRR): "distance1", "distance2"
            - circle_line (RRP variants): "distance"
            - line_line (PP variants): none required
        x: Initial x position hint (optional).
        y: Initial y position hint (optional).
        name: Human-readable identifier.

    Returns:
        The appropriate dyad instance (RRRDyad, RRPDyad, or PPDyad).

    Raises:
        ValueError: If signature is unknown or required anchors/constraints
            are missing.

    Examples:
        >>> # Circle-circle (RRR)
        >>> dyad = create_dyad(
        ...     signature="RRR",
        ...     anchors={"anchor1": crank.output, "anchor2": ground2},
        ...     constraints={"distance1": 2.0, "distance2": 1.5},
        ...     name="rocker",
        ... )

        >>> # Circle-line (RT_R - slider crank)
        >>> dyad = create_dyad(
        ...     signature="RT_R",
        ...     anchors={"revolute": crank.output, "line1": L1, "line2": L2},
        ...     constraints={"distance": 1.5},
        ...     name="slider",
        ... )

        >>> # Line-line (T_R_T - double slider)
        >>> dyad = create_dyad(
        ...     signature="T_R_T",
        ...     anchors={
        ...         "line1_p1": A, "line1_p2": B,
        ...         "line2_p1": C, "line2_p2": D,
        ...     },
        ...     name="double_slider",
        ... )
    """
    normalized, geometry = _parse_isomer_signature(signature)
    constraints = constraints or {}

    if geometry == "circle_circle":
        return _create_rrr_dyad(anchors, constraints, x, y, name)
    elif geometry == "circle_line":
        return _create_rrp_dyad(anchors, constraints, x, y, name)
    elif geometry == "line_line":
        return _create_pp_dyad(anchors, constraints, x, y, name)
    else:
        msg = f"Unknown geometry type: {geometry}"
        raise ValueError(msg)


def _create_rrr_dyad(
    anchors: dict[str, Dyad | _AnchorProxy | Component],
    constraints: dict[str, float],
    x: float | None,
    y: float | None,
    name: str | None,
) -> RRRDyad:
    """Create an RRRDyad from anchors and constraints."""
    # Validate required anchors
    required_anchors = ["anchor1", "anchor2"]
    missing = [k for k in required_anchors if k not in anchors]
    if missing:
        msg = f"Missing required anchors for RRR: {missing}"
        raise ValueError(msg)

    # Validate required constraints
    required_constraints = ["distance1", "distance2"]
    missing = [k for k in required_constraints if k not in constraints]
    if missing:
        msg = f"Missing required constraints for RRR: {missing}"
        raise ValueError(msg)

    return RRRDyad(
        anchor1=anchors["anchor1"],
        anchor2=anchors["anchor2"],
        distance1=constraints["distance1"],
        distance2=constraints["distance2"],
        x=x,
        y=y,
        name=name,
    )


def _create_rrp_dyad(
    anchors: dict[str, Dyad | _AnchorProxy | Component],
    constraints: dict[str, float],
    x: float | None,
    y: float | None,
    name: str | None,
) -> RRPDyad:
    """Create an RRPDyad from anchors and constraints."""
    # Validate required anchors
    required_anchors = ["revolute", "line1", "line2"]
    missing = [k for k in required_anchors if k not in anchors]
    if missing:
        msg = f"Missing required anchors for RRP/circle-line: {missing}"
        raise ValueError(msg)

    # Validate required constraints
    required_constraints = ["distance"]
    missing = [k for k in required_constraints if k not in constraints]
    if missing:
        msg = f"Missing required constraints for RRP/circle-line: {missing}"
        raise ValueError(msg)

    return RRPDyad(
        revolute_anchor=anchors["revolute"],
        line_anchor1=anchors["line1"],
        line_anchor2=anchors["line2"],
        distance=constraints["distance"],
        x=x,
        y=y,
        name=name,
    )


def _create_pp_dyad(
    anchors: dict[str, Dyad | _AnchorProxy | Component],
    constraints: dict[str, float],
    x: float | None,
    y: float | None,
    name: str | None,
) -> PPDyad:
    """Create a PPDyad from anchors and constraints."""
    # Validate required anchors
    required_anchors = ["line1_p1", "line1_p2", "line2_p1", "line2_p2"]
    missing = [k for k in required_anchors if k not in anchors]
    if missing:
        msg = f"Missing required anchors for PP/line-line: {missing}"
        raise ValueError(msg)

    # PP dyad has no distance constraints

    return PPDyad(
        line1_anchor1=anchors["line1_p1"],
        line1_anchor2=anchors["line1_p2"],
        line2_anchor1=anchors["line2_p1"],
        line2_anchor2=anchors["line2_p2"],
        x=x,
        y=y,
        name=name,
    )


def get_isomer_geometry(signature: str) -> str:
    """Get the geometry type for an isomer signature.

    Args:
        signature: The isomer signature.

    Returns:
        Geometry type: "circle_circle", "circle_line", or "line_line".

    Raises:
        ValueError: If signature is unknown.
    """
    _, geometry = _parse_isomer_signature(signature)
    return geometry


def get_required_anchors(signature: str) -> list[str]:
    """Get the required anchor names for an isomer signature.

    Args:
        signature: The isomer signature.

    Returns:
        List of required anchor keys.

    Raises:
        ValueError: If signature is unknown.
    """
    _, geometry = _parse_isomer_signature(signature)

    if geometry == "circle_circle":
        return ["anchor1", "anchor2"]
    elif geometry == "circle_line":
        return ["revolute", "line1", "line2"]
    elif geometry == "line_line":
        return ["line1_p1", "line1_p2", "line2_p1", "line2_p2"]
    else:
        return []


def get_required_constraints(signature: str) -> list[str]:
    """Get the required constraint names for an isomer signature.

    Args:
        signature: The isomer signature.

    Returns:
        List of required constraint keys.

    Raises:
        ValueError: If signature is unknown.
    """
    _, geometry = _parse_isomer_signature(signature)

    if geometry == "circle_circle":
        return ["distance1", "distance2"]
    elif geometry == "circle_line":
        return ["distance"]
    elif geometry == "line_line":
        return []  # No constraints
    else:
        return []
