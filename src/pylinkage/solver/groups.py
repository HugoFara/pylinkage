"""Solver functions for Assur groups.

This module provides standalone solving functions for Assur groups.
These functions operate on pure geometric data (positions, distances)
and delegate to the numba-optimized joint solvers.

The Assur group classes in pylinkage.assur define structure (logical properties),
while this module provides the solving behavior.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .joints import solve_linear, solve_revolute

if TYPE_CHECKING:
    from .._types import Coord


def solve_rrr_dyad(
    anchor0_pos: Coord,
    anchor1_pos: Coord,
    distance0: float,
    distance1: float,
    hint: Coord | None = None,
) -> Coord:
    """Solve an RRR dyad position using circle-circle intersection.

    An RRR dyad consists of one internal node connected to two anchor nodes
    by links of known lengths. The internal node position is at the intersection
    of two circles centered at the anchors.

    Args:
        anchor0_pos: Position (x, y) of the first anchor node.
        anchor1_pos: Position (x, y) of the second anchor node.
        distance0: Distance from anchor0 to internal node.
        distance1: Distance from anchor1 to internal node.
        hint: Current position for disambiguation when two solutions exist.
              If None, defaults to anchor0_pos.

    Returns:
        Computed (x, y) position for the internal node.

    Raises:
        ValueError: If the circles don't intersect (unbuildable configuration).

    Example:
        >>> pos = solve_rrr_dyad(
        ...     anchor0_pos=(0.0, 0.0),
        ...     anchor1_pos=(3.0, 0.0),
        ...     distance0=2.0,
        ...     distance1=2.0,
        ... )
        >>> # pos is approximately (1.5, 1.32...) or (1.5, -1.32...)
    """
    hint_x = hint[0] if hint else anchor0_pos[0]
    hint_y = hint[1] if hint else anchor0_pos[1]

    new_x, new_y = solve_revolute(
        hint_x,
        hint_y,
        anchor0_pos[0],
        anchor0_pos[1],
        distance0,
        anchor1_pos[0],
        anchor1_pos[1],
        distance1,
    )

    if math.isnan(new_x):
        raise ValueError(
            f"Unbuildable RRR dyad: circles don't intersect. "
            f"anchor0={anchor0_pos}, anchor1={anchor1_pos}, "
            f"d0={distance0}, d1={distance1}"
        )

    return (new_x, new_y)


def solve_rrp_dyad(
    anchor_pos: Coord,
    line_pos1: Coord,
    line_pos2: Coord,
    distance: float,
    hint: Coord | None = None,
) -> Coord:
    """Solve an RRP dyad position using circle-line intersection.

    An RRP dyad consists of one internal node connected to an anchor by a link
    of known length, and constrained to lie on a line defined by two other nodes.
    The internal node position is at the intersection of the circle and line.

    Args:
        anchor_pos: Position (x, y) of the revolute anchor node.
        line_pos1: Position (x, y) of the first node defining the line.
        line_pos2: Position (x, y) of the second node defining the line.
        distance: Distance from anchor to internal node.
        hint: Current position for disambiguation when two solutions exist.
              If None, defaults to anchor_pos.

    Returns:
        Computed (x, y) position for the internal node.

    Raises:
        ValueError: If the circle and line don't intersect (unbuildable).

    Example:
        >>> pos = solve_rrp_dyad(
        ...     anchor_pos=(0.0, 0.0),
        ...     line_pos1=(0.0, 2.0),
        ...     line_pos2=(4.0, 2.0),
        ...     distance=2.5,
        ... )
        >>> # pos is on the line y=2 at distance 2.5 from origin
    """
    hint_x = hint[0] if hint else anchor_pos[0]
    hint_y = hint[1] if hint else anchor_pos[1]

    new_x, new_y = solve_linear(
        hint_x,
        hint_y,
        anchor_pos[0],
        anchor_pos[1],
        distance,
        line_pos1[0],
        line_pos1[1],
        line_pos2[0],
        line_pos2[1],
    )

    if math.isnan(new_x):
        raise ValueError(
            f"Unbuildable RRP dyad: circle-line don't intersect. "
            f"anchor={anchor_pos}, line=({line_pos1}, {line_pos2}), "
            f"distance={distance}"
        )

    return (new_x, new_y)
