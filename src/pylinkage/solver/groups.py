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

import numpy as np

from .joints import solve_line_line, solve_linear, solve_revolute

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


def solve_pp_dyad(
    line1_pos1: Coord,
    line1_pos2: Coord,
    line2_pos1: Coord,
    line2_pos2: Coord,
) -> Coord:
    """Solve a PP dyad position using line-line intersection.

    A PP dyad consists of one internal node constrained to lie at the
    intersection of two lines. Each line is defined by two points.

    This is used for isomers like T_R_T, T_RT_, _TRT_.

    Args:
        line1_pos1: Position (x, y) of the first point on line 1.
        line1_pos2: Position (x, y) of the second point on line 1.
        line2_pos1: Position (x, y) of the first point on line 2.
        line2_pos2: Position (x, y) of the second point on line 2.

    Returns:
        Computed (x, y) position for the internal node.

    Raises:
        ValueError: If the lines are parallel (no intersection).

    Example:
        >>> pos = solve_pp_dyad(
        ...     line1_pos1=(0.0, 0.0),
        ...     line1_pos2=(4.0, 0.0),
        ...     line2_pos1=(2.0, -1.0),
        ...     line2_pos2=(2.0, 3.0),
        ... )
        >>> # pos is (2.0, 0.0) - intersection of x-axis and line x=2
    """
    new_x, new_y = solve_line_line(
        line1_pos1[0],
        line1_pos1[1],
        line1_pos2[0],
        line1_pos2[1],
        line2_pos1[0],
        line2_pos1[1],
        line2_pos2[0],
        line2_pos2[1],
    )

    if math.isnan(new_x):
        raise ValueError(
            f"Unbuildable PP dyad: lines are parallel. "
            f"line1=({line1_pos1}, {line1_pos2}), "
            f"line2=({line2_pos1}, {line2_pos2})"
        )

    return (new_x, new_y)


def solve_triad(
    constraints: list[tuple[str, str, float]],
    positions: dict[str, Coord],
    internal_ids: tuple[str, str],
    hints: dict[str, Coord] | None = None,
) -> dict[str, Coord]:
    """Solve a triad (Class II Assur group) using Newton-Raphson.

    A triad has 2 internal nodes whose positions are unknown. The
    constraints are distance equations between pairs of nodes (some
    known, some unknown). We solve the system of equations:

        (x_a - x_b)² + (y_a - y_b)² = d² for each edge

    where (x_a, y_a) and (x_b, y_b) are node positions and d is the
    edge distance. Known node positions are substituted, leaving 4
    unknowns (x0, y0, x1, y1) for the 2 internal nodes.

    Args:
        constraints: List of (node_a, node_b, distance) tuples.
            Each defines a distance constraint between two nodes.
            At least 4 constraints are needed (4 unknowns).
        positions: Known positions of anchor nodes.
        internal_ids: The two internal node IDs (order matches unknowns).
        hints: Optional initial guess positions for internal nodes.
            If not provided, centroid of anchors is used.

    Returns:
        Dict mapping internal node IDs to their computed (x, y) positions.

    Raises:
        ValueError: If the system cannot be solved (unbuildable).
    """
    from scipy.optimize import fsolve

    id0, id1 = internal_ids

    # Build initial guess from hints or anchor centroid
    if hints and id0 in hints:
        x0_init, y0_init = hints[id0]
    else:
        anchor_positions = list(positions.values())
        x0_init = sum(p[0] for p in anchor_positions) / len(anchor_positions)
        y0_init = sum(p[1] for p in anchor_positions) / len(anchor_positions)

    if hints and id1 in hints:
        x1_init, y1_init = hints[id1]
    else:
        anchor_positions = list(positions.values())
        x1_init = sum(p[0] for p in anchor_positions) / len(anchor_positions)
        y1_init = sum(p[1] for p in anchor_positions) / len(anchor_positions)
        # Offset slightly to avoid degenerate initial guess
        x1_init += 0.1
        y1_init += 0.1

    x_init = np.array([x0_init, y0_init, x1_init, y1_init])

    def _get_pos(node_id: str, x: np.ndarray) -> tuple[float, float]:
        """Get position of a node — from unknowns or known positions."""
        if node_id == id0:
            return (x[0], x[1])
        elif node_id == id1:
            return (x[2], x[3])
        else:
            return positions[node_id]

    def residuals(x: np.ndarray) -> np.ndarray:
        """Compute distance constraint residuals."""
        res = np.empty(len(constraints))
        for i, (na, nb, dist) in enumerate(constraints):
            pa = _get_pos(na, x)
            pb = _get_pos(nb, x)
            dx = pa[0] - pb[0]
            dy = pa[1] - pb[1]
            # Residual: actual_distance² - target_distance²
            # Using squared distances avoids sqrt and is smoother
            res[i] = dx * dx + dy * dy - dist * dist
        return res

    solution, info, ier, msg = fsolve(residuals, x_init, full_output=True)

    if ier != 1:
        # Try with perturbed initial guess
        x_perturbed = x_init + np.array([0.5, -0.5, -0.5, 0.5])
        solution, info, ier, msg = fsolve(residuals, x_perturbed, full_output=True)

    if ier != 1:
        raise ValueError(
            f"Triad solver did not converge: {msg}. "
            f"Internal nodes: {internal_ids}, "
            f"constraints: {len(constraints)}"
        )

    # Verify solution quality
    res = residuals(solution)
    max_residual = float(np.max(np.abs(res)))
    if max_residual > 1e-6:
        raise ValueError(
            f"Triad solver converged but residual too large: {max_residual:.2e}. "
            f"Mechanism may be unbuildable at this configuration."
        )

    return {
        id0: (float(solution[0]), float(solution[1])),
        id1: (float(solution[2]), float(solution[3])),
    }
