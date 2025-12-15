"""High-level solving functions for linkage mechanisms.

This module provides the primary API for solving linkage kinematics using
Assur group decomposition. It bridges between the structural analysis
(assur module) and the low-level numba solvers (solver.joints).

The key functions are:
- solve_group(): Solve a single Assur group given anchor positions and dimensions
- solve_decomposition(): Solve all groups in decomposition order

These functions are the canonical location for solving behavior.
The Assur group classes themselves are pure topology (no solve methods).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..dimensions import Dimensions
from ..exceptions import UnbuildableError
from .groups import solve_rrp_dyad, solve_rrr_dyad

if TYPE_CHECKING:
    from .._types import Coord, NodeId
    from ..assur.decomposition import DecompositionResult
    from ..assur.graph import LinkageGraph
    from ..assur.groups import AssurGroup, DyadRRP, DyadRRR


def solve_group(
    group: AssurGroup,
    positions: dict[NodeId, Coord],
    dimensions: Dimensions,
    hint_positions: dict[NodeId, Coord] | None = None,
) -> dict[NodeId, Coord]:
    """Solve positions for an Assur group.

    This is the main dispatch function that routes to the appropriate
    solver based on group type (RRR, RRP, etc.).

    Args:
        group: The Assur group to solve (topology only).
        positions: Known positions of anchor nodes. Must contain entries
            for all nodes in group.anchor_nodes.
        dimensions: Dimensions object containing edge distances.
        hint_positions: Optional positions to use as hints for disambiguation
            when there are multiple solutions.

    Returns:
        Dict mapping internal node IDs to their computed (x, y) positions.

    Raises:
        UnbuildableError: If the group cannot be solved geometrically.
        NotImplementedError: If the group type is not yet supported.
        ValueError: If required constraints or positions are missing.

    Example:
        >>> from pylinkage.assur import DyadRRR, decompose_assur_groups
        >>> from pylinkage.dimensions import Dimensions
        >>> result = decompose_assur_groups(graph)
        >>> dims = Dimensions(node_positions={...}, edge_distances={...})
        >>> positions = {n: dims.get_node_position(n) for n in result.ground}
        >>> for group in result.groups:
        ...     new_pos = solve_group(group, positions, dims)
        ...     positions.update(new_pos)
    """
    # Import here to avoid circular imports at module load time
    from ..assur.groups import DyadRRP, DyadRRR

    if isinstance(group, DyadRRR):
        return _solve_dyad_rrr(group, positions, dimensions, hint_positions)
    elif isinstance(group, DyadRRP):
        return _solve_dyad_rrp(group, positions, dimensions, hint_positions)
    else:
        raise NotImplementedError(
            f"Solver not implemented for group type: {group.joint_signature}"
        )


def _solve_dyad_rrr(
    group: DyadRRR,
    positions: dict[NodeId, Coord],
    dimensions: Dimensions,
    hint_positions: dict[NodeId, Coord] | None,
) -> dict[NodeId, Coord]:
    """Solve an RRR dyad group."""
    if len(group.internal_nodes) != 1:
        raise ValueError("DyadRRR must have exactly 1 internal node")
    if len(group.anchor_nodes) != 2:
        raise ValueError("DyadRRR must have exactly 2 anchor nodes")

    internal_id = group.internal_nodes[0]
    anchor0_id, anchor1_id = group.anchor_nodes

    # Validate anchor positions exist
    if anchor0_id not in positions:
        raise ValueError(f"Anchor {anchor0_id} position not found")
    if anchor1_id not in positions:
        raise ValueError(f"Anchor {anchor1_id} position not found")

    # Get distances from dimensions using edge IDs
    distance0: float | None = None
    distance1: float | None = None
    if len(group.internal_edges) >= 2:
        distance0 = dimensions.get_edge_distance(group.internal_edges[0])
        distance1 = dimensions.get_edge_distance(group.internal_edges[1])
    elif len(group.internal_edges) == 1:
        distance0 = dimensions.get_edge_distance(group.internal_edges[0])

    if distance0 is None or distance1 is None:
        raise ValueError(
            f"DyadRRR distances must be set in dimensions for edges {group.internal_edges}"
        )

    # Get hint from hint_positions if available
    hint: Coord | None = None
    if hint_positions and internal_id in hint_positions:
        hint = hint_positions[internal_id]
    elif dimensions.get_node_position(internal_id):
        hint = dimensions.get_node_position(internal_id)

    try:
        pos = solve_rrr_dyad(
            anchor0_pos=positions[anchor0_id],
            anchor1_pos=positions[anchor1_id],
            distance0=distance0,
            distance1=distance1,
            hint=hint,
        )
    except ValueError as e:
        raise UnbuildableError(
            internal_id,
            message=f"No circle intersection for RRR dyad at {internal_id}: {e}",
        ) from e

    return {internal_id: pos}


def _solve_dyad_rrp(
    group: DyadRRP,
    positions: dict[NodeId, Coord],
    dimensions: Dimensions,
    hint_positions: dict[NodeId, Coord] | None,
) -> dict[NodeId, Coord]:
    """Solve an RRP dyad group."""
    if len(group.internal_nodes) != 1:
        raise ValueError("DyadRRP must have exactly 1 internal node")
    if len(group.anchor_nodes) < 1:
        raise ValueError("DyadRRP must have at least 1 anchor node")
    if group.line_node1 is None or group.line_node2 is None:
        raise ValueError("DyadRRP line nodes must be set")

    internal_id = group.internal_nodes[0]
    revolute_anchor = group.anchor_nodes[0]

    # Validate required positions exist
    if revolute_anchor not in positions:
        raise ValueError(f"Anchor {revolute_anchor} position not found")
    if group.line_node1 not in positions:
        raise ValueError(f"Line node {group.line_node1} position not found")
    if group.line_node2 not in positions:
        raise ValueError(f"Line node {group.line_node2} position not found")

    # Get distance from dimensions using edge ID
    revolute_distance: float | None = None
    if len(group.internal_edges) >= 1:
        revolute_distance = dimensions.get_edge_distance(group.internal_edges[0])

    if revolute_distance is None:
        raise ValueError(
            f"DyadRRP revolute_distance must be set in dimensions for edge {group.internal_edges[0] if group.internal_edges else 'N/A'}"
        )

    # Get hint from hint_positions if available
    hint: Coord | None = None
    if hint_positions and internal_id in hint_positions:
        hint = hint_positions[internal_id]
    elif dimensions.get_node_position(internal_id):
        hint = dimensions.get_node_position(internal_id)

    try:
        pos = solve_rrp_dyad(
            anchor_pos=positions[revolute_anchor],
            line_pos1=positions[group.line_node1],
            line_pos2=positions[group.line_node2],
            distance=revolute_distance,
            hint=hint,
        )
    except ValueError as e:
        raise UnbuildableError(
            internal_id,
            message=f"No circle-line intersection for RRP dyad at {internal_id}: {e}",
        ) from e

    return {internal_id: pos}


def solve_decomposition(
    result: DecompositionResult,
    dimensions: Dimensions,
    initial_positions: dict[NodeId, Coord] | None = None,
) -> dict[NodeId, Coord]:
    """Solve the kinematics using a decomposition result and dimensions.

    This function computes the positions of all nodes by solving
    the Assur groups in order. It is the canonical location for
    decomposition-based solving.

    Args:
        result: The decomposition result with groups in solving order.
            Must have a valid graph reference.
        dimensions: Dimensions object containing node positions and edge distances.
        initial_positions: Optional override positions for nodes.
            If not provided, positions from dimensions are used.

    Returns:
        Dictionary mapping all node IDs to their computed (x, y) positions.

    Raises:
        UnbuildableError: If any group cannot be solved.
        ValueError: If required positions are missing or graph is None.

    Example:
        >>> from pylinkage.assur import decompose_assur_groups
        >>> from pylinkage.dimensions import Dimensions
        >>> result = decompose_assur_groups(graph)
        >>> dims = Dimensions(node_positions={...}, edge_distances={...})
        >>> positions = solve_decomposition(result, dims)
        >>> print(positions)
    """
    if result.graph is None:
        raise ValueError("DecompositionResult has no graph reference")

    positions: dict[NodeId, Coord] = {}

    # Initialize ground positions
    for node_id in result.ground:
        if initial_positions and node_id in initial_positions:
            positions[node_id] = initial_positions[node_id]
        else:
            pos = dimensions.get_node_position(node_id)
            if pos is not None:
                positions[node_id] = pos
            else:
                raise ValueError(f"Ground node {node_id} has no position in dimensions")

    # Initialize driver positions
    for node_id in result.drivers:
        if initial_positions and node_id in initial_positions:
            positions[node_id] = initial_positions[node_id]
        else:
            pos = dimensions.get_node_position(node_id)
            if pos is not None:
                positions[node_id] = pos
            else:
                raise ValueError(f"Driver node {node_id} has no position in dimensions")

    # Solve each group in order
    for group in result.groups:
        new_positions = solve_group(group, positions, dimensions)
        positions.update(new_positions)

    return positions
