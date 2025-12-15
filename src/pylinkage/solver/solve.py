"""High-level solving functions for linkage mechanisms.

This module provides the primary API for solving linkage kinematics using
Assur group decomposition. It bridges between the structural analysis
(assur module) and the low-level numba solvers (solver.joints).

The key functions are:
- solve_group(): Solve a single Assur group given anchor positions
- solve_decomposition(): Solve all groups in decomposition order

These functions are the canonical location for solving behavior.
The Assur group classes themselves are pure data (no solve methods).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    graph: LinkageGraph | None = None,
) -> dict[NodeId, Coord]:
    """Solve positions for an Assur group.

    This is the main dispatch function that routes to the appropriate
    solver based on group type (RRR, RRP, etc.).

    Args:
        group: The Assur group to solve. Must have its constraint
            attributes (distances, line nodes) properly set.
        positions: Known positions of anchor nodes. Must contain entries
            for all nodes in group.anchor_nodes.
        graph: Optional linkage graph for looking up current positions
            as hints for disambiguation.

    Returns:
        Dict mapping internal node IDs to their computed (x, y) positions.

    Raises:
        UnbuildableError: If the group cannot be solved geometrically.
        NotImplementedError: If the group type is not yet supported.
        ValueError: If required constraints or positions are missing.

    Example:
        >>> from pylinkage.assur import DyadRRR, decompose_assur_groups
        >>> result = decompose_assur_groups(graph)
        >>> positions = {n: graph.nodes[n].position for n in result.ground}
        >>> for group in result.groups:
        ...     new_pos = solve_group(group, positions, graph)
        ...     positions.update(new_pos)
    """
    # Import here to avoid circular imports at module load time
    from ..assur.groups import DyadRRP, DyadRRR

    if isinstance(group, DyadRRR):
        return _solve_dyad_rrr(group, positions, graph)
    elif isinstance(group, DyadRRP):
        return _solve_dyad_rrp(group, positions, graph)
    else:
        raise NotImplementedError(
            f"Solver not implemented for group type: {group.joint_signature}"
        )


def _solve_dyad_rrr(
    group: DyadRRR,
    positions: dict[NodeId, Coord],
    graph: LinkageGraph | None,
) -> dict[NodeId, Coord]:
    """Solve an RRR dyad group."""
    if len(group.internal_nodes) != 1:
        raise ValueError("DyadRRR must have exactly 1 internal node")
    if len(group.anchor_nodes) != 2:
        raise ValueError("DyadRRR must have exactly 2 anchor nodes")
    if group.distance0 is None or group.distance1 is None:
        raise ValueError("DyadRRR distances must be set")

    internal_id = group.internal_nodes[0]
    anchor0_id, anchor1_id = group.anchor_nodes

    # Validate anchor positions exist
    if anchor0_id not in positions:
        raise ValueError(f"Anchor {anchor0_id} position not found")
    if anchor1_id not in positions:
        raise ValueError(f"Anchor {anchor1_id} position not found")

    # Get hint from graph if available
    hint: Coord | None = None
    if graph is not None and internal_id in graph.nodes:
        node = graph.nodes[internal_id]
        if node.position[0] is not None and node.position[1] is not None:
            hint = (node.position[0], node.position[1])

    try:
        pos = solve_rrr_dyad(
            anchor0_pos=positions[anchor0_id],
            anchor1_pos=positions[anchor1_id],
            distance0=group.distance0,
            distance1=group.distance1,
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
    graph: LinkageGraph | None,
) -> dict[NodeId, Coord]:
    """Solve an RRP dyad group."""
    if len(group.internal_nodes) != 1:
        raise ValueError("DyadRRP must have exactly 1 internal node")
    if len(group.anchor_nodes) < 1:
        raise ValueError("DyadRRP must have at least 1 anchor node")
    if group.revolute_distance is None:
        raise ValueError("DyadRRP revolute_distance must be set")
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

    # Get hint from graph if available
    hint: Coord | None = None
    if graph is not None and internal_id in graph.nodes:
        node = graph.nodes[internal_id]
        if node.position[0] is not None and node.position[1] is not None:
            hint = (node.position[0], node.position[1])

    try:
        pos = solve_rrp_dyad(
            anchor_pos=positions[revolute_anchor],
            line_pos1=positions[group.line_node1],
            line_pos2=positions[group.line_node2],
            distance=group.revolute_distance,
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
    initial_positions: dict[NodeId, Coord] | None = None,
) -> dict[NodeId, Coord]:
    """Solve the kinematics using a decomposition result.

    This function computes the positions of all nodes by solving
    the Assur groups in order. It is the canonical location for
    decomposition-based solving (replacing assur.decomposition.solve_decomposition).

    Args:
        result: The decomposition result with groups in solving order.
            Must have a valid graph reference.
        initial_positions: Optional override positions for nodes.
            If not provided, positions from graph nodes are used.

    Returns:
        Dictionary mapping all node IDs to their computed (x, y) positions.

    Raises:
        UnbuildableError: If any group cannot be solved.
        ValueError: If required positions are missing or graph is None.

    Example:
        >>> from pylinkage.assur import decompose_assur_groups
        >>> result = decompose_assur_groups(graph)
        >>> positions = solve_decomposition(result)
        >>> print(positions)
    """
    if result.graph is None:
        raise ValueError("DecompositionResult has no graph reference")

    graph = result.graph
    positions: dict[NodeId, Coord] = {}

    # Initialize ground positions
    for node_id in result.ground:
        node = graph.nodes[node_id]
        if initial_positions and node_id in initial_positions:
            positions[node_id] = initial_positions[node_id]
        elif node.position[0] is not None and node.position[1] is not None:
            positions[node_id] = (node.position[0], node.position[1])
        else:
            raise ValueError(f"Ground node {node_id} has no position")

    # Initialize driver positions
    for node_id in result.drivers:
        node = graph.nodes[node_id]
        if initial_positions and node_id in initial_positions:
            positions[node_id] = initial_positions[node_id]
        elif node.position[0] is not None and node.position[1] is not None:
            positions[node_id] = (node.position[0], node.position[1])
        else:
            raise ValueError(f"Driver node {node_id} has no position")

    # Solve each group in order
    for group in result.groups:
        new_positions = solve_group(group, positions, graph)
        positions.update(new_positions)

    return positions
