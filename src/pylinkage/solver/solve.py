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
from .groups import solve_pp_dyad, solve_rrp_dyad, solve_rrr_dyad, solve_triad

if TYPE_CHECKING:
    from .._types import Coord, NodeId
    from ..assur.decomposition import DecompositionResult
    from ..assur.groups import AssurGroup


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
    category = group.solver_category
    if category == "circle_circle":
        return _solve_dyad_rrr(group, positions, dimensions, hint_positions)
    elif category == "circle_line":
        return _solve_dyad_rrp(group, positions, dimensions, hint_positions)
    elif category == "line_line":
        return _solve_dyad_pp(group, positions, dimensions, hint_positions)
    elif category == "newton_raphson":
        return _solve_triad(group, positions, dimensions, hint_positions)
    else:
        raise NotImplementedError(
            f"Solver not implemented for group type: {group.joint_signature} (category: {category})"
        )


def _solve_dyad_rrr(
    group: AssurGroup,
    positions: dict[NodeId, Coord],
    dimensions: Dimensions,
    hint_positions: dict[NodeId, Coord] | None,
) -> dict[NodeId, Coord]:
    """Solve a circle-circle dyad group (RRR and all-revolute variants)."""
    if len(group.internal_nodes) != 1:
        raise ValueError("Dyad must have exactly 1 internal node")
    if len(group.anchor_nodes) != 2:
        raise ValueError("Circle-circle dyad must have exactly 2 anchor nodes")

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
    group: AssurGroup,
    positions: dict[NodeId, Coord],
    dimensions: Dimensions,
    hint_positions: dict[NodeId, Coord] | None,
) -> dict[NodeId, Coord]:
    """Solve a circle-line dyad group (RRP, RPR, PRR variants)."""
    if len(group.internal_nodes) != 1:
        raise ValueError("Dyad must have exactly 1 internal node")
    if len(group.anchor_nodes) < 1:
        raise ValueError("Circle-line dyad must have at least 1 anchor node")
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
            "DyadRRP revolute_distance must be set in dimensions"
            f" for edge {group.internal_edges[0] if group.internal_edges else 'N/A'}"
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


def _solve_dyad_pp(
    group: AssurGroup,
    positions: dict[NodeId, Coord],
    dimensions: Dimensions,
    hint_positions: dict[NodeId, Coord] | None,
) -> dict[NodeId, Coord]:
    """Solve a line-line dyad group (PP and multi-prismatic variants)."""
    if len(group.internal_nodes) != 1:
        raise ValueError("Dyad must have exactly 1 internal node")

    # Validate line nodes are set
    if (
        group.line2_node1 is None
        or group.line2_node2 is None
        or group.line_node1 is None
        or group.line_node2 is None
    ):
        raise ValueError("Line-line dyad: all four line nodes must be set")

    internal_id = group.internal_nodes[0]

    # Validate required positions exist
    for node_id in [
        group.line_node1,
        group.line_node2,
        group.line2_node1,
        group.line2_node2,
    ]:
        if node_id not in positions:
            raise ValueError(f"Line node {node_id} position not found")

    try:
        pos = solve_pp_dyad(
            line1_pos1=positions[group.line_node1],
            line1_pos2=positions[group.line_node2],
            line2_pos1=positions[group.line2_node1],
            line2_pos2=positions[group.line2_node2],
        )
    except ValueError as e:
        raise UnbuildableError(
            internal_id,
            message=f"No line-line intersection for PP dyad at {internal_id}: {e}",
        ) from e

    return {internal_id: pos}


def _solve_triad(
    group: AssurGroup,
    positions: dict[NodeId, Coord],
    dimensions: Dimensions,
    hint_positions: dict[NodeId, Coord] | None,
) -> dict[NodeId, Coord]:
    """Solve a triad (Class II) using Newton-Raphson on distance constraints.

    Builds distance constraints from the group's edge_map and dimensions,
    then delegates to solve_triad() for the numerical solve.
    """
    if len(group.internal_nodes) != 2:
        raise ValueError(
            f"Triad must have exactly 2 internal nodes, got {len(group.internal_nodes)}"
        )

    internal_ids = (group.internal_nodes[0], group.internal_nodes[1])

    # Validate anchor positions exist
    for anchor_id in group.anchor_nodes:
        if anchor_id not in positions:
            raise ValueError(f"Anchor {anchor_id} position not found")

    # Build constraints from edge_map + dimensions
    # edge_map: {edge_id: (node_a, node_b)}
    if not hasattr(group, "edge_map") or not group.edge_map:
        raise ValueError(
            "Triad has no edge_map — cannot determine which nodes "
            "each edge connects. Ensure the triad was created by "
            "decompose_assur_groups() or has edge_map populated."
        )

    constraints: list[tuple[str, str, float]] = []
    for edge_id, (node_a, node_b) in group.edge_map.items():
        distance = dimensions.get_edge_distance(edge_id)
        if distance is None:
            raise ValueError(
                f"Triad edge '{edge_id}' ({node_a} ↔ {node_b}) has no distance in dimensions"
            )
        constraints.append((node_a, node_b, distance))

    if len(constraints) < 4:
        raise ValueError(f"Triad needs at least 4 distance constraints, got {len(constraints)}")

    # Build hints
    hints: dict[str, Coord] = {}
    for nid in internal_ids:
        if hint_positions and nid in hint_positions:
            hints[nid] = hint_positions[nid]
        elif dimensions.get_node_position(nid):
            hints[nid] = dimensions.get_node_position(nid)

    try:
        return solve_triad(
            constraints=constraints,
            positions=positions,
            internal_ids=internal_ids,
            hints=hints or None,
        )
    except ValueError as e:
        raise UnbuildableError(
            internal_ids[0],
            message=f"Triad solver failed for {internal_ids}: {e}",
        ) from e


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
