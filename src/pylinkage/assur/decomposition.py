"""Assur group decomposition algorithm.

This module provides algorithms to decompose a linkage graph into
Assur groups, which enables systematic kinematic analysis.

The decomposition algorithm:
1. Identifies ground (frame) nodes
2. Identifies driver (input) nodes
3. Iteratively finds solvable Assur groups where all anchors are known
4. Returns groups in solving order
"""

from dataclasses import dataclass, field

from .._types import Coord
from ._types import JointType, NodeId, NodeRole
from .graph import LinkageGraph
from .groups import AssurGroup, DyadRRP, DyadRRR


@dataclass
class DecompositionResult:
    """Result of Assur group decomposition.

    Contains the ordered sequence of structural elements needed to
    solve the linkage kinematics.

    Attributes:
        ground: Node IDs of ground/frame points (fixed, don't move).
        drivers: Node IDs of driver/input joints (cranks, motors).
        groups: Assur groups in solving order.
        graph: Reference to the original graph.

    Example:
        >>> result = decompose_assur_groups(graph)
        >>> print(f"Ground: {result.ground}")
        >>> print(f"Drivers: {result.drivers}")
        >>> for group in result.groups:
        ...     print(f"  {group.joint_signature}: {group.internal_nodes}")
    """

    ground: list[NodeId] = field(default_factory=list)
    drivers: list[NodeId] = field(default_factory=list)
    groups: list[AssurGroup] = field(default_factory=list)
    graph: LinkageGraph | None = None

    def solve_order(self) -> list[NodeId | AssurGroup]:
        """Return the complete solving order (drivers + groups).

        Returns:
            List of driver node IDs followed by AssurGroup objects,
            in the order they should be solved.
        """
        result: list[NodeId | AssurGroup] = []
        result.extend(self.drivers)
        result.extend(self.groups)
        return result

    def all_nodes_in_order(self) -> list[NodeId]:
        """Return all node IDs in solving order.

        Returns:
            Ground nodes, then driver nodes, then internal nodes
            from each group in order.
        """
        result = list(self.ground)
        result.extend(self.drivers)
        for group in self.groups:
            result.extend(group.internal_nodes)
        return result


def decompose_assur_groups(graph: LinkageGraph) -> DecompositionResult:
    """Decompose a linkage graph into Assur groups.

    This algorithm identifies:
    1. Ground nodes (fixed frame points)
    2. Driver nodes (inputs/motors)
    3. Assur groups in solvable order

    The algorithm works iteratively:
    - Start with ground and drivers as "known"
    - Find groups that can be solved (all anchors known)
    - Add solved group nodes to known set
    - Repeat until all nodes are assigned

    Args:
        graph: The linkage graph to decompose.

    Returns:
        DecompositionResult with groups in solving order.

    Raises:
        ValueError: If decomposition fails (e.g., underconstrained mechanism).

    Example:
        >>> graph = LinkageGraph(name="Four-bar")
        >>> # ... add nodes and edges ...
        >>> result = decompose_assur_groups(graph)
        >>> print(f"Found {len(result.groups)} Assur groups")
    """
    result = DecompositionResult(graph=graph)

    # Phase 1: Identify ground and drivers
    for node in graph.nodes.values():
        if node.role == NodeRole.GROUND:
            result.ground.append(node.id)
        elif node.role == NodeRole.DRIVER:
            result.drivers.append(node.id)

    # Known positions: ground + drivers
    known: set[NodeId] = set(result.ground) | set(result.drivers)

    # Remaining nodes to assign to groups
    remaining: set[NodeId] = {
        n.id for n in graph.nodes.values()
        if n.id not in known
    }

    # Phase 2: Iteratively identify Assur groups
    max_iterations = len(remaining) + 1  # Safety limit

    for _ in range(max_iterations):
        if not remaining:
            break

        found_group = False

        # Try to form a dyad from remaining nodes
        for node_id in list(remaining):
            node = graph.nodes[node_id]
            neighbors = graph.neighbors(node_id)

            # Count known vs unknown neighbors
            known_neighbors = [n for n in neighbors if n in known]

            # For RRR dyad: need exactly 2 known neighbors (revolute type)
            if len(known_neighbors) >= 2:
                group = _try_create_rrr_dyad(graph, node_id, known_neighbors)
                if group is not None:
                    result.groups.append(group)
                    known.add(node_id)
                    remaining.remove(node_id)
                    found_group = True
                    break

            # For RRP dyad: need 1 revolute + line constraint (3 known nodes)
            if len(known_neighbors) >= 3:
                rrp_group = _try_create_rrp_dyad(graph, node_id, known_neighbors)
                if rrp_group is not None:
                    result.groups.append(rrp_group)
                    known.add(node_id)
                    remaining.remove(node_id)
                    found_group = True
                    break

        if not found_group:
            # Could not find any solvable group
            raise ValueError(
                f"Cannot decompose: {len(remaining)} nodes remain unsolved. "
                f"Remaining: {remaining}. "
                f"This may indicate an underconstrained or invalid mechanism."
            )

    return result


def _try_create_rrr_dyad(
    graph: LinkageGraph,
    internal_id: NodeId,
    known_neighbors: list[NodeId],
) -> DyadRRR | None:
    """Try to create an RRR dyad from the given node and known neighbors.

    Args:
        graph: The linkage graph.
        internal_id: The candidate internal node.
        known_neighbors: List of known neighbor node IDs.

    Returns:
        A DyadRRR if successful, None otherwise.
    """
    if len(known_neighbors) < 2:
        return None

    internal_node = graph.nodes[internal_id]

    # RRR requires revolute joint
    if internal_node.joint_type != JointType.REVOLUTE:
        return None

    anchor0_id, anchor1_id = known_neighbors[0], known_neighbors[1]

    # Get edge distances
    edge0 = graph.get_edge_between(internal_id, anchor0_id)
    edge1 = graph.get_edge_between(internal_id, anchor1_id)

    if edge0 is None or edge1 is None:
        return None

    if edge0.distance is None or edge1.distance is None:
        return None

    return DyadRRR(
        internal_nodes=(internal_id,),
        anchor_nodes=(anchor0_id, anchor1_id),
        internal_edges=(edge0.id, edge1.id),
        distance0=edge0.distance,
        distance1=edge1.distance,
    )


def _try_create_rrp_dyad(
    graph: LinkageGraph,
    internal_id: NodeId,
    known_neighbors: list[NodeId],
) -> DyadRRP | None:
    """Try to create an RRP dyad from the given node and known neighbors.

    This is more complex than RRR - we need to identify which connection
    is the revolute (with distance) and which two define the line.

    Args:
        graph: The linkage graph.
        internal_id: The candidate internal node.
        known_neighbors: List of known neighbor node IDs.

    Returns:
        A DyadRRP if successful, None otherwise.
    """
    if len(known_neighbors) < 3:
        return None

    # Find an edge with a distance constraint (revolute connection)
    revolute_anchor = None
    revolute_distance = None
    revolute_edge_id = None

    for neighbor_id in known_neighbors:
        edge = graph.get_edge_between(internal_id, neighbor_id)
        if edge is not None and edge.distance is not None:
            revolute_anchor = neighbor_id
            revolute_distance = edge.distance
            revolute_edge_id = edge.id
            break

    if revolute_anchor is None:
        return None

    # The other two known neighbors define the line
    line_nodes = [n for n in known_neighbors if n != revolute_anchor]
    if len(line_nodes) < 2:
        return None

    return DyadRRP(
        internal_nodes=(internal_id,),
        anchor_nodes=(revolute_anchor,),
        internal_edges=(revolute_edge_id,) if revolute_edge_id else (),
        revolute_distance=revolute_distance,
        line_node1=line_nodes[0],
        line_node2=line_nodes[1],
    )


def validate_decomposition(result: DecompositionResult) -> list[str]:
    """Validate a decomposition result.

    Checks for common issues that might indicate an invalid decomposition.

    Args:
        result: The DecompositionResult to validate.

    Returns:
        List of warning/error messages (empty if valid).
    """
    messages: list[str] = []

    if not result.ground:
        messages.append("No ground nodes found - mechanism has no fixed frame")

    if not result.drivers:
        messages.append("No driver nodes found - mechanism has no input")

    # Check all nodes are accounted for
    if result.graph is not None:
        all_nodes = set(result.graph.nodes.keys())
        accounted: set[NodeId] = set(result.ground) | set(result.drivers)
        for group in result.groups:
            accounted.update(group.internal_nodes)

        missing = all_nodes - accounted
        if missing:
            messages.append(f"Nodes not in any group: {missing}")

        extra = accounted - all_nodes
        if extra:
            messages.append(f"Nodes in groups but not in graph: {extra}")

    return messages


def solve_decomposition(
    result: DecompositionResult,
    initial_positions: dict[NodeId, Coord] | None = None,
) -> dict[NodeId, Coord]:
    """Solve the kinematics using a decomposition result.

    This function computes the positions of all nodes by solving
    the Assur groups in order.

    Args:
        result: The decomposition result with groups in solving order.
        initial_positions: Optional initial positions for nodes.
            If not provided, positions from the graph nodes are used.

    Returns:
        Dictionary mapping all node IDs to their computed (x, y) positions.

    Raises:
        UnbuildableError: If any group cannot be solved.
        ValueError: If required positions are missing.
    """
    if result.graph is None:
        raise ValueError("DecompositionResult has no graph reference")

    graph = result.graph
    positions: dict[NodeId, Coord] = {}

    # Initialize with ground and driver positions
    for node_id in result.ground:
        node = graph.nodes[node_id]
        if initial_positions and node_id in initial_positions:
            positions[node_id] = initial_positions[node_id]
        elif node.position[0] is not None and node.position[1] is not None:
            positions[node_id] = (node.position[0], node.position[1])
        else:
            raise ValueError(f"Ground node {node_id} has no position")

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
        new_positions = group.solve(graph, positions)
        positions.update(new_positions)

    return positions
