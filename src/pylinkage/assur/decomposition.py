"""Assur group decomposition algorithm (topology only).

This module provides algorithms to decompose a linkage graph into
Assur groups, which enables systematic kinematic analysis.

The decomposition algorithm:
1. Identifies ground (frame) nodes
2. Identifies driver (input) nodes
3. Iteratively finds solvable Assur groups where all anchors are known
4. Returns groups in solving order

IMPORTANT: This module provides structural/topological decomposition only.
Assur groups are pure topology. For solving kinematics, use
pylinkage.solver.solve.solve_decomposition() with a Dimensions object.
"""

import warnings
from dataclasses import dataclass, field

from .._types import Coord
from ._types import JointType, NodeId, NodeRole
from .graph import LinkageGraph
from .groups import AssurGroup, Dyad, Triad


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

        # Try to form a triad from pairs of remaining nodes
        if not found_group:
            found_group = _try_find_triad(
                graph, remaining, known, result,
            )

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
) -> Dyad | None:
    """Try to create an RRR dyad from the given node and known neighbors.

    Args:
        graph: The linkage graph (topology only).
        internal_id: The candidate internal node.
        known_neighbors: List of known neighbor node IDs.

    Returns:
        A Dyad with signature "RRR" if successful, None otherwise.
    """
    if len(known_neighbors) < 2:
        return None

    internal_node = graph.nodes[internal_id]

    # RRR requires revolute joint
    if internal_node.joint_type != JointType.REVOLUTE:
        return None

    anchor0_id, anchor1_id = known_neighbors[0], known_neighbors[1]

    # Get edges (topology only, no distance check)
    edge0 = graph.get_edge_between(internal_id, anchor0_id)
    edge1 = graph.get_edge_between(internal_id, anchor1_id)

    if edge0 is None or edge1 is None:
        return None

    return Dyad(
        _signature="RRR",
        internal_nodes=(internal_id,),
        anchor_nodes=(anchor0_id, anchor1_id),
        internal_edges=(edge0.id, edge1.id),
    )


def _try_create_rrp_dyad(
    graph: LinkageGraph,
    internal_id: NodeId,
    known_neighbors: list[NodeId],
) -> Dyad | None:
    """Try to create an RRP dyad from the given node and known neighbors.

    This is more complex than RRR - we need to identify which connection
    is the revolute anchor and which two define the line.

    Args:
        graph: The linkage graph (topology only).
        internal_id: The candidate internal node.
        known_neighbors: List of known neighbor node IDs.

    Returns:
        A Dyad with signature "RRP" if successful, None otherwise.
    """
    if len(known_neighbors) < 3:
        return None

    # Find the first edge to use as revolute anchor
    revolute_anchor = None
    revolute_edge_id = None

    for neighbor_id in known_neighbors:
        edge = graph.get_edge_between(internal_id, neighbor_id)
        if edge is not None:
            revolute_anchor = neighbor_id
            revolute_edge_id = edge.id
            break

    if revolute_anchor is None:
        return None

    # The other two known neighbors define the line
    line_nodes = [n for n in known_neighbors if n != revolute_anchor]
    if len(line_nodes) < 2:
        return None

    return Dyad(
        _signature="RRP",
        internal_nodes=(internal_id,),
        anchor_nodes=(revolute_anchor,),
        internal_edges=(revolute_edge_id,) if revolute_edge_id else (),
        line_node1=line_nodes[0],
        line_node2=line_nodes[1],
    )


def _try_find_triad(
    graph: LinkageGraph,
    remaining: set[NodeId],
    known: set[NodeId],
    result: DecompositionResult,
) -> bool:
    """Try to find a triad among remaining nodes.

    A triad has 2 internal nodes and 3 anchor nodes (all known).
    We search all pairs of remaining nodes and check if they form
    a valid triad with the known set.

    Args:
        graph: The linkage graph.
        remaining: Set of unsolved node IDs (modified in place on success).
        known: Set of solved node IDs (modified in place on success).
        result: DecompositionResult to append to (modified in place).

    Returns:
        True if a triad was found and added, False otherwise.
    """
    remaining_list = list(remaining)
    for i, node_a in enumerate(remaining_list):
        for node_b in remaining_list[i + 1:]:
            triad = _try_create_triad(graph, node_a, node_b, known)
            if triad is not None:
                result.groups.append(triad)
                known.add(node_a)
                known.add(node_b)
                remaining.discard(node_a)
                remaining.discard(node_b)
                return True
    return False


def _try_create_triad(
    graph: LinkageGraph,
    internal_a: NodeId,
    internal_b: NodeId,
    known: set[NodeId],
) -> Triad | None:
    """Try to create a triad from two internal nodes.

    A valid triad requires:
    - The two internal nodes are connected by an edge (or share anchors)
    - Together they have at least 3 distinct known anchor neighbors
    - Total of 4 edges connecting internals to anchors + each other

    Args:
        graph: The linkage graph.
        internal_a: First candidate internal node.
        internal_b: Second candidate internal node.
        known: Set of currently known node IDs.

    Returns:
        A Triad instance if valid, None otherwise.
    """
    neighbors_a = set(graph.neighbors(internal_a))
    neighbors_b = set(graph.neighbors(internal_b))

    # Known anchors reachable from each internal node
    known_anchors_a = neighbors_a & known
    known_anchors_b = neighbors_b & known

    # All distinct known anchors
    all_known_anchors = known_anchors_a | known_anchors_b

    # A triad needs at least 3 known anchor nodes
    if len(all_known_anchors) < 3:
        return None

    # Collect edges: internal↔anchor and internal↔internal
    # edge_map tracks which node pair each edge connects (for the solver)
    edge_ids: list[str] = []
    edge_map: dict[str, tuple[NodeId, NodeId]] = {}
    anchor_ids: list[NodeId] = []

    # Edges from internal_a to known anchors
    for anchor_id in known_anchors_a:
        edge = graph.get_edge_between(internal_a, anchor_id)
        if edge is not None:
            edge_ids.append(edge.id)
            edge_map[edge.id] = (internal_a, anchor_id)
            if anchor_id not in anchor_ids:
                anchor_ids.append(anchor_id)

    # Edges from internal_b to known anchors
    for anchor_id in known_anchors_b:
        edge = graph.get_edge_between(internal_b, anchor_id)
        if edge is not None:
            edge_ids.append(edge.id)
            edge_map[edge.id] = (internal_b, anchor_id)
            if anchor_id not in anchor_ids:
                anchor_ids.append(anchor_id)

    # Edge between the two internals (if any)
    internal_edge = graph.get_edge_between(internal_a, internal_b)
    if internal_edge is not None:
        edge_ids.append(internal_edge.id)
        edge_map[internal_edge.id] = (internal_a, internal_b)

    # Need at least 4 edges total for a valid triad
    if len(edge_ids) < 4:
        return None

    # Need at least 3 anchor nodes
    if len(anchor_ids) < 3:
        return None

    # Build the signature from joint types
    # Convention: anchors first (3), then internals (2), then inter-internal
    sig_chars = []
    joint_char = {JointType.REVOLUTE: "R", JointType.PRISMATIC: "P"}
    for aid in anchor_ids[:3]:
        node = graph.nodes[aid]
        sig_chars.append(joint_char.get(node.joint_type, "R"))
    for nid in (internal_a, internal_b):
        node = graph.nodes[nid]
        sig_chars.append(joint_char.get(node.joint_type, "R"))
    # 6th character: constraint type between internals
    if internal_edge is not None:
        # Edge exists → revolute connection between them
        sig_chars.append("R")
    else:
        # No direct edge ��� infer from shared anchor pattern
        sig_chars.append("R")
    signature = "".join(sig_chars)

    return Triad(
        _signature=signature,
        internal_nodes=(internal_a, internal_b),
        anchor_nodes=tuple(anchor_ids[:3]),
        internal_edges=tuple(edge_ids),
        edge_map=edge_map,
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


