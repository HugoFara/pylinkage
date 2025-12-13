"""Core graph elements for the hypergraph representation.

This module provides the fundamental data structures for representing
planar linkages as hypergraphs: Node, Edge, and Hyperedge.
"""


from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._types import EdgeId, HyperedgeId, JointType, NodeId, NodeRole

if TYPE_CHECKING:
    from .._types import MaybeCoord


@dataclass
class Node:
    """A joint in the linkage hypergraph.

    Represents both the topological (connectivity) and geometric
    (position) aspects of a joint.

    Attributes:
        id: Unique identifier for this node.
        position: Current (x, y) coordinates, may contain None if uninitialized.
        role: The role in the mechanism (GROUND, DRIVER, or DRIVEN).
        joint_type: The kinematic joint type (REVOLUTE or PRISMATIC).
        angle: For DRIVER nodes, the rotation angle per step (radians).
        initial_angle: For DRIVER nodes, the starting angle (radians).
        name: Human-readable name for display.

    Example:
        >>> node = Node("A", position=(0.0, 0.0), role=NodeRole.GROUND)
        >>> node.joint_type
        <JointType.REVOLUTE: 1>
    """

    id: NodeId
    position: "MaybeCoord" = (None, None)
    role: NodeRole = NodeRole.DRIVEN
    joint_type: JointType = JointType.REVOLUTE
    angle: float | None = None
    initial_angle: float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        """Set default name to id if not provided."""
        if self.name is None:
            self.name = self.id

    def __hash__(self) -> int:
        """Hash by id for use in sets and dict keys."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality by id."""
        if isinstance(other, Node):
            return self.id == other.id
        return False


@dataclass
class Edge:
    """A binary connection between two nodes.

    Edges represent rigid links connecting exactly two joints.
    For N-way rigid bodies (N > 2), use Hyperedge instead.

    Attributes:
        id: Unique identifier for this edge.
        source: ID of the source node.
        target: ID of the target node.
        distance: The distance constraint between the nodes.

    Example:
        >>> edge = Edge("AB", source="A", target="B", distance=1.0)
    """

    id: EdgeId
    source: NodeId
    target: NodeId
    distance: float | None = None

    def __hash__(self) -> int:
        """Hash by id for use in sets and dict keys."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality by id."""
        if isinstance(other, Edge):
            return self.id == other.id
        return False

    def connects(self, node_id: NodeId) -> bool:
        """Check if this edge connects to the given node."""
        return self.source == node_id or self.target == node_id

    def other_node(self, node_id: NodeId) -> NodeId:
        """Get the node on the other end of this edge.

        Args:
            node_id: One endpoint of this edge.

        Returns:
            The ID of the other endpoint.

        Raises:
            ValueError: If node_id is not an endpoint of this edge.
        """
        if self.source == node_id:
            return self.target
        if self.target == node_id:
            return self.source
        raise ValueError(f"Node {node_id} is not connected by edge {self.id}")


@dataclass
class Hyperedge:
    """An N-way rigid body connection.

    A hyperedge connects N nodes (joints) that belong to the same
    rigid link. It stores pairwise distance constraints between
    these nodes, representing a fully constrained rigid body.

    This is more expressive than multiple edges with a shared body_id
    because it explicitly groups all related constraints together.

    Attributes:
        id: Unique identifier for this hyperedge.
        nodes: Tuple of node IDs connected by this rigid body.
        constraints: Dict mapping node pairs (as sorted tuples) to distances.
        name: Human-readable name for the rigid body.

    Example:
        >>> # Triangle rigid body connecting A, B, C
        >>> he = Hyperedge(
        ...     id="triangle",
        ...     nodes=("A", "B", "C"),
        ...     constraints={
        ...         ("A", "B"): 1.0,
        ...         ("B", "C"): 1.5,
        ...         ("A", "C"): 2.0,
        ...     },
        ...     name="Triangle Link"
        ... )
    """

    id: HyperedgeId
    nodes: tuple[NodeId, ...] = field(default_factory=tuple)
    constraints: dict[tuple[NodeId, NodeId], float] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize constraints."""
        # Ensure all constraint keys are sorted tuples for consistency
        normalized: dict[tuple[NodeId, NodeId], float] = {}
        for (n1, n2), dist in self.constraints.items():
            key = (min(n1, n2), max(n1, n2))
            if key in normalized:
                raise ValueError(f"Duplicate constraint for {key}")
            normalized[key] = dist
        object.__setattr__(self, "constraints", normalized)

        # Validate all nodes in constraints are in nodes tuple
        for n1, n2 in self.constraints:
            if n1 not in self.nodes or n2 not in self.nodes:
                raise ValueError(
                    f"Constraint nodes ({n1}, {n2}) not in hyperedge nodes {self.nodes}"
                )

    def __hash__(self) -> int:
        """Hash by id for use in sets and dict keys."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality by id."""
        if isinstance(other, Hyperedge):
            return self.id == other.id
        return False

    def to_edges(self, prefix: str = "") -> list[Edge]:
        """Convert hyperedge to regular binary edges.

        Creates one Edge for each pairwise constraint in this hyperedge.

        Args:
            prefix: Optional prefix for generated edge IDs.

        Returns:
            List of Edge objects representing all pairwise constraints.

        Example:
            >>> he = Hyperedge("tri", ("A", "B", "C"), {("A", "B"): 1.0, ("B", "C"): 2.0})
            >>> edges = he.to_edges()
            >>> len(edges)
            2
        """
        edges = []
        for (n1, n2), dist in self.constraints.items():
            edge_id = f"{prefix}{self.id}_{n1}_{n2}" if prefix else f"{self.id}_{n1}_{n2}"
            edges.append(
                Edge(
                    id=edge_id,
                    source=n1,
                    target=n2,
                    distance=dist,
                )
            )
        return edges

    @classmethod
    def from_edges(
        cls,
        edges: list[Edge],
        hyperedge_id: HyperedgeId,
        name: str | None = None,
    ) -> "Hyperedge":
        """Create a hyperedge from a collection of edges.

        Assumes all edges belong to the same rigid body.

        Args:
            edges: List of edges to combine.
            hyperedge_id: ID for the new hyperedge.
            name: Optional name for the hyperedge.

        Returns:
            A new Hyperedge combining all the edges.

        Raises:
            ValueError: If edges list is empty.

        Example:
            >>> edges = [
            ...     Edge("e1", "A", "B", 1.0),
            ...     Edge("e2", "B", "C", 2.0),
            ... ]
            >>> he = Hyperedge.from_edges(edges, "combined")
            >>> "A" in he.nodes
            True
        """
        if not edges:
            raise ValueError("Cannot create hyperedge from empty edge list")

        nodes_set: set[NodeId] = set()
        constraints: dict[tuple[NodeId, NodeId], float] = {}

        for edge in edges:
            nodes_set.add(edge.source)
            nodes_set.add(edge.target)
            if edge.distance is not None:
                constraints[(edge.source, edge.target)] = edge.distance

        return cls(
            id=hyperedge_id,
            nodes=tuple(sorted(nodes_set)),
            constraints=constraints,
            name=name,
        )

    def get_distance(self, node1: NodeId, node2: NodeId) -> float | None:
        """Get the distance constraint between two nodes.

        Args:
            node1: First node ID.
            node2: Second node ID.

        Returns:
            The distance constraint, or None if not specified.
        """
        key = (min(node1, node2), max(node1, node2))
        return self.constraints.get(key)
