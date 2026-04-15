"""Core graph elements for the hypergraph representation.

This module provides the fundamental data structures for representing
planar linkages as hypergraphs: Node, Edge, and Hyperedge.

These are pure topological elements - they define structure and connectivity
only. Dimensional data (positions, distances, angles) is stored separately
in the Dimensions class (see pylinkage.dimensions).
"""

from dataclasses import dataclass, field

from .._types import EdgeId, HyperedgeId, JointType, NodeId, NodeRole


@dataclass
class Node:
    """A joint in the linkage hypergraph (topology only).

    Represents the topological aspect of a joint - its identity, role,
    and type. Geometric data (position, angles) is stored separately
    in a Dimensions object.

    Attributes:
        id: Unique identifier for this node.
        role: The role in the mechanism (GROUND, DRIVER, or DRIVEN).
        joint_type: The kinematic joint type (REVOLUTE or PRISMATIC).
        name: Human-readable name for display.

    Example:
        >>> node = Node("A", role=NodeRole.GROUND)
        >>> node.joint_type
        <JointType.REVOLUTE: 1>
    """

    id: NodeId
    role: NodeRole = NodeRole.DRIVEN
    joint_type: JointType = JointType.REVOLUTE
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
    """A binary connection between two nodes (topology only).

    Edges represent rigid links connecting exactly two joints.
    For N-way rigid bodies (N > 2), use Hyperedge instead.

    Distance constraints are stored separately in a Dimensions object.

    Attributes:
        id: Unique identifier for this edge.
        source: ID of the source node.
        target: ID of the target node.

    Example:
        >>> edge = Edge("AB", source="A", target="B")
    """

    id: EdgeId
    source: NodeId
    target: NodeId

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
    """An N-way rigid body connection (topology only).

    A hyperedge connects N nodes (joints) that belong to the same
    rigid link. This is a topological grouping - distance constraints
    are stored separately in a Dimensions object.

    This is more expressive than multiple edges with a shared body_id
    because it explicitly groups all related nodes together.

    Attributes:
        id: Unique identifier for this hyperedge.
        nodes: Tuple of node IDs connected by this rigid body.
        name: Human-readable name for the rigid body.

    Example:
        >>> # Triangle rigid body connecting A, B, C
        >>> he = Hyperedge(
        ...     id="triangle",
        ...     nodes=("A", "B", "C"),
        ...     name="Triangle Link"
        ... )
    """

    id: HyperedgeId
    nodes: tuple[NodeId, ...] = field(default_factory=tuple)
    name: str | None = None

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

        Creates one Edge for each pair of adjacent nodes in this hyperedge.
        For a hyperedge with N nodes, creates N-1 edges forming a chain.

        Args:
            prefix: Optional prefix for generated edge IDs.

        Returns:
            List of Edge objects representing pairwise connections.

        Example:
            >>> he = Hyperedge("tri", ("A", "B", "C"))
            >>> edges = he.to_edges()
            >>> len(edges)
            2
        """
        edges = []
        nodes_list = list(self.nodes)
        for i in range(len(nodes_list) - 1):
            n1, n2 = nodes_list[i], nodes_list[i + 1]
            edge_id = f"{prefix}{self.id}_{n1}_{n2}" if prefix else f"{self.id}_{n1}_{n2}"
            edges.append(
                Edge(
                    id=edge_id,
                    source=n1,
                    target=n2,
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
            ...     Edge("e1", "A", "B"),
            ...     Edge("e2", "B", "C"),
            ... ]
            >>> he = Hyperedge.from_edges(edges, "combined")
            >>> "A" in he.nodes
            True
        """
        if not edges:
            raise ValueError("Cannot create hyperedge from empty edge list")

        nodes_set: set[NodeId] = set()

        for edge in edges:
            nodes_set.add(edge.source)
            nodes_set.add(edge.target)

        return cls(
            id=hyperedge_id,
            nodes=tuple(sorted(nodes_set)),
            name=name,
        )
