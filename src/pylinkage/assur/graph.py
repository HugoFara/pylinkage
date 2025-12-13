"""Graph-based representation of planar linkages.

This module provides data structures for representing linkages as graphs
where nodes are joints and edges are rigid links. This representation
enables Assur group decomposition and provides an alternative syntax
for defining linkages.
"""


import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._types import EdgeId, JointType, NodeId, NodeRole

if TYPE_CHECKING:
    pass

# Type alias for positions that may have None coordinates
_MaybeCoord = tuple[float | None, float | None]


@dataclass
class Node:
    """A joint in the linkage graph.

    Represents both the topological (connectivity) and geometric
    (position, constraints) aspects of a joint.

    Attributes:
        id: Unique identifier for this node.
        joint_type: The kinematic joint type (REVOLUTE or PRISMATIC).
        role: The role in the mechanism (GROUND, DRIVER, or DRIVEN).
        position: Current (x, y) coordinates, may contain None if uninitialized.
        angle: For DRIVER nodes, the rotation angle per step (radians).
        initial_angle: For DRIVER nodes, the starting angle (radians).
        name: Human-readable name for display.

    Example:
        >>> node = Node("A", role=NodeRole.GROUND, position=(0, 0))
        >>> node.joint_type
        <JointType.REVOLUTE: 1>
    """

    id: NodeId
    joint_type: JointType = JointType.REVOLUTE
    role: NodeRole = NodeRole.DRIVEN
    position: _MaybeCoord = (None, None)
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
    """A rigid link between two joints.

    Edges represent rigid bodies connecting joints. The constraint
    (typically distance) depends on the joint types at each end.

    Attributes:
        id: Unique identifier for this edge.
        source: ID of the source node.
        target: ID of the target node.
        distance: The distance constraint between the nodes (for R-R or R-P).
        body_id: Optional tag for grouping edges that belong to the same rigid body.

    Example:
        >>> edge = Edge("AB", source="A", target="B", distance=1.0)
    """

    id: EdgeId
    source: NodeId
    target: NodeId
    distance: float | None = None
    body_id: str | None = None

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
class LinkageGraph:
    """Graph representation of a planar linkage.

    A linkage is represented as a graph where:
    - Nodes are joints (revolute R or prismatic P)
    - Edges are rigid links with distance constraints

    This representation enables:
    - Assur group decomposition
    - Alternative linkage definition syntax
    - Graph algorithms for analysis

    Attributes:
        nodes: Dictionary mapping node IDs to Node objects.
        edges: Dictionary mapping edge IDs to Edge objects.
        name: Human-readable name for the linkage.

    Example:
        >>> graph = LinkageGraph(name="Four-bar")
        >>> graph.add_node(Node("A", role=NodeRole.GROUND, position=(0, 0)))
        >>> graph.add_node(Node("B", role=NodeRole.DRIVER, position=(0, 1), angle=0.31))
        >>> graph.add_edge(Edge("AB", source="A", target="B", distance=1.0))
        >>> graph.neighbors("A")
        ['B']
    """

    nodes: dict[NodeId, Node] = field(default_factory=dict)
    edges: dict[EdgeId, Edge] = field(default_factory=dict)
    name: str = ""

    # Adjacency structure (computed lazily)
    _adjacency: dict[NodeId, list[EdgeId]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Args:
            node: The Node to add.

        Raises:
            ValueError: If a node with the same ID already exists.
        """
        if node.id in self.nodes:
            raise ValueError(f"Node with id '{node.id}' already exists")
        self.nodes[node.id] = node
        if node.id not in self._adjacency:
            self._adjacency[node.id] = []

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph.

        Args:
            edge: The Edge to add.

        Raises:
            ValueError: If an edge with the same ID already exists,
                or if source/target nodes don't exist.
        """
        if edge.id in self.edges:
            raise ValueError(f"Edge with id '{edge.id}' already exists")
        if edge.source not in self.nodes:
            raise ValueError(f"Source node '{edge.source}' not found")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node '{edge.target}' not found")

        self.edges[edge.id] = edge
        self._adjacency.setdefault(edge.source, []).append(edge.id)
        self._adjacency.setdefault(edge.target, []).append(edge.id)

    def remove_node(self, node_id: NodeId) -> Node:
        """Remove a node and all its edges from the graph.

        Args:
            node_id: ID of the node to remove.

        Returns:
            The removed Node.

        Raises:
            KeyError: If node not found.
        """
        # Remove all edges connected to this node
        edges_to_remove = list(self._adjacency.get(node_id, []))
        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)

        # Remove from adjacency
        self._adjacency.pop(node_id, None)

        # Remove and return the node
        return self.nodes.pop(node_id)

    def remove_edge(self, edge_id: EdgeId) -> Edge:
        """Remove an edge from the graph.

        Args:
            edge_id: ID of the edge to remove.

        Returns:
            The removed Edge.

        Raises:
            KeyError: If edge not found.
        """
        edge = self.edges.pop(edge_id)

        # Update adjacency lists
        if edge.source in self._adjacency:
            self._adjacency[edge.source] = [
                e for e in self._adjacency[edge.source] if e != edge_id
            ]
        if edge.target in self._adjacency:
            self._adjacency[edge.target] = [
                e for e in self._adjacency[edge.target] if e != edge_id
            ]

        return edge

    def neighbors(self, node_id: NodeId) -> list[NodeId]:
        """Get all nodes connected to the given node.

        Args:
            node_id: The node to find neighbors of.

        Returns:
            List of node IDs connected to the given node.
        """
        result = []
        for edge_id in self._adjacency.get(node_id, []):
            edge = self.edges[edge_id]
            other = edge.other_node(node_id)
            result.append(other)
        return result

    def get_edge_between(self, node1: NodeId, node2: NodeId) -> Edge | None:
        """Get the edge connecting two nodes, if any.

        Args:
            node1: First node ID.
            node2: Second node ID.

        Returns:
            The Edge connecting the nodes, or None if not connected.
        """
        for edge_id in self._adjacency.get(node1, []):
            edge = self.edges[edge_id]
            if edge.connects(node2):
                return edge
        return None

    def get_edges_for_node(self, node_id: NodeId) -> list[Edge]:
        """Get all edges connected to a node.

        Args:
            node_id: The node ID.

        Returns:
            List of Edge objects connected to the node.
        """
        return [self.edges[eid] for eid in self._adjacency.get(node_id, [])]

    def ground_nodes(self) -> list[Node]:
        """Get all ground/frame nodes."""
        return [n for n in self.nodes.values() if n.role == NodeRole.GROUND]

    def driver_nodes(self) -> list[Node]:
        """Get all driver/input nodes."""
        return [n for n in self.nodes.values() if n.role == NodeRole.DRIVER]

    def driven_nodes(self) -> list[Node]:
        """Get all driven nodes (Assur group members)."""
        return [n for n in self.nodes.values() if n.role == NodeRole.DRIVEN]

    def degree(self, node_id: NodeId) -> int:
        """Get the degree (number of connections) of a node.

        Args:
            node_id: The node ID.

        Returns:
            Number of edges connected to the node.
        """
        return len(self._adjacency.get(node_id, []))

    def copy(self) -> "LinkageGraph":
        """Create a deep copy of the graph.

        Returns:
            A new LinkageGraph with copied nodes and edges.
        """
        return copy.deepcopy(self)

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.nodes)

    def __contains__(self, item: NodeId | Node) -> bool:
        """Check if a node (by ID or object) is in the graph."""
        if isinstance(item, Node):
            return item.id in self.nodes
        return item in self.nodes
