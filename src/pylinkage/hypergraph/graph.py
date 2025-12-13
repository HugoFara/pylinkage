"""Hypergraph-based representation of planar linkages.

This module provides the HypergraphLinkage class, which represents linkages
as hypergraphs where nodes are joints and connections can be either binary
edges or N-way hyperedges (rigid bodies).
"""


import copy
from dataclasses import dataclass, field

from ._types import EdgeId, HyperedgeId, NodeId, NodeRole
from .core import Edge, Hyperedge, Node


@dataclass
class HypergraphLinkage:
    """Abstract hypergraph representation of a planar linkage.

    A linkage is represented as a hypergraph where:
    - Nodes are joints (revolute R or prismatic P)
    - Edges are binary links with distance constraints
    - Hyperedges are N-way rigid bodies with pairwise constraints

    This representation enables:
    - First-class rigid body representation via hyperedges
    - Hierarchical composition via components
    - Conversion to other representations (Assur, joint-based)

    Attributes:
        nodes: Dictionary mapping node IDs to Node objects.
        edges: Dictionary mapping edge IDs to Edge objects.
        hyperedges: Dictionary mapping hyperedge IDs to Hyperedge objects.
        name: Human-readable name for the linkage.

    Example:
        >>> graph = HypergraphLinkage(name="Four-bar")
        >>> graph.add_node(Node("A", role=NodeRole.GROUND, position=(0, 0)))
        >>> graph.add_node(Node("B", role=NodeRole.DRIVER, position=(0, 1), angle=0.31))
        >>> graph.add_edge(Edge("AB", source="A", target="B", distance=1.0))
    """

    nodes: dict[NodeId, Node] = field(default_factory=dict)
    edges: dict[EdgeId, Edge] = field(default_factory=dict)
    hyperedges: dict[HyperedgeId, Hyperedge] = field(default_factory=dict)
    name: str = ""

    # Adjacency structure (computed lazily)
    _adjacency: dict[NodeId, list[EdgeId]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _hyperedge_membership: dict[NodeId, list[HyperedgeId]] = field(
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
        if node.id not in self._hyperedge_membership:
            self._hyperedge_membership[node.id] = []

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

    def add_hyperedge(self, hyperedge: Hyperedge) -> None:
        """Add a hyperedge to the graph.

        Args:
            hyperedge: The Hyperedge to add.

        Raises:
            ValueError: If a hyperedge with the same ID already exists,
                or if any referenced nodes don't exist.
        """
        if hyperedge.id in self.hyperedges:
            raise ValueError(f"Hyperedge with id '{hyperedge.id}' already exists")

        for node_id in hyperedge.nodes:
            if node_id not in self.nodes:
                raise ValueError(f"Node '{node_id}' not found for hyperedge")

        self.hyperedges[hyperedge.id] = hyperedge
        for node_id in hyperedge.nodes:
            self._hyperedge_membership.setdefault(node_id, []).append(hyperedge.id)

    def remove_node(self, node_id: NodeId) -> Node:
        """Remove a node and all its connections from the graph.

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

        # Remove all hyperedges containing this node
        hyperedges_to_remove = list(self._hyperedge_membership.get(node_id, []))
        for he_id in hyperedges_to_remove:
            self.remove_hyperedge(he_id)

        # Remove from adjacency structures
        self._adjacency.pop(node_id, None)
        self._hyperedge_membership.pop(node_id, None)

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

    def remove_hyperedge(self, hyperedge_id: HyperedgeId) -> Hyperedge:
        """Remove a hyperedge from the graph.

        Args:
            hyperedge_id: ID of the hyperedge to remove.

        Returns:
            The removed Hyperedge.

        Raises:
            KeyError: If hyperedge not found.
        """
        hyperedge = self.hyperedges.pop(hyperedge_id)

        # Update membership lists
        for node_id in hyperedge.nodes:
            if node_id in self._hyperedge_membership:
                self._hyperedge_membership[node_id] = [
                    he for he in self._hyperedge_membership[node_id] if he != hyperedge_id
                ]

        return hyperedge

    def neighbors(self, node_id: NodeId) -> list[NodeId]:
        """Get all nodes connected to the given node via edges.

        Args:
            node_id: The node to find neighbors of.

        Returns:
            List of node IDs connected via edges.
        """
        result = []
        for edge_id in self._adjacency.get(node_id, []):
            edge = self.edges[edge_id]
            other = edge.other_node(node_id)
            result.append(other)
        return result

    def hyperedge_neighbors(self, node_id: NodeId) -> list[NodeId]:
        """Get all nodes sharing a hyperedge with the given node.

        Args:
            node_id: The node to find hyperedge neighbors of.

        Returns:
            List of node IDs sharing hyperedges with the given node.
        """
        result_set: set[NodeId] = set()
        for he_id in self._hyperedge_membership.get(node_id, []):
            hyperedge = self.hyperedges[he_id]
            for n in hyperedge.nodes:
                if n != node_id:
                    result_set.add(n)
        return list(result_set)

    def all_neighbors(self, node_id: NodeId) -> list[NodeId]:
        """Get all nodes connected to the given node via edges or hyperedges.

        Args:
            node_id: The node to find neighbors of.

        Returns:
            List of all connected node IDs (deduplicated).
        """
        result_set = set(self.neighbors(node_id))
        result_set.update(self.hyperedge_neighbors(node_id))
        return list(result_set)

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

    def get_hyperedges_for_node(self, node_id: NodeId) -> list[Hyperedge]:
        """Get all hyperedges containing a node.

        Args:
            node_id: The node ID.

        Returns:
            List of Hyperedge objects containing the node.
        """
        return [
            self.hyperedges[hid] for hid in self._hyperedge_membership.get(node_id, [])
        ]

    def ground_nodes(self) -> list[Node]:
        """Get all ground/frame nodes."""
        return [n for n in self.nodes.values() if n.role == NodeRole.GROUND]

    def driver_nodes(self) -> list[Node]:
        """Get all driver/input nodes."""
        return [n for n in self.nodes.values() if n.role == NodeRole.DRIVER]

    def driven_nodes(self) -> list[Node]:
        """Get all driven nodes."""
        return [n for n in self.nodes.values() if n.role == NodeRole.DRIVEN]

    def degree(self, node_id: NodeId) -> int:
        """Get the edge degree (number of edge connections) of a node.

        Args:
            node_id: The node ID.

        Returns:
            Number of edges connected to the node.
        """
        return len(self._adjacency.get(node_id, []))

    def hyperdegree(self, node_id: NodeId) -> int:
        """Get the hyperedge degree (number of hyperedge memberships) of a node.

        Args:
            node_id: The node ID.

        Returns:
            Number of hyperedges containing the node.
        """
        return len(self._hyperedge_membership.get(node_id, []))

    def to_simple_graph(self) -> "HypergraphLinkage":
        """Convert to a simple graph by expanding hyperedges to edges.

        Creates a new HypergraphLinkage with no hyperedges, where each
        hyperedge has been converted to its equivalent edges.

        Returns:
            A new HypergraphLinkage with only nodes and edges.
        """
        result = HypergraphLinkage(name=self.name)

        # Copy all nodes
        for node in self.nodes.values():
            result.add_node(
                Node(
                    id=node.id,
                    position=node.position,
                    role=node.role,
                    joint_type=node.joint_type,
                    angle=node.angle,
                    initial_angle=node.initial_angle,
                    name=node.name,
                )
            )

        # Copy all existing edges
        for edge in self.edges.values():
            result.add_edge(
                Edge(
                    id=edge.id,
                    source=edge.source,
                    target=edge.target,
                    distance=edge.distance,
                )
            )

        # Convert hyperedges to edges
        for hyperedge in self.hyperedges.values():
            for new_edge in hyperedge.to_edges():
                # Avoid duplicate edges
                if new_edge.id not in result.edges:
                    result.add_edge(new_edge)

        return result

    def copy(self) -> "HypergraphLinkage":
        """Create a deep copy of the graph.

        Returns:
            A new HypergraphLinkage with copied nodes, edges, and hyperedges.
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
