"""Conversion between hypergraph and other representations.

This module provides functions to convert between the hypergraph
representation and other formats:
- Assur graph (LinkageGraph)
- Joint-based Linkage

This enables the hypergraph to serve as an abstract mathematical
foundation that can be converted to specific representations for
different purposes (analysis, simulation, etc.).
"""


from typing import TYPE_CHECKING

from ._types import JointType, NodeRole
from .core import Edge, Hyperedge, Node
from .graph import HypergraphLinkage

if TYPE_CHECKING:
    from ..assur.graph import LinkageGraph as AssurLinkageGraph
    from ..linkage.linkage import Linkage


def to_assur_graph(hypergraph: HypergraphLinkage) -> "AssurLinkageGraph":
    """Convert a HypergraphLinkage to an Assur LinkageGraph.

    This converts the hypergraph to the simpler graph representation
    used by the Assur decomposition system. Hyperedges are expanded
    to regular edges.

    Args:
        hypergraph: The HypergraphLinkage to convert.

    Returns:
        An Assur LinkageGraph suitable for decomposition and analysis.

    Example:
        >>> hg = HypergraphLinkage(name="Four-bar")
        >>> # ... add nodes and edges ...
        >>> assur_graph = to_assur_graph(hg)
    """
    from ..assur._types import JointType as AssurJointType
    from ..assur._types import NodeRole as AssurNodeRole
    from ..assur.graph import Edge as AssurEdge
    from ..assur.graph import LinkageGraph as AssurLinkageGraph
    from ..assur.graph import Node as AssurNode

    # First convert to simple graph (expand hyperedges)
    simple = hypergraph.to_simple_graph()

    assur_graph = AssurLinkageGraph(name=simple.name)

    # Map our types to Assur types
    role_map = {
        NodeRole.GROUND: AssurNodeRole.GROUND,
        NodeRole.DRIVER: AssurNodeRole.DRIVER,
        NodeRole.DRIVEN: AssurNodeRole.DRIVEN,
    }
    joint_type_map = {
        JointType.REVOLUTE: AssurJointType.REVOLUTE,
        JointType.PRISMATIC: AssurJointType.PRISMATIC,
    }

    # Convert nodes
    for node in simple.nodes.values():
        assur_node = AssurNode(
            id=node.id,
            joint_type=joint_type_map[node.joint_type],
            role=role_map[node.role],
            position=node.position,
            angle=node.angle,
            initial_angle=node.initial_angle,
            name=node.name,
        )
        assur_graph.add_node(assur_node)

    # Convert edges
    for edge in simple.edges.values():
        assur_edge = AssurEdge(
            id=edge.id,
            source=edge.source,
            target=edge.target,
            distance=edge.distance,
        )
        assur_graph.add_edge(assur_edge)

    return assur_graph


def from_assur_graph(assur_graph: "AssurLinkageGraph") -> HypergraphLinkage:
    """Convert an Assur LinkageGraph to a HypergraphLinkage.

    This converts from the Assur graph representation to the more
    abstract hypergraph representation. Edges with the same body_id
    are grouped into hyperedges.

    Args:
        assur_graph: The Assur LinkageGraph to convert.

    Returns:
        A HypergraphLinkage representation.

    Example:
        >>> assur_graph = LinkageGraph(name="Four-bar")
        >>> # ... add nodes and edges ...
        >>> hg = from_assur_graph(assur_graph)
    """
    from ..assur._types import JointType as AssurJointType
    from ..assur._types import NodeRole as AssurNodeRole

    hypergraph = HypergraphLinkage(name=assur_graph.name)

    # Map Assur types to our types
    role_map = {
        AssurNodeRole.GROUND: NodeRole.GROUND,
        AssurNodeRole.DRIVER: NodeRole.DRIVER,
        AssurNodeRole.DRIVEN: NodeRole.DRIVEN,
    }
    joint_type_map = {
        AssurJointType.REVOLUTE: JointType.REVOLUTE,
        AssurJointType.PRISMATIC: JointType.PRISMATIC,
    }

    # Convert nodes
    for node in assur_graph.nodes.values():
        hyper_node = Node(
            id=node.id,
            position=node.position,
            role=role_map[node.role],
            joint_type=joint_type_map[node.joint_type],
            angle=node.angle,
            initial_angle=node.initial_angle,
            name=node.name,
        )
        hypergraph.add_node(hyper_node)

    # Group edges by body_id to potentially create hyperedges
    body_edges: dict[str | None, list[tuple[str, str, str, float | None]]] = {}
    for edge in assur_graph.edges.values():
        body_id = getattr(edge, "body_id", None)
        if body_id not in body_edges:
            body_edges[body_id] = []
        body_edges[body_id].append((edge.id, edge.source, edge.target, edge.distance))

    # Convert edges - edges with body_id become hyperedges, others stay as edges
    for body_id, edges in body_edges.items():
        if body_id is not None and len(edges) > 1:
            # Create hyperedge from grouped edges
            nodes_set: set[str] = set()
            constraints: dict[tuple[str, str], float] = {}
            for _, source, target, distance in edges:
                nodes_set.add(source)
                nodes_set.add(target)
                if distance is not None:
                    constraints[(source, target)] = distance

            hyperedge = Hyperedge(
                id=body_id,
                nodes=tuple(sorted(nodes_set)),
                constraints=constraints,
                name=body_id,
            )
            hypergraph.add_hyperedge(hyperedge)
        else:
            # Keep as regular edges
            for edge_id, source, target, distance in edges:
                hyper_edge = Edge(
                    id=edge_id,
                    source=source,
                    target=target,
                    distance=distance,
                )
                hypergraph.add_edge(hyper_edge)

    return hypergraph


def to_linkage(hypergraph: HypergraphLinkage) -> "Linkage":
    """Convert a HypergraphLinkage to a joint-based Linkage.

    This converts the hypergraph to a Linkage that can be used for
    simulation. The conversion goes through the Assur representation
    to leverage the existing decomposition and conversion logic.

    Args:
        hypergraph: The HypergraphLinkage to convert.

    Returns:
        A Linkage instance ready for simulation.

    Example:
        >>> hg = HypergraphLinkage(name="Four-bar")
        >>> # ... add nodes, edges, hyperedges ...
        >>> linkage = to_linkage(hg)
        >>> for coords in linkage.step():
        ...     print(coords)
    """
    from ..assur.conversion import graph_to_linkage

    # Convert to Assur graph first
    assur_graph = to_assur_graph(hypergraph)

    # Then use existing conversion
    return graph_to_linkage(assur_graph)


def from_linkage(linkage: "Linkage") -> HypergraphLinkage:
    """Convert a joint-based Linkage to a HypergraphLinkage.

    This converts an existing Linkage to the hypergraph representation
    for analysis or manipulation. The conversion goes through the
    Assur representation to leverage existing conversion logic.

    Args:
        linkage: The Linkage to convert.

    Returns:
        A HypergraphLinkage representation.

    Example:
        >>> linkage = Linkage(joints=[...], order=[...])
        >>> hg = from_linkage(linkage)
    """
    from ..assur.conversion import linkage_to_graph

    # Convert to Assur graph first
    assur_graph = linkage_to_graph(linkage)

    # Then convert to hypergraph
    return from_assur_graph(assur_graph)
