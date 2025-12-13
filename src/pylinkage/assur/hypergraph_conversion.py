"""Conversion between Assur graph and hypergraph representations.

This module provides functions to convert between the Assur LinkageGraph
and the hypergraph HypergraphLinkage representations.

Since assur is built on top of hypergraph (as a formal kinematic theory
on top of abstract graph math), these conversions live in the assur module.
"""

from typing import TYPE_CHECKING

from ._types import JointType, NodeId, NodeRole
from .graph import Edge as AssurEdge
from .graph import LinkageGraph
from .graph import Node as AssurNode

if TYPE_CHECKING:
    from ..hypergraph.graph import HypergraphLinkage


def from_hypergraph(hypergraph: "HypergraphLinkage") -> LinkageGraph:
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
        >>> assur_graph = from_hypergraph(hg)
    """
    from ..hypergraph._types import JointType as HyperJointType
    from ..hypergraph._types import NodeRole as HyperNodeRole

    # First convert to simple graph (expand hyperedges)
    simple = hypergraph.to_simple_graph()

    assur_graph = LinkageGraph(name=simple.name)

    # Map hypergraph types to assur types (they're the same enums now)
    role_map = {
        HyperNodeRole.GROUND: NodeRole.GROUND,
        HyperNodeRole.DRIVER: NodeRole.DRIVER,
        HyperNodeRole.DRIVEN: NodeRole.DRIVEN,
    }
    joint_type_map = {
        HyperJointType.REVOLUTE: JointType.REVOLUTE,
        HyperJointType.PRISMATIC: JointType.PRISMATIC,
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


def to_hypergraph(assur_graph: LinkageGraph) -> "HypergraphLinkage":
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
        >>> hg = to_hypergraph(assur_graph)
    """
    from ..hypergraph._types import JointType as HyperJointType
    from ..hypergraph._types import NodeRole as HyperNodeRole
    from ..hypergraph.core import Edge as HyperEdge
    from ..hypergraph.core import Hyperedge
    from ..hypergraph.core import Node as HyperNode
    from ..hypergraph.graph import HypergraphLinkage

    hypergraph = HypergraphLinkage(name=assur_graph.name)

    # Map assur types to hypergraph types
    role_map = {
        NodeRole.GROUND: HyperNodeRole.GROUND,
        NodeRole.DRIVER: HyperNodeRole.DRIVER,
        NodeRole.DRIVEN: HyperNodeRole.DRIVEN,
    }
    joint_type_map = {
        JointType.REVOLUTE: HyperJointType.REVOLUTE,
        JointType.PRISMATIC: HyperJointType.PRISMATIC,
    }

    # Convert nodes
    for node in assur_graph.nodes.values():
        hyper_node = HyperNode(
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
            nodes_set: set[NodeId] = set()
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
                hyper_edge = HyperEdge(
                    id=edge_id,
                    source=source,
                    target=target,
                    distance=distance,
                )
                hypergraph.add_edge(hyper_edge)

    return hypergraph
