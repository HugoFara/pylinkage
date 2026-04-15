"""Conversion between Assur graph and hypergraph representations (topology only).

This module provides functions to convert between the Assur LinkageGraph
and the hypergraph HypergraphLinkage representations.

These conversions are pure topology - no dimensional data is transferred.
Dimensions are handled separately and passed through to conversion functions
that need them (e.g., graph_to_mechanism, to_mechanism).

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
    """Convert a HypergraphLinkage to an Assur LinkageGraph (topology only).

    This converts the hypergraph to the simpler graph representation
    used by the Assur decomposition system. Hyperedges are expanded
    to regular edges.

    Both representations are pure topology - no dimensional data is transferred.
    To convert with dimensions, use this function then pass the same Dimensions
    object to graph_to_mechanism.

    Args:
        hypergraph: The HypergraphLinkage to convert (topology only).

    Returns:
        An Assur LinkageGraph suitable for decomposition and analysis.

    Example:
        >>> hg = HypergraphLinkage(name="Four-bar")
        >>> # ... add nodes and edges ...
        >>> assur_graph = from_hypergraph(hg)
    """
    from .._types import JointType as HyperJointType
    from .._types import NodeRole as HyperNodeRole

    assur_graph = LinkageGraph(name=hypergraph.name)

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

    # Convert nodes (topology only)
    for node in hypergraph.nodes.values():
        assur_node = AssurNode(
            id=node.id,
            joint_type=joint_type_map[node.joint_type],
            role=role_map[node.role],
            name=node.name,
        )
        assur_graph.add_node(assur_node)

    # Convert edges (topology only)
    for edge in hypergraph.edges.values():
        assur_edge = AssurEdge(
            id=edge.id,
            source=edge.source,
            target=edge.target,
        )
        assur_graph.add_edge(assur_edge)

    # Expand hyperedges as cliques (all pairwise edges).
    # A hyperedge represents a rigid body connecting 3+ joints.
    # For Assur decomposition, all joints on the same rigid body must
    # be mutual neighbors — chain expansion is insufficient because a
    # joint at one end of the chain won't see a joint at the other end
    # as a neighbor, even though they're on the same rigid link.
    seen_edges: set[tuple[NodeId, NodeId]] = set()
    for existing_edge in assur_graph.edges.values():
        src, tgt = existing_edge.source, existing_edge.target
        pair = (min(src, tgt), max(src, tgt))
        seen_edges.add(pair)

    for he in hypergraph.hyperedges.values():
        nodes_list = list(he.nodes)
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                n1, n2 = nodes_list[i], nodes_list[j]
                pair = (min(n1, n2), max(n1, n2))
                if pair not in seen_edges:
                    seen_edges.add(pair)
                    edge_id = f"{he.id}_{n1}_{n2}"
                    assur_graph.add_edge(AssurEdge(
                        id=edge_id,
                        source=n1,
                        target=n2,
                        body_id=he.id,
                    ))

    return assur_graph


def to_hypergraph(assur_graph: LinkageGraph) -> "HypergraphLinkage":
    """Convert an Assur LinkageGraph to a HypergraphLinkage (topology only).

    This converts from the Assur graph representation to the more
    abstract hypergraph representation. Edges with the same body_id
    are grouped into hyperedges.

    Both representations are pure topology - no dimensional data is transferred.

    Args:
        assur_graph: The Assur LinkageGraph to convert (topology only).

    Returns:
        A HypergraphLinkage representation (topology only).

    Example:
        >>> assur_graph = LinkageGraph(name="Four-bar")
        >>> # ... add nodes and edges ...
        >>> hg = to_hypergraph(assur_graph)
    """
    from .._types import JointType as HyperJointType
    from .._types import NodeRole as HyperNodeRole
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

    # Convert nodes (topology only)
    for node in assur_graph.nodes.values():
        hyper_node = HyperNode(
            id=node.id,
            role=role_map[node.role],
            joint_type=joint_type_map[node.joint_type],
            name=node.name,
        )
        hypergraph.add_node(hyper_node)

    # Group edges by body_id to potentially create hyperedges
    body_edges: dict[str | None, list[tuple[str, str, str]]] = {}
    for edge in assur_graph.edges.values():
        body_id = getattr(edge, "body_id", None)
        if body_id not in body_edges:
            body_edges[body_id] = []
        body_edges[body_id].append((edge.id, edge.source, edge.target))

    # Convert edges - edges with body_id become hyperedges, others stay as edges
    for body_id, edges in body_edges.items():
        if body_id is not None and len(edges) > 1:
            # Create hyperedge from grouped edges
            nodes_set: set[NodeId] = set()
            for _, source, target in edges:
                nodes_set.add(source)
                nodes_set.add(target)

            hyperedge = Hyperedge(
                id=body_id,
                nodes=tuple(sorted(nodes_set)),
                name=body_id,
            )
            hypergraph.add_hyperedge(hyperedge)
        else:
            # Keep as regular edges
            for edge_id, source, target in edges:
                hyper_edge = HyperEdge(
                    id=edge_id,
                    source=source,
                    target=target,
                )
                hypergraph.add_edge(hyper_edge)

    return hypergraph
