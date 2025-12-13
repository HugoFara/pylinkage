"""Assur group decomposition for planar linkages.

This module provides a graph-based representation of linkage mechanisms
and tools for Assur group decomposition. It offers an alternative way
to define and analyze linkages using formal kinematic theory.

Key components:
- LinkageGraph: Graph representation with nodes (joints) and edges (links)
- AssurGroup: Base class for Assur groups (structural units)
- DyadRRR, DyadRRP: Class I Assur groups (dyads)
- decompose_assur_groups: Decomposition algorithm
- Conversion functions between graph and Linkage representations

Example usage:

    >>> from pylinkage.assur import (
    ...     LinkageGraph, Node, Edge,
    ...     JointType, NodeRole,
    ...     graph_to_linkage, decompose_assur_groups
    ... )
    >>>
    >>> # Create a four-bar linkage using graph syntax
    >>> graph = LinkageGraph(name="Four-bar")
    >>> graph.add_node(Node("A", role=NodeRole.GROUND, position=(0, 0)))
    >>> graph.add_node(Node("D", role=NodeRole.GROUND, position=(3, 0)))
    >>> graph.add_node(Node("B", role=NodeRole.DRIVER, position=(0, 1), angle=0.31))
    >>> graph.add_node(Node("C", role=NodeRole.DRIVEN, position=(3, 2)))
    >>> graph.add_edge(Edge("AB", source="A", target="B", distance=1))
    >>> graph.add_edge(Edge("BC", source="B", target="C", distance=3))
    >>> graph.add_edge(Edge("CD", source="C", target="D", distance=1))
    >>>
    >>> # Analyze the structure
    >>> result = decompose_assur_groups(graph)
    >>> print(f"Groups: {[g.joint_signature for g in result.groups]}")
    >>>
    >>> # Convert to Linkage for simulation
    >>> linkage = graph_to_linkage(graph)
"""

__all__ = [
    # Types
    "JointType",
    "NodeRole",
    "NodeId",
    "EdgeId",
    # Graph structures
    "LinkageGraph",
    "Node",
    "Edge",
    # Assur groups
    "AssurGroup",
    "DyadRRR",
    "DyadRRP",
    "DyadRPR",
    "DyadPRR",
    "DYAD_TYPES",
    "identify_dyad_type",
    # Decomposition
    "DecompositionResult",
    "decompose_assur_groups",
    "validate_decomposition",
    "solve_decomposition",
    # Conversion to/from Linkage
    "linkage_to_graph",
    "graph_to_linkage",
    # Conversion to/from Hypergraph
    "from_hypergraph",
    "to_hypergraph",
    # Serialization
    "graph_to_dict",
    "graph_from_dict",
    "graph_to_json",
    "graph_from_json",
]

from ._types import EdgeId, JointType, NodeId, NodeRole
from .conversion import graph_to_linkage, linkage_to_graph
from .decomposition import (
    DecompositionResult,
    decompose_assur_groups,
    solve_decomposition,
    validate_decomposition,
)
from .graph import Edge, LinkageGraph, Node
from .groups import (
    DYAD_TYPES,
    AssurGroup,
    DyadPRR,
    DyadRPR,
    DyadRRP,
    DyadRRR,
    identify_dyad_type,
)
from .hypergraph_conversion import from_hypergraph, to_hypergraph
from .serialization import (
    graph_from_dict,
    graph_from_json,
    graph_to_dict,
    graph_to_json,
)
