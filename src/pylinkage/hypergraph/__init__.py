"""Hypergraph-based representation of planar linkages (topology only).

This module provides a hierarchical hypergraph abstraction for defining
and manipulating planar linkages. It serves as an abstract mathematical
foundation that can be converted to other representations (Assur graphs,
Mechanism) for different purposes.

This is a pure topological representation - dimensional data (positions,
distances, angles) is stored separately in a Dimensions object.

Key concepts:
- **HypergraphLinkage**: Abstract graph supporting both edges and hyperedges
- **ComponentInstance**: Instance of a topology with connection ports
- **HierarchicalLinkage**: Composition of component instances

Example usage::

    from pylinkage.hypergraph import (
        HypergraphLinkage, Node, Edge, NodeRole,
        to_mechanism
    )
    from pylinkage.dimensions import Dimensions, DriverAngle

    # Create topology
    hg = HypergraphLinkage(name="Four-bar")
    hg.add_node(Node("A", role=NodeRole.GROUND))
    hg.add_node(Node("B", role=NodeRole.DRIVER))
    hg.add_node(Node("C", role=NodeRole.DRIVEN))
    hg.add_node(Node("D", role=NodeRole.GROUND))
    hg.add_edge(Edge("AB", "A", "B"))
    hg.add_edge(Edge("BC", "B", "C"))
    hg.add_edge(Edge("CD", "C", "D"))

    # Create dimensions
    dims = Dimensions(
        node_positions={"A": (0, 0), "B": (1, 0), "C": (2, 1), "D": (3, 0)},
        driver_angles={"B": DriverAngle(0.1)},
        edge_distances={"AB": 1.0, "BC": 2.0, "CD": 2.0},
    )

    # Convert to mechanism for simulation
    mechanism = to_mechanism(hg, dims)
"""

# Types
from ._types import (
    EdgeId,
    HyperedgeId,
    JointType,
    NodeId,
    NodeRole,
    PortId,
)

# Conversion functions
# Legacy (deprecated): to_linkage, from_linkage
# Preferred: to_mechanism, from_mechanism
from .conversion import from_linkage, to_linkage

# Core graph elements
from .core import Edge, Hyperedge, Node

# Graph representation
from .graph import HypergraphLinkage

# Hierarchical composition
from .hierarchy import ComponentInstance, Connection, HierarchicalLinkage
from .mechanism_conversion import from_mechanism, to_mechanism

# Serialization
from .serialization import (
    graph_from_dict,
    graph_from_json,
    graph_to_dict,
    graph_to_json,
    hierarchical_from_dict,
    hierarchical_from_json,
    hierarchical_to_dict,
    hierarchical_to_json,
)

__all__ = [
    # Types
    "NodeId",
    "EdgeId",
    "HyperedgeId",
    "PortId",
    "JointType",
    "NodeRole",
    # Core elements
    "Node",
    "Edge",
    "Hyperedge",
    # Graph
    "HypergraphLinkage",
    # Hierarchy
    "ComponentInstance",
    "Connection",
    "HierarchicalLinkage",
    # Conversion
    # Preferred (new Mechanism model):
    "to_mechanism",
    "from_mechanism",
    # Legacy (deprecated, converts to Linkage):
    "to_linkage",
    "from_linkage",
    # Serialization
    "graph_to_dict",
    "graph_from_dict",
    "graph_to_json",
    "graph_from_json",
    "hierarchical_to_dict",
    "hierarchical_from_dict",
    "hierarchical_to_json",
    "hierarchical_from_json",
]
