"""Hypergraph-based representation of planar linkages.

This module provides a hierarchical hypergraph abstraction for defining
and manipulating planar linkages. It serves as an abstract mathematical
foundation that can be converted to other representations (Assur graphs,
joint-based linkages) for different purposes.

Key concepts:
- **HypergraphLinkage**: Abstract graph supporting both edges and hyperedges
- **Component**: Reusable linkage subgraph with ports and parameters
- **HierarchicalLinkage**: Composition of component instances

Example usage::

    from pylinkage.hypergraph import (
        HierarchicalLinkage, ComponentInstance, Connection,
        FOURBAR, to_linkage
    )

    # Create component instances
    leg1 = ComponentInstance("leg1", FOURBAR, {"crank_length": 1.5})
    leg2 = ComponentInstance("leg2", FOURBAR, {"crank_length": 1.5})

    # Assemble hierarchy
    walker = HierarchicalLinkage(
        instances={"leg1": leg1, "leg2": leg2},
        connections=[Connection("leg1", "output", "leg2", "input")],
    )

    # Flatten and simulate
    flat_graph = walker.flatten()
    linkage = to_linkage(flat_graph)
    linkage.step()
"""

# Types
from ._types import (
    ComponentId,
    EdgeId,
    HyperedgeId,
    JointType,
    NodeId,
    NodeRole,
    PortId,
)

# Component system
from .components import Component, ParameterMapping, ParameterSpec, Port

# Conversion functions (to/from Linkage only - for Assur conversion use assur module)
from .conversion import from_linkage, to_linkage

# Core graph elements
from .core import Edge, Hyperedge, Node

# Graph representation
from .graph import HypergraphLinkage

# Hierarchical composition
from .hierarchy import ComponentInstance, Connection, HierarchicalLinkage

# Component library
from .library import (
    COMPONENT_LIBRARY,
    CRANK_SLIDER,
    DYAD,
    FOURBAR,
    get_component,
    list_components,
    register_component,
)

# Serialization
from .serialization import (
    component_from_dict,
    component_to_dict,
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
    "ComponentId",
    "PortId",
    "JointType",
    "NodeRole",
    # Core elements
    "Node",
    "Edge",
    "Hyperedge",
    # Graph
    "HypergraphLinkage",
    # Components
    "Port",
    "ParameterSpec",
    "ParameterMapping",
    "Component",
    # Hierarchy
    "ComponentInstance",
    "Connection",
    "HierarchicalLinkage",
    # Conversion (to/from Linkage - for Assur use assur.from_hypergraph/to_hypergraph)
    "to_linkage",
    "from_linkage",
    # Serialization
    "graph_to_dict",
    "graph_from_dict",
    "graph_to_json",
    "graph_from_json",
    "component_to_dict",
    "component_from_dict",
    "hierarchical_to_dict",
    "hierarchical_from_dict",
    "hierarchical_to_json",
    "hierarchical_from_json",
    # Library
    "COMPONENT_LIBRARY",
    "register_component",
    "get_component",
    "list_components",
    "FOURBAR",
    "CRANK_SLIDER",
    "DYAD",
]
