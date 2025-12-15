"""Serialization for hypergraph structures (topology only).

This module provides functions to convert hypergraph objects to and from
dictionaries and JSON files. This enables persistence and interoperability.

Note: This module only handles topological data. For dimensional data,
use the serialization functions in pylinkage.dimensions.
"""

import json
from pathlib import Path
from typing import Any

from ._types import JointType, NodeRole
from .core import Edge, Hyperedge, Node
from .graph import HypergraphLinkage
from .hierarchy import ComponentInstance, Connection, HierarchicalLinkage

# ============================================================================
# Node serialization (topology only)
# ============================================================================


def node_to_dict(node: Node) -> dict[str, Any]:
    """Convert a Node to a dictionary (topology only).

    Args:
        node: The Node to convert.

    Returns:
        Dictionary representation of the node.
    """
    return {
        "id": node.id,
        "role": node.role.name,
        "joint_type": node.joint_type.name,
        "name": node.name,
    }


def node_from_dict(data: dict[str, Any]) -> Node:
    """Create a Node from a dictionary.

    Args:
        data: Dictionary representation.

    Returns:
        A Node instance.
    """
    return Node(
        id=data["id"],
        role=NodeRole[data.get("role", "DRIVEN")],
        joint_type=JointType[data.get("joint_type", "REVOLUTE")],
        name=data.get("name"),
    )


# ============================================================================
# Edge serialization (topology only)
# ============================================================================


def edge_to_dict(edge: Edge) -> dict[str, Any]:
    """Convert an Edge to a dictionary (topology only).

    Args:
        edge: The Edge to convert.

    Returns:
        Dictionary representation of the edge.
    """
    return {
        "id": edge.id,
        "source": edge.source,
        "target": edge.target,
    }


def edge_from_dict(data: dict[str, Any]) -> Edge:
    """Create an Edge from a dictionary.

    Args:
        data: Dictionary representation.

    Returns:
        An Edge instance.
    """
    return Edge(
        id=data["id"],
        source=data["source"],
        target=data["target"],
    )


# ============================================================================
# Hyperedge serialization (topology only)
# ============================================================================


def hyperedge_to_dict(hyperedge: Hyperedge) -> dict[str, Any]:
    """Convert a Hyperedge to a dictionary (topology only).

    Args:
        hyperedge: The Hyperedge to convert.

    Returns:
        Dictionary representation of the hyperedge.
    """
    return {
        "id": hyperedge.id,
        "nodes": list(hyperedge.nodes),
        "name": hyperedge.name,
    }


def hyperedge_from_dict(data: dict[str, Any]) -> Hyperedge:
    """Create a Hyperedge from a dictionary.

    Args:
        data: Dictionary representation.

    Returns:
        A Hyperedge instance.
    """
    return Hyperedge(
        id=data["id"],
        nodes=tuple(data["nodes"]),
        name=data.get("name"),
    )


# ============================================================================
# HypergraphLinkage serialization
# ============================================================================


def graph_to_dict(graph: HypergraphLinkage) -> dict[str, Any]:
    """Convert a HypergraphLinkage to a dictionary.

    Args:
        graph: The HypergraphLinkage to convert.

    Returns:
        Dictionary representation of the graph.
    """
    return {
        "name": graph.name,
        "nodes": [node_to_dict(n) for n in graph.nodes.values()],
        "edges": [edge_to_dict(e) for e in graph.edges.values()],
        "hyperedges": [hyperedge_to_dict(h) for h in graph.hyperedges.values()],
    }


def graph_from_dict(data: dict[str, Any]) -> HypergraphLinkage:
    """Create a HypergraphLinkage from a dictionary.

    Args:
        data: Dictionary representation.

    Returns:
        A HypergraphLinkage instance.
    """
    graph = HypergraphLinkage(name=data.get("name", ""))

    for node_data in data.get("nodes", []):
        graph.add_node(node_from_dict(node_data))

    for edge_data in data.get("edges", []):
        graph.add_edge(edge_from_dict(edge_data))

    for he_data in data.get("hyperedges", []):
        graph.add_hyperedge(hyperedge_from_dict(he_data))

    return graph


def graph_to_json(graph: HypergraphLinkage, path: str | Path) -> None:
    """Save a HypergraphLinkage to a JSON file.

    Args:
        graph: The HypergraphLinkage to save.
        path: Path to the output file.
    """
    data = graph_to_dict(graph)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def graph_from_json(path: str | Path) -> HypergraphLinkage:
    """Load a HypergraphLinkage from a JSON file.

    Args:
        path: Path to the input file.

    Returns:
        A HypergraphLinkage instance.
    """
    with open(path) as f:
        data = json.load(f)
    return graph_from_dict(data)


# ============================================================================
# Hierarchy serialization
# ============================================================================


def component_instance_to_dict(instance: ComponentInstance) -> dict[str, Any]:
    """Convert a ComponentInstance to a dictionary.

    Args:
        instance: The ComponentInstance to convert.

    Returns:
        Dictionary representation.
    """
    return {
        "id": instance.id,
        "topology": graph_to_dict(instance.topology),
        "ports": instance.ports,
        "name": instance.name,
    }


def component_instance_from_dict(data: dict[str, Any]) -> ComponentInstance:
    """Create a ComponentInstance from a dictionary.

    Args:
        data: Dictionary representation.

    Returns:
        A ComponentInstance.
    """
    return ComponentInstance(
        id=data["id"],
        topology=graph_from_dict(data["topology"]),
        ports=data.get("ports", {}),
        name=data.get("name", ""),
    )


def connection_to_dict(connection: Connection) -> dict[str, Any]:
    """Convert a Connection to a dictionary."""
    return {
        "from_instance": connection.from_instance,
        "from_port": connection.from_port,
        "to_instance": connection.to_instance,
        "to_port": connection.to_port,
    }


def connection_from_dict(data: dict[str, Any]) -> Connection:
    """Create a Connection from a dictionary."""
    return Connection(
        from_instance=data["from_instance"],
        from_port=data["from_port"],
        to_instance=data["to_instance"],
        to_port=data["to_port"],
    )


def hierarchical_to_dict(linkage: HierarchicalLinkage) -> dict[str, Any]:
    """Convert a HierarchicalLinkage to a dictionary.

    Args:
        linkage: The HierarchicalLinkage to convert.

    Returns:
        Dictionary representation.
    """
    return {
        "name": linkage.name,
        "instances": {
            iid: component_instance_to_dict(inst)
            for iid, inst in linkage.instances.items()
        },
        "connections": [connection_to_dict(c) for c in linkage.connections],
    }


def hierarchical_from_dict(data: dict[str, Any]) -> HierarchicalLinkage:
    """Create a HierarchicalLinkage from a dictionary.

    Args:
        data: Dictionary representation.

    Returns:
        A HierarchicalLinkage instance.
    """
    instances = {
        iid: component_instance_from_dict(inst_data)
        for iid, inst_data in data.get("instances", {}).items()
    }
    connections = [connection_from_dict(c) for c in data.get("connections", [])]

    return HierarchicalLinkage(
        instances=instances,
        connections=connections,
        name=data.get("name", ""),
    )


def hierarchical_to_json(linkage: HierarchicalLinkage, path: str | Path) -> None:
    """Save a HierarchicalLinkage to a JSON file.

    Args:
        linkage: The HierarchicalLinkage to save.
        path: Path to the output file.
    """
    data = hierarchical_to_dict(linkage)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def hierarchical_from_json(path: str | Path) -> HierarchicalLinkage:
    """Load a HierarchicalLinkage from a JSON file.

    Args:
        path: Path to the input file.

    Returns:
        A HierarchicalLinkage instance.
    """
    with open(path) as f:
        data = json.load(f)
    return hierarchical_from_dict(data)
