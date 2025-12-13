"""Serialization support for graph representations.

This module provides functions to serialize and deserialize LinkageGraph
objects to/from dictionaries and JSON files.
"""


import json
from pathlib import Path
from typing import Any

from ._types import JointType, NodeRole
from .graph import Edge, LinkageGraph, Node


def node_to_dict(node: Node) -> dict[str, Any]:
    """Convert a Node to a dictionary.

    Args:
        node: The Node to convert.

    Returns:
        Dictionary representation of the node.
    """
    return {
        "id": node.id,
        "joint_type": node.joint_type.name,
        "role": node.role.name,
        "position": list(node.position),
        "angle": node.angle,
        "initial_angle": node.initial_angle,
        "name": node.name,
    }


def node_from_dict(data: dict[str, Any]) -> Node:
    """Create a Node from a dictionary.

    Args:
        data: Dictionary containing node data.

    Returns:
        A new Node instance.
    """
    position = tuple(data.get("position", [None, None]))
    return Node(
        id=data["id"],
        joint_type=JointType[data.get("joint_type", "REVOLUTE")],
        role=NodeRole[data.get("role", "DRIVEN")],
        position=(position[0], position[1]),
        angle=data.get("angle"),
        initial_angle=data.get("initial_angle"),
        name=data.get("name"),
    )


def edge_to_dict(edge: Edge) -> dict[str, Any]:
    """Convert an Edge to a dictionary.

    Args:
        edge: The Edge to convert.

    Returns:
        Dictionary representation of the edge.
    """
    return {
        "id": edge.id,
        "source": edge.source,
        "target": edge.target,
        "distance": edge.distance,
        "body_id": edge.body_id,
    }


def edge_from_dict(data: dict[str, Any]) -> Edge:
    """Create an Edge from a dictionary.

    Args:
        data: Dictionary containing edge data.

    Returns:
        A new Edge instance.
    """
    return Edge(
        id=data["id"],
        source=data["source"],
        target=data["target"],
        distance=data.get("distance"),
        body_id=data.get("body_id"),
    )


def graph_to_dict(graph: LinkageGraph) -> dict[str, Any]:
    """Convert a LinkageGraph to a dictionary.

    The dictionary format is suitable for JSON serialization and can be
    used to reconstruct the graph later.

    Args:
        graph: The LinkageGraph to convert.

    Returns:
        Dictionary representation of the graph.

    Example:
        >>> data = graph_to_dict(graph)
        >>> json.dumps(data, indent=2)
    """
    return {
        "name": graph.name,
        "nodes": [node_to_dict(node) for node in graph.nodes.values()],
        "edges": [edge_to_dict(edge) for edge in graph.edges.values()],
    }


def graph_from_dict(data: dict[str, Any]) -> LinkageGraph:
    """Create a LinkageGraph from a dictionary.

    Args:
        data: Dictionary containing graph data (as produced by graph_to_dict).

    Returns:
        A new LinkageGraph instance.

    Example:
        >>> data = graph_to_dict(original_graph)
        >>> restored_graph = graph_from_dict(data)
    """
    graph = LinkageGraph(name=data.get("name", ""))

    # Add nodes first
    for node_data in data.get("nodes", []):
        node = node_from_dict(node_data)
        graph.add_node(node)

    # Then add edges
    for edge_data in data.get("edges", []):
        edge = edge_from_dict(edge_data)
        graph.add_edge(edge)

    return graph


def graph_to_json(graph: LinkageGraph, path: str | Path) -> None:
    """Save a LinkageGraph to a JSON file.

    Args:
        graph: The LinkageGraph to save.
        path: Path to the output JSON file.

    Example:
        >>> graph_to_json(graph, "my_linkage.json")
        >>> restored = graph_from_json("my_linkage.json")
    """
    data = graph_to_dict(graph)
    path = Path(path)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def graph_from_json(path: str | Path) -> LinkageGraph:
    """Load a LinkageGraph from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        A new LinkageGraph instance.

    Example:
        >>> graph_to_json(graph, "my_linkage.json")
        >>> restored = graph_from_json("my_linkage.json")
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return graph_from_dict(data)
