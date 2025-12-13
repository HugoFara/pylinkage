"""Serialization for hypergraph structures.

This module provides functions to convert hypergraph objects to and from
dictionaries and JSON files. This enables persistence and interoperability.
"""

import json
from pathlib import Path
from typing import Any

from ._types import JointType, NodeRole
from .components import Component, ParameterMapping, ParameterSpec, Port
from .core import Edge, Hyperedge, Node
from .graph import HypergraphLinkage
from .hierarchy import ComponentInstance, Connection, HierarchicalLinkage

# ============================================================================
# Node serialization
# ============================================================================


def node_to_dict(node: Node) -> dict[str, Any]:
    """Convert a Node to a dictionary.

    Args:
        node: The Node to convert.

    Returns:
        Dictionary representation of the node.
    """
    return {
        "id": node.id,
        "position": list(node.position),
        "role": node.role.name,
        "joint_type": node.joint_type.name,
        "angle": node.angle,
        "initial_angle": node.initial_angle,
        "name": node.name,
    }


def node_from_dict(data: dict[str, Any]) -> Node:
    """Create a Node from a dictionary.

    Args:
        data: Dictionary representation.

    Returns:
        A Node instance.
    """
    position = tuple(data.get("position", [None, None]))
    return Node(
        id=data["id"],
        position=(position[0], position[1]),
        role=NodeRole[data.get("role", "DRIVEN")],
        joint_type=JointType[data.get("joint_type", "REVOLUTE")],
        angle=data.get("angle"),
        initial_angle=data.get("initial_angle"),
        name=data.get("name"),
    )


# ============================================================================
# Edge serialization
# ============================================================================


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
        distance=data.get("distance"),
    )


# ============================================================================
# Hyperedge serialization
# ============================================================================


def hyperedge_to_dict(hyperedge: Hyperedge) -> dict[str, Any]:
    """Convert a Hyperedge to a dictionary.

    Args:
        hyperedge: The Hyperedge to convert.

    Returns:
        Dictionary representation of the hyperedge.
    """
    # Convert constraint keys from tuples to lists for JSON compatibility
    constraints = [
        {"nodes": list(key), "distance": dist}
        for key, dist in hyperedge.constraints.items()
    ]
    return {
        "id": hyperedge.id,
        "nodes": list(hyperedge.nodes),
        "constraints": constraints,
        "name": hyperedge.name,
    }


def hyperedge_from_dict(data: dict[str, Any]) -> Hyperedge:
    """Create a Hyperedge from a dictionary.

    Args:
        data: Dictionary representation.

    Returns:
        A Hyperedge instance.
    """
    constraints = {
        tuple(c["nodes"]): c["distance"] for c in data.get("constraints", [])
    }
    return Hyperedge(
        id=data["id"],
        nodes=tuple(data["nodes"]),
        constraints=constraints,
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
# Component serialization
# ============================================================================


def port_to_dict(port: Port) -> dict[str, Any]:
    """Convert a Port to a dictionary."""
    return {
        "id": port.id,
        "internal_node": port.internal_node,
        "description": port.description,
    }


def port_from_dict(data: dict[str, Any]) -> Port:
    """Create a Port from a dictionary."""
    return Port(
        id=data["id"],
        internal_node=data["internal_node"],
        description=data.get("description", ""),
    )


def parameter_spec_to_dict(spec: ParameterSpec) -> dict[str, Any]:
    """Convert a ParameterSpec to a dictionary.

    Note: The validator function is not serialized.
    """
    return {
        "name": spec.name,
        "description": spec.description,
        "default": spec.default,
        "min_value": spec.min_value,
        "max_value": spec.max_value,
    }


def parameter_spec_from_dict(data: dict[str, Any]) -> ParameterSpec:
    """Create a ParameterSpec from a dictionary."""
    return ParameterSpec(
        name=data["name"],
        description=data.get("description", ""),
        default=data.get("default"),
        min_value=data.get("min_value"),
        max_value=data.get("max_value"),
    )


def parameter_mapping_to_dict(mapping: ParameterMapping) -> dict[str, Any]:
    """Convert a ParameterMapping to a dictionary."""
    return {
        "parameter_name": mapping.parameter_name,
        "edge_ids": mapping.edge_ids,
        "hyperedge_constraints": [
            {"hyperedge_id": he_id, "nodes": list(nodes)}
            for he_id, nodes in mapping.hyperedge_constraints
        ],
    }


def parameter_mapping_from_dict(data: dict[str, Any]) -> ParameterMapping:
    """Create a ParameterMapping from a dictionary."""
    hyperedge_constraints = [
        (item["hyperedge_id"], tuple(item["nodes"]))
        for item in data.get("hyperedge_constraints", [])
    ]
    return ParameterMapping(
        parameter_name=data["parameter_name"],
        edge_ids=data.get("edge_ids", []),
        hyperedge_constraints=hyperedge_constraints,
    )


def component_to_dict(component: Component) -> dict[str, Any]:
    """Convert a Component to a dictionary."""
    return {
        "id": component.id,
        "internal_graph": graph_to_dict(component.internal_graph),
        "ports": {pid: port_to_dict(p) for pid, p in component.ports.items()},
        "parameters": {
            pname: parameter_spec_to_dict(spec)
            for pname, spec in component.parameters.items()
        },
        "parameter_mappings": [
            parameter_mapping_to_dict(m) for m in component.parameter_mappings
        ],
        "name": component.name,
    }


def component_from_dict(data: dict[str, Any]) -> Component:
    """Create a Component from a dictionary."""
    return Component(
        id=data["id"],
        internal_graph=graph_from_dict(data["internal_graph"]),
        ports={pid: port_from_dict(p) for pid, p in data.get("ports", {}).items()},
        parameters={
            pname: parameter_spec_from_dict(spec)
            for pname, spec in data.get("parameters", {}).items()
        },
        parameter_mappings=[
            parameter_mapping_from_dict(m) for m in data.get("parameter_mappings", [])
        ],
        name=data.get("name", ""),
    )


# ============================================================================
# Hierarchy serialization
# ============================================================================


def component_instance_to_dict(
    instance: ComponentInstance, include_component: bool = True
) -> dict[str, Any]:
    """Convert a ComponentInstance to a dictionary.

    Args:
        instance: The ComponentInstance to convert.
        include_component: If True, includes full component definition.
            If False, only includes component ID (for when component
            is defined elsewhere).

    Returns:
        Dictionary representation.
    """
    result: dict[str, Any] = {
        "id": instance.id,
        "parameters": instance.parameters,
        "name": instance.name,
    }
    if include_component:
        result["component"] = component_to_dict(instance.component)
    else:
        result["component_id"] = instance.component.id
    return result


def component_instance_from_dict(
    data: dict[str, Any], component_registry: dict[str, Component] | None = None
) -> ComponentInstance:
    """Create a ComponentInstance from a dictionary.

    Args:
        data: Dictionary representation.
        component_registry: Optional registry to look up components by ID.
            If data contains 'component', this is not used.

    Returns:
        A ComponentInstance.

    Raises:
        ValueError: If component cannot be resolved.
    """
    if "component" in data:
        component = component_from_dict(data["component"])
    elif component_registry and "component_id" in data:
        component_id = data["component_id"]
        if component_id not in component_registry:
            raise ValueError(f"Component '{component_id}' not found in registry")
        component = component_registry[component_id]
    else:
        raise ValueError("Cannot resolve component: no component data or registry")

    return ComponentInstance(
        id=data["id"],
        component=component,
        parameters=data.get("parameters", {}),
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


def hierarchical_to_dict(
    linkage: HierarchicalLinkage, include_components: bool = True
) -> dict[str, Any]:
    """Convert a HierarchicalLinkage to a dictionary.

    Args:
        linkage: The HierarchicalLinkage to convert.
        include_components: If True, includes full component definitions.

    Returns:
        Dictionary representation.
    """
    return {
        "name": linkage.name,
        "instances": {
            iid: component_instance_to_dict(inst, include_component=include_components)
            for iid, inst in linkage.instances.items()
        },
        "connections": [connection_to_dict(c) for c in linkage.connections],
    }


def hierarchical_from_dict(
    data: dict[str, Any], component_registry: dict[str, Component] | None = None
) -> HierarchicalLinkage:
    """Create a HierarchicalLinkage from a dictionary.

    Args:
        data: Dictionary representation.
        component_registry: Optional registry for looking up components.

    Returns:
        A HierarchicalLinkage instance.
    """
    instances = {
        iid: component_instance_from_dict(inst_data, component_registry)
        for iid, inst_data in data.get("instances", {}).items()
    }
    connections = [connection_from_dict(c) for c in data.get("connections", [])]

    return HierarchicalLinkage(
        instances=instances,
        connections=connections,
        name=data.get("name", ""),
    )


def hierarchical_to_json(
    linkage: HierarchicalLinkage, path: str | Path, include_components: bool = True
) -> None:
    """Save a HierarchicalLinkage to a JSON file.

    Args:
        linkage: The HierarchicalLinkage to save.
        path: Path to the output file.
        include_components: If True, includes full component definitions.
    """
    data = hierarchical_to_dict(linkage, include_components=include_components)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def hierarchical_from_json(
    path: str | Path, component_registry: dict[str, Component] | None = None
) -> HierarchicalLinkage:
    """Load a HierarchicalLinkage from a JSON file.

    Args:
        path: Path to the input file.
        component_registry: Optional registry for looking up components.

    Returns:
        A HierarchicalLinkage instance.
    """
    with open(path) as f:
        data = json.load(f)
    return hierarchical_from_dict(data, component_registry)
