"""Hierarchical linkage composition.

This module provides classes for assembling linkages from component instances
and flattening them to a single HypergraphLinkage for analysis or simulation.
"""


from dataclasses import dataclass, field
from typing import Any

from ._types import NodeId, PortId
from .components import Component
from .core import Edge, Hyperedge, Node
from .graph import HypergraphLinkage


@dataclass
class ComponentInstance:
    """An instance of a component with specific parameters.

    Represents a concrete instantiation of a Component template with
    specific parameter values and a unique instance ID.

    Attributes:
        id: Unique identifier for this instance within the hierarchy.
        component: The component type being instantiated.
        parameters: Parameter values for this instance.
        name: Human-readable name for this instance.

    Example:
        >>> instance = ComponentInstance(
        ...     id="left_leg",
        ...     component=leg_component,
        ...     parameters={"crank_length": 1.5},
        ...     name="Left Leg"
        ... )
    """

    id: str
    component: Component
    parameters: dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def get_qualified_node_id(self, internal_node: NodeId) -> NodeId:
        """Get the globally qualified node ID for an internal node.

        Args:
            internal_node: The internal node ID within the component.

        Returns:
            A globally unique node ID like "instance_id.internal_node".
        """
        return f"{self.id}.{internal_node}"

    def get_port_qualified_node(self, port_id: PortId) -> NodeId:
        """Get the qualified node ID for a port.

        Args:
            port_id: The port ID.

        Returns:
            The globally qualified node ID for the port's internal node.

        Raises:
            KeyError: If port not found.
        """
        internal_node = self.component.get_port_node(port_id)
        return self.get_qualified_node_id(internal_node)


@dataclass
class Connection:
    """Connection between two component ports.

    Defines how two component instances are connected via their ports.
    During flattening, connected ports share the same position.

    Attributes:
        from_instance: Instance ID of the source component.
        from_port: Port ID on the source component.
        to_instance: Instance ID of the target component.
        to_port: Port ID on the target component.

    Example:
        >>> conn = Connection(
        ...     from_instance="leg1",
        ...     from_port="output",
        ...     to_instance="leg2",
        ...     to_port="input"
        ... )
    """

    from_instance: str
    from_port: PortId
    to_instance: str
    to_port: PortId

    def __post_init__(self) -> None:
        """Validate connection structure."""
        if self.from_instance == self.to_instance and self.from_port == self.to_port:
            raise ValueError("Cannot connect a port to itself")


@dataclass
class HierarchicalLinkage:
    """Top-level container for hierarchical linkage definition.

    A hierarchical linkage contains component instances and defines
    connections between them. It can be flattened to a HypergraphLinkage
    for simulation or further processing.

    Attributes:
        instances: Dictionary of component instances by ID.
        connections: List of connections between component ports.
        name: Human-readable name for the linkage.

    Example:
        >>> # Create instances
        >>> leg1 = ComponentInstance("leg1", leg_component, {"length": 2.0})
        >>> leg2 = ComponentInstance("leg2", leg_component, {"length": 2.0})
        >>>
        >>> # Create hierarchical linkage
        >>> linkage = HierarchicalLinkage(
        ...     instances={"leg1": leg1, "leg2": leg2},
        ...     connections=[
        ...         Connection("leg1", "output", "leg2", "input"),
        ...     ],
        ...     name="Two-Leg Walker"
        ... )
        >>>
        >>> # Flatten for simulation
        >>> flat_graph = linkage.flatten()
    """

    instances: dict[str, ComponentInstance] = field(default_factory=dict)
    connections: list[Connection] = field(default_factory=list)
    name: str = ""

    def __post_init__(self) -> None:
        """Validate hierarchical structure."""
        self._validate_connections()

    def _validate_connections(self) -> None:
        """Validate all connections reference valid instances and ports."""
        for conn in self.connections:
            # Check instances exist
            if conn.from_instance not in self.instances:
                raise ValueError(
                    f"Connection references unknown instance '{conn.from_instance}'"
                )
            if conn.to_instance not in self.instances:
                raise ValueError(
                    f"Connection references unknown instance '{conn.to_instance}'"
                )

            # Check ports exist
            from_inst = self.instances[conn.from_instance]
            to_inst = self.instances[conn.to_instance]

            if conn.from_port not in from_inst.component.ports:
                raise ValueError(
                    f"Connection references unknown port '{conn.from_port}' "
                    f"on instance '{conn.from_instance}'"
                )
            if conn.to_port not in to_inst.component.ports:
                raise ValueError(
                    f"Connection references unknown port '{conn.to_port}' "
                    f"on instance '{conn.to_instance}'"
                )

    def add_instance(self, instance: ComponentInstance) -> None:
        """Add a component instance.

        Args:
            instance: The ComponentInstance to add.

        Raises:
            ValueError: If an instance with the same ID already exists.
        """
        if instance.id in self.instances:
            raise ValueError(f"Instance with id '{instance.id}' already exists")
        self.instances[instance.id] = instance

    def add_connection(self, connection: Connection) -> None:
        """Add a connection between ports.

        Args:
            connection: The Connection to add.

        Raises:
            ValueError: If the connection references invalid instances or ports.
        """
        # Validate before adding
        if connection.from_instance not in self.instances:
            raise ValueError(
                f"Connection references unknown instance '{connection.from_instance}'"
            )
        if connection.to_instance not in self.instances:
            raise ValueError(
                f"Connection references unknown instance '{connection.to_instance}'"
            )

        from_inst = self.instances[connection.from_instance]
        to_inst = self.instances[connection.to_instance]

        if connection.from_port not in from_inst.component.ports:
            raise ValueError(
                f"Connection references unknown port '{connection.from_port}'"
            )
        if connection.to_port not in to_inst.component.ports:
            raise ValueError(
                f"Connection references unknown port '{connection.to_port}'"
            )

        self.connections.append(connection)

    def flatten(self) -> HypergraphLinkage:
        """Flatten the hierarchy to a single HypergraphLinkage.

        This is the key method that converts the hierarchical representation
        to a flat hypergraph that can be converted to other representations
        or used for analysis.

        The flattening process:
        1. Creates qualified nodes from all component instances
        2. Creates qualified edges from all component instances
        3. Creates qualified hyperedges from all component instances
        4. Merges connected ports into single nodes

        Returns:
            A HypergraphLinkage with all components expanded and connected.
        """
        flat_graph = HypergraphLinkage(name=self.name)

        # Track port node merging: maps qualified node IDs to their canonical ID
        node_merging: dict[NodeId, NodeId] = {}

        # First pass: build merge map from connections
        for conn in self.connections:
            from_inst = self.instances[conn.from_instance]
            to_inst = self.instances[conn.to_instance]

            from_node = from_inst.get_port_qualified_node(conn.from_port)
            to_node = to_inst.get_port_qualified_node(conn.to_port)

            # Use the "from" node as canonical (arbitrary choice)
            # Follow existing merges
            canonical_from = node_merging.get(from_node, from_node)
            canonical_to = node_merging.get(to_node, to_node)

            # If both already have different canonicals, merge them
            if canonical_from != canonical_to:
                # Update all references to canonical_to -> canonical_from
                for key, val in list(node_merging.items()):
                    if val == canonical_to:
                        node_merging[key] = canonical_from
                node_merging[to_node] = canonical_from
            else:
                node_merging[to_node] = canonical_from

        def get_canonical(node_id: NodeId) -> NodeId:
            """Get the canonical (merged) node ID."""
            return node_merging.get(node_id, node_id)

        # Second pass: add nodes from all instances
        added_canonical_nodes: set[NodeId] = set()

        for instance_id, instance in self.instances.items():
            # Apply parameters to get configured component
            configured = instance.component.copy_with_parameters(instance.parameters)

            for node in configured.internal_graph.nodes.values():
                qualified_id = instance.get_qualified_node_id(node.id)
                canonical_id = get_canonical(qualified_id)

                # Skip if this canonical node already added
                if canonical_id in added_canonical_nodes:
                    continue

                qualified_name = (
                    f"{instance.name}.{node.name}"
                    if instance.name
                    else f"{instance_id}.{node.name}"
                )

                flat_graph.add_node(
                    Node(
                        id=canonical_id,
                        position=node.position,
                        role=node.role,
                        joint_type=node.joint_type,
                        angle=node.angle,
                        initial_angle=node.initial_angle,
                        name=qualified_name,
                    )
                )
                added_canonical_nodes.add(canonical_id)

        # Third pass: add edges from all instances
        for instance_id, instance in self.instances.items():
            configured = instance.component.copy_with_parameters(instance.parameters)

            for edge in configured.internal_graph.edges.values():
                qualified_source = get_canonical(
                    instance.get_qualified_node_id(edge.source)
                )
                qualified_target = get_canonical(
                    instance.get_qualified_node_id(edge.target)
                )

                # Skip self-loops created by merging
                if qualified_source == qualified_target:
                    continue

                qualified_edge_id = f"{instance_id}.{edge.id}"

                flat_graph.add_edge(
                    Edge(
                        id=qualified_edge_id,
                        source=qualified_source,
                        target=qualified_target,
                        distance=edge.distance,
                    )
                )

        # Fourth pass: add hyperedges from all instances
        for instance_id, instance in self.instances.items():
            configured = instance.component.copy_with_parameters(instance.parameters)

            for hyperedge in configured.internal_graph.hyperedges.values():
                qualified_nodes = tuple(
                    get_canonical(instance.get_qualified_node_id(n))
                    for n in hyperedge.nodes
                )

                # Deduplicate nodes (in case of merging)
                unique_nodes = tuple(dict.fromkeys(qualified_nodes))

                # Update constraints with canonical node IDs
                qualified_constraints: dict[tuple[NodeId, NodeId], float] = {}
                for (n1, n2), dist in hyperedge.constraints.items():
                    qn1 = get_canonical(instance.get_qualified_node_id(n1))
                    qn2 = get_canonical(instance.get_qualified_node_id(n2))
                    if qn1 != qn2:  # Skip constraints between merged nodes
                        constraint_key: tuple[NodeId, NodeId] = (min(qn1, qn2), max(qn1, qn2))
                        qualified_constraints[constraint_key] = dist

                if len(unique_nodes) >= 2 and qualified_constraints:
                    qualified_he_id = f"{instance_id}.{hyperedge.id}"
                    flat_graph.add_hyperedge(
                        Hyperedge(
                            id=qualified_he_id,
                            nodes=unique_nodes,
                            constraints=qualified_constraints,
                            name=f"{instance_id}.{hyperedge.name}"
                            if hyperedge.name
                            else None,
                        )
                    )

        return flat_graph

    def get_all_port_nodes(self) -> dict[str, NodeId]:
        """Get all exposed port nodes with their qualified IDs.

        Returns:
            Dictionary mapping "instance_id.port_id" to qualified node ID.
        """
        result = {}
        for inst_id, instance in self.instances.items():
            for port_id in instance.component.ports:
                key = f"{inst_id}.{port_id}"
                result[key] = instance.get_port_qualified_node(port_id)
        return result
