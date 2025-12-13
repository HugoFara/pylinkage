"""Component system for reusable linkage building blocks.

This module provides the Component class and related structures that enable
defining reusable, parameterized linkage subgraphs with well-defined interfaces.
"""


from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ._types import ComponentId, NodeId, PortId
from .core import Edge, Hyperedge
from .graph import HypergraphLinkage


@dataclass
class Port:
    """External interface point for a component.

    Ports allow components to be connected together. Each port
    maps to a specific node in the component's internal graph.

    Attributes:
        id: Unique identifier for this port within the component.
        internal_node: The node ID in the internal graph that this port exposes.
        description: Human-readable description of the port's purpose.

    Example:
        >>> port = Port("input", "crank_joint", "Input from motor")
    """

    id: PortId
    internal_node: NodeId
    description: str = ""


@dataclass
class ParameterSpec:
    """Specification for a component parameter.

    Parameters allow components to be configured (e.g., link lengths).
    Each parameter has a name, optional description, default value,
    and optional validation function.

    Attributes:
        name: Parameter name (e.g., "crank_length").
        description: Human-readable description.
        default: Default value if not specified.
        min_value: Optional minimum allowed value.
        max_value: Optional maximum allowed value.
        validator: Optional validation function (value) -> bool.

    Example:
        >>> spec = ParameterSpec(
        ...     name="crank_length",
        ...     description="Length of the crank arm",
        ...     default=1.0,
        ...     min_value=0.1,
        ... )
    """

    name: str
    description: str = ""
    default: Any = None
    min_value: float | None = None
    max_value: float | None = None
    validator: Callable[[Any], bool] | None = None

    def validate(self, value: Any) -> bool:
        """Validate a parameter value.

        Args:
            value: The value to validate.

        Returns:
            True if valid, False otherwise.
        """
        # Check bounds if numeric
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False

        # Check custom validator
        if self.validator is not None:
            return self.validator(value)

        return True


@dataclass
class ParameterMapping:
    """Maps a parameter to constraints in the graph.

    Defines how a parameter value should be applied to the internal
    graph structure (e.g., which edge distances to update).

    Attributes:
        parameter_name: Name of the parameter this mapping applies to.
        edge_ids: List of edge IDs whose distances should be set.
        hyperedge_constraints: List of (hyperedge_id, (node1, node2)) tuples
            specifying which hyperedge constraints to update.
    """

    parameter_name: str
    edge_ids: list[str] = field(default_factory=list)
    hyperedge_constraints: list[tuple[str, tuple[NodeId, NodeId]]] = field(
        default_factory=list
    )


@dataclass
class Component:
    """Reusable linkage subgraph with ports and parameters.

    A component encapsulates a HypergraphLinkage as internal structure,
    exposes ports for external connections, and supports parameterization.
    This enables building complex linkages from reusable building blocks.

    Attributes:
        id: Unique identifier for this component type.
        internal_graph: The hypergraph structure inside this component.
        ports: Dictionary of ports that can be connected externally.
        parameters: Dictionary of parameter specifications.
        parameter_mappings: How parameters map to graph constraints.
        name: Human-readable name for this component type.

    Example:
        >>> # Create a simple two-bar component
        >>> graph = HypergraphLinkage(name="TwoBar-Internal")
        >>> graph.add_node(Node("A", role=NodeRole.GROUND, position=(0, 0)))
        >>> graph.add_node(Node("B", role=NodeRole.DRIVEN, position=(1, 0)))
        >>> graph.add_edge(Edge("AB", source="A", target="B", distance=1.0))
        >>> component = Component(
        ...     id="twobar",
        ...     internal_graph=graph,
        ...     ports={
        ...         "base": Port("base", "A", "Fixed base point"),
        ...         "tip": Port("tip", "B", "Moving endpoint"),
        ...     },
        ...     parameters={
        ...         "length": ParameterSpec("length", "Link length", default=1.0),
        ...     },
        ...     name="Two-Bar Link"
        ... )
    """

    id: ComponentId
    internal_graph: HypergraphLinkage
    ports: dict[PortId, Port] = field(default_factory=dict)
    parameters: dict[str, ParameterSpec] = field(default_factory=dict)
    parameter_mappings: list[ParameterMapping] = field(default_factory=list)
    name: str = ""

    def __post_init__(self) -> None:
        """Validate component structure."""
        # Ensure all port internal_nodes exist in the graph
        for port in self.ports.values():
            if port.internal_node not in self.internal_graph.nodes:
                raise ValueError(
                    f"Port '{port.id}' references non-existent node '{port.internal_node}'"
                )

        # Validate parameter mappings reference valid parameters
        param_names = set(self.parameters.keys())
        for mapping in self.parameter_mappings:
            if mapping.parameter_name not in param_names:
                raise ValueError(
                    f"Parameter mapping references unknown parameter '{mapping.parameter_name}'"
                )

    def get_port_node(self, port_id: PortId) -> NodeId:
        """Get the internal node ID for a port.

        Args:
            port_id: The port ID.

        Returns:
            The internal node ID.

        Raises:
            KeyError: If port not found.
        """
        return self.ports[port_id].internal_node

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Validate a set of parameter values.

        Args:
            params: Dictionary of parameter name to value.

        Returns:
            List of validation error messages (empty if all valid).
        """
        errors = []
        for param_name, value in params.items():
            if param_name not in self.parameters:
                errors.append(f"Unknown parameter: {param_name}")
            elif not self.parameters[param_name].validate(value):
                errors.append(f"Invalid value for parameter {param_name}: {value}")
        return errors

    def get_default_parameters(self) -> dict[str, Any]:
        """Get default values for all parameters.

        Returns:
            Dictionary of parameter name to default value.
        """
        return {name: spec.default for name, spec in self.parameters.items()}

    def copy_with_parameters(self, params: dict[str, Any]) -> "Component":
        """Create a copy of this component with parameters applied.

        This creates a deep copy of the component and updates the internal
        graph constraints according to the parameter mappings.

        Args:
            params: Dictionary of parameter name to value.

        Returns:
            A new Component with parameters applied.

        Raises:
            ValueError: If any parameter is invalid.
        """
        # Validate parameters
        errors = self.validate_parameters(params)
        if errors:
            raise ValueError(f"Parameter validation failed: {'; '.join(errors)}")

        # Merge with defaults
        full_params = self.get_default_parameters()
        full_params.update(params)

        # Deep copy the internal graph
        new_graph = self.internal_graph.copy()

        # Apply parameter mappings
        for mapping in self.parameter_mappings:
            value = full_params.get(mapping.parameter_name)
            if value is None:
                continue

            # Update edge distances
            for edge_id in mapping.edge_ids:
                if edge_id in new_graph.edges:
                    old_edge = new_graph.edges[edge_id]
                    new_graph.edges[edge_id] = Edge(
                        id=old_edge.id,
                        source=old_edge.source,
                        target=old_edge.target,
                        distance=float(value),
                    )

            # Update hyperedge constraints
            for he_id, (n1, n2) in mapping.hyperedge_constraints:
                if he_id in new_graph.hyperedges:
                    old_he = new_graph.hyperedges[he_id]
                    # Update the specific constraint
                    key = (min(n1, n2), max(n1, n2))
                    if key in old_he.constraints:
                        new_constraints = dict(old_he.constraints)
                        new_constraints[key] = float(value)
                        new_graph.hyperedges[he_id] = Hyperedge(
                            id=old_he.id,
                            nodes=old_he.nodes,
                            constraints=new_constraints,
                            name=old_he.name,
                        )

        return Component(
            id=self.id,
            internal_graph=new_graph,
            ports=dict(self.ports),
            parameters=dict(self.parameters),
            parameter_mappings=list(self.parameter_mappings),
            name=self.name,
        )

    def __hash__(self) -> int:
        """Hash by id."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality by id."""
        if isinstance(other, Component):
            return self.id == other.id
        return False
