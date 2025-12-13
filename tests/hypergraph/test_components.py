"""Tests for hypergraph component system."""

import pytest

from pylinkage.hypergraph import (
    Component,
    Edge,
    HypergraphLinkage,
    Node,
    NodeRole,
    ParameterMapping,
    ParameterSpec,
    Port,
)


class TestPort:
    """Tests for the Port class."""

    def test_port_creation(self):
        """Test creating a port."""
        port = Port("input", "node_A", "Input port")
        assert port.id == "input"
        assert port.internal_node == "node_A"
        assert port.description == "Input port"

    def test_port_default_description(self):
        """Test port with default description."""
        port = Port("output", "node_B")
        assert port.description == ""


class TestParameterSpec:
    """Tests for the ParameterSpec class."""

    def test_parameter_spec_creation(self):
        """Test creating a parameter spec."""
        spec = ParameterSpec("length", "Link length", default=1.0)
        assert spec.name == "length"
        assert spec.description == "Link length"
        assert spec.default == 1.0

    def test_parameter_spec_validate_default(self):
        """Test that default validation passes."""
        spec = ParameterSpec("length", default=1.0)
        assert spec.validate(2.0)
        assert spec.validate(0.5)

    def test_parameter_spec_validate_bounds(self):
        """Test validation with bounds."""
        spec = ParameterSpec("length", default=1.0, min_value=0.1, max_value=10.0)
        assert spec.validate(1.0)
        assert spec.validate(0.1)
        assert spec.validate(10.0)
        assert not spec.validate(0.05)
        assert not spec.validate(15.0)

    def test_parameter_spec_custom_validator(self):
        """Test validation with custom validator."""
        spec = ParameterSpec(
            "even_number",
            default=2,
            validator=lambda x: isinstance(x, int) and x % 2 == 0,
        )
        assert spec.validate(2)
        assert spec.validate(4)
        assert not spec.validate(3)
        assert not spec.validate(1.5)


class TestComponent:
    """Tests for the Component class."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph for testing."""
        graph = HypergraphLinkage(name="Simple")
        graph.add_node(Node("A", position=(0, 0), role=NodeRole.GROUND))
        graph.add_node(Node("B", position=(1, 0), role=NodeRole.DRIVEN))
        graph.add_edge(Edge("AB", "A", "B", 1.0))
        return graph

    @pytest.fixture
    def simple_component(self, simple_graph):
        """Create a simple component for testing."""
        return Component(
            id="simple",
            internal_graph=simple_graph,
            ports={
                "base": Port("base", "A", "Base point"),
                "tip": Port("tip", "B", "Tip point"),
            },
            parameters={
                "length": ParameterSpec("length", "Link length", default=1.0),
            },
            parameter_mappings=[
                ParameterMapping("length", edge_ids=["AB"]),
            ],
            name="Simple Link",
        )

    def test_component_creation(self, simple_component):
        """Test creating a component."""
        assert simple_component.id == "simple"
        assert simple_component.name == "Simple Link"
        assert len(simple_component.ports) == 2
        assert len(simple_component.parameters) == 1

    def test_component_invalid_port_raises(self, simple_graph):
        """Test that invalid port internal_node raises."""
        with pytest.raises(ValueError, match="non-existent node"):
            Component(
                id="bad",
                internal_graph=simple_graph,
                ports={"bad": Port("bad", "NONEXISTENT", "")},
            )

    def test_component_get_port_node(self, simple_component):
        """Test getting port node ID."""
        assert simple_component.get_port_node("base") == "A"
        assert simple_component.get_port_node("tip") == "B"

    def test_component_get_default_parameters(self, simple_component):
        """Test getting default parameters."""
        defaults = simple_component.get_default_parameters()
        assert defaults == {"length": 1.0}

    def test_component_validate_parameters(self, simple_component):
        """Test parameter validation."""
        errors = simple_component.validate_parameters({"length": 2.0})
        assert len(errors) == 0

        errors = simple_component.validate_parameters({"unknown": 1.0})
        assert len(errors) == 1
        assert "unknown" in errors[0].lower()

    def test_component_copy_with_parameters(self, simple_component):
        """Test copying component with parameters."""
        new_component = simple_component.copy_with_parameters({"length": 5.0})

        # Original unchanged
        assert simple_component.internal_graph.edges["AB"].distance == 1.0

        # New component has updated distance
        assert new_component.internal_graph.edges["AB"].distance == 5.0

    def test_component_copy_with_invalid_parameters_raises(self, simple_component):
        """Test that invalid parameters raise during copy."""
        with pytest.raises(ValueError, match="validation failed"):
            simple_component.copy_with_parameters({"unknown": 1.0})

    def test_component_equality_by_id(self, simple_graph):
        """Test that components are equal by ID."""
        comp1 = Component(id="same", internal_graph=simple_graph)
        comp2 = Component(id="same", internal_graph=simple_graph.copy())
        assert comp1 == comp2

        comp3 = Component(id="different", internal_graph=simple_graph)
        assert comp1 != comp3
