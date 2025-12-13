"""Tests for hierarchical linkage composition."""

import pytest

from pylinkage.hypergraph import (
    Component,
    ComponentInstance,
    Connection,
    Edge,
    HierarchicalLinkage,
    HypergraphLinkage,
    Node,
    NodeRole,
    ParameterMapping,
    ParameterSpec,
    Port,
)


@pytest.fixture
def simple_component():
    """Create a simple two-node component."""
    graph = HypergraphLinkage(name="Simple")
    graph.add_node(Node("A", position=(0, 0), role=NodeRole.GROUND))
    graph.add_node(Node("B", position=(1, 0), role=NodeRole.DRIVEN))
    graph.add_edge(Edge("AB", "A", "B", 1.0))

    return Component(
        id="simple",
        internal_graph=graph,
        ports={
            "input": Port("input", "A", "Input"),
            "output": Port("output", "B", "Output"),
        },
        parameters={
            "length": ParameterSpec("length", default=1.0),
        },
        parameter_mappings=[
            ParameterMapping("length", edge_ids=["AB"]),
        ],
        name="Simple",
    )


class TestComponentInstance:
    """Tests for ComponentInstance class."""

    def test_instance_creation(self, simple_component):
        """Test creating a component instance."""
        instance = ComponentInstance(
            id="inst1",
            component=simple_component,
            parameters={"length": 2.0},
            name="Instance 1",
        )
        assert instance.id == "inst1"
        assert instance.component == simple_component
        assert instance.parameters == {"length": 2.0}
        assert instance.name == "Instance 1"

    def test_get_qualified_node_id(self, simple_component):
        """Test getting qualified node IDs."""
        instance = ComponentInstance("inst1", simple_component)
        assert instance.get_qualified_node_id("A") == "inst1.A"
        assert instance.get_qualified_node_id("B") == "inst1.B"

    def test_get_port_qualified_node(self, simple_component):
        """Test getting qualified node for port."""
        instance = ComponentInstance("inst1", simple_component)
        assert instance.get_port_qualified_node("input") == "inst1.A"
        assert instance.get_port_qualified_node("output") == "inst1.B"


class TestConnection:
    """Tests for Connection class."""

    def test_connection_creation(self):
        """Test creating a connection."""
        conn = Connection("inst1", "output", "inst2", "input")
        assert conn.from_instance == "inst1"
        assert conn.from_port == "output"
        assert conn.to_instance == "inst2"
        assert conn.to_port == "input"

    def test_connection_self_loop_raises(self):
        """Test that connecting a port to itself raises."""
        with pytest.raises(ValueError, match="itself"):
            Connection("inst1", "output", "inst1", "output")


class TestHierarchicalLinkage:
    """Tests for HierarchicalLinkage class."""

    def test_hierarchical_creation_empty(self):
        """Test creating an empty hierarchical linkage."""
        linkage = HierarchicalLinkage(name="Empty")
        assert linkage.name == "Empty"
        assert len(linkage.instances) == 0
        assert len(linkage.connections) == 0

    def test_add_instance(self, simple_component):
        """Test adding instances."""
        linkage = HierarchicalLinkage()
        instance = ComponentInstance("inst1", simple_component)
        linkage.add_instance(instance)
        assert "inst1" in linkage.instances

    def test_add_instance_duplicate_raises(self, simple_component):
        """Test that adding duplicate instance raises."""
        linkage = HierarchicalLinkage()
        linkage.add_instance(ComponentInstance("inst1", simple_component))
        with pytest.raises(ValueError, match="already exists"):
            linkage.add_instance(ComponentInstance("inst1", simple_component))

    def test_add_connection(self, simple_component):
        """Test adding connections."""
        linkage = HierarchicalLinkage()
        linkage.add_instance(ComponentInstance("inst1", simple_component))
        linkage.add_instance(ComponentInstance("inst2", simple_component))

        conn = Connection("inst1", "output", "inst2", "input")
        linkage.add_connection(conn)
        assert len(linkage.connections) == 1

    def test_add_connection_invalid_instance_raises(self, simple_component):
        """Test that connection to invalid instance raises."""
        linkage = HierarchicalLinkage()
        linkage.add_instance(ComponentInstance("inst1", simple_component))

        with pytest.raises(ValueError, match="unknown instance"):
            linkage.add_connection(
                Connection("inst1", "output", "nonexistent", "input")
            )

    def test_add_connection_invalid_port_raises(self, simple_component):
        """Test that connection to invalid port raises."""
        linkage = HierarchicalLinkage()
        linkage.add_instance(ComponentInstance("inst1", simple_component))
        linkage.add_instance(ComponentInstance("inst2", simple_component))

        with pytest.raises(ValueError, match="unknown port"):
            linkage.add_connection(
                Connection("inst1", "nonexistent_port", "inst2", "input")
            )

    def test_flatten_single_instance(self, simple_component):
        """Test flattening with a single instance."""
        linkage = HierarchicalLinkage(name="Single")
        linkage.add_instance(ComponentInstance("inst1", simple_component))

        flat = linkage.flatten()
        assert flat.name == "Single"
        assert "inst1.A" in flat.nodes
        assert "inst1.B" in flat.nodes
        assert "inst1.AB" in flat.edges

    def test_flatten_with_parameters(self, simple_component):
        """Test flattening applies parameters."""
        linkage = HierarchicalLinkage()
        linkage.add_instance(
            ComponentInstance("inst1", simple_component, parameters={"length": 5.0})
        )

        flat = linkage.flatten()
        assert flat.edges["inst1.AB"].distance == 5.0

    def test_flatten_with_connection(self, simple_component):
        """Test flattening merges connected ports."""
        linkage = HierarchicalLinkage()
        linkage.add_instance(ComponentInstance("inst1", simple_component))
        linkage.add_instance(ComponentInstance("inst2", simple_component))
        linkage.add_connection(Connection("inst1", "output", "inst2", "input"))

        flat = linkage.flatten()

        # inst1.B and inst2.A should be merged
        # The canonical node should be inst1.B (from_port is canonical)
        assert "inst1.B" in flat.nodes
        # inst2.A should not exist as a separate node
        assert "inst2.A" not in flat.nodes

        # inst2's internal edge should reference the merged node
        edge = flat.edges["inst2.AB"]
        assert edge.source == "inst1.B"
        assert edge.target == "inst2.B"

    def test_flatten_chain_of_three(self, simple_component):
        """Test flattening a chain of three components."""
        linkage = HierarchicalLinkage()
        linkage.add_instance(ComponentInstance("inst1", simple_component))
        linkage.add_instance(ComponentInstance("inst2", simple_component))
        linkage.add_instance(ComponentInstance("inst3", simple_component))
        linkage.add_connection(Connection("inst1", "output", "inst2", "input"))
        linkage.add_connection(Connection("inst2", "output", "inst3", "input"))

        flat = linkage.flatten()

        # Should have 4 nodes: inst1.A, inst1.B(=inst2.A), inst2.B(=inst3.A), inst3.B
        assert len(flat.nodes) == 4
        assert "inst1.A" in flat.nodes
        assert "inst1.B" in flat.nodes
        assert "inst2.B" in flat.nodes
        assert "inst3.B" in flat.nodes

    def test_get_all_port_nodes(self, simple_component):
        """Test getting all port nodes."""
        linkage = HierarchicalLinkage()
        linkage.add_instance(ComponentInstance("inst1", simple_component))
        linkage.add_instance(ComponentInstance("inst2", simple_component))

        ports = linkage.get_all_port_nodes()
        assert ports["inst1.input"] == "inst1.A"
        assert ports["inst1.output"] == "inst1.B"
        assert ports["inst2.input"] == "inst2.A"
        assert ports["inst2.output"] == "inst2.B"
