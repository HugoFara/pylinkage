"""Tests for hierarchical linkage composition (topology only)."""

import pytest

from pylinkage.hypergraph import (
    ComponentInstance,
    Connection,
    Edge,
    HierarchicalLinkage,
    HypergraphLinkage,
    Node,
    NodeRole,
)


@pytest.fixture
def simple_topology():
    """Create a simple two-node topology."""
    graph = HypergraphLinkage(name="Simple")
    graph.add_node(Node("A", role=NodeRole.GROUND))
    graph.add_node(Node("B", role=NodeRole.DRIVEN))
    graph.add_edge(Edge("AB", "A", "B"))
    return graph


@pytest.fixture
def simple_instance(simple_topology):
    """Create a simple component instance."""
    return ComponentInstance(
        id="inst1",
        topology=simple_topology,
        ports={"input": "A", "output": "B"},
        name="Instance 1",
    )


class TestComponentInstance:
    """Tests for ComponentInstance class."""

    def test_instance_creation(self, simple_topology):
        """Test creating a component instance."""
        instance = ComponentInstance(
            id="inst1",
            topology=simple_topology,
            ports={"input": "A", "output": "B"},
            name="Instance 1",
        )
        assert instance.id == "inst1"
        assert instance.topology == simple_topology
        assert instance.ports == {"input": "A", "output": "B"}
        assert instance.name == "Instance 1"

    def test_get_qualified_node_id(self, simple_instance):
        """Test getting qualified node IDs."""
        assert simple_instance.get_qualified_node_id("A") == "inst1.A"
        assert simple_instance.get_qualified_node_id("B") == "inst1.B"

    def test_get_port_qualified_node(self, simple_instance):
        """Test getting qualified node for port."""
        assert simple_instance.get_port_qualified_node("input") == "inst1.A"
        assert simple_instance.get_port_qualified_node("output") == "inst1.B"

    def test_get_port_qualified_node_invalid_raises(self, simple_instance):
        """Test that getting invalid port raises."""
        with pytest.raises(KeyError, match="nonexistent"):
            simple_instance.get_port_qualified_node("nonexistent")


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

    def test_add_instance(self, simple_topology):
        """Test adding instances."""
        linkage = HierarchicalLinkage()
        instance = ComponentInstance(
            id="inst1",
            topology=simple_topology,
            ports={"input": "A", "output": "B"},
        )
        linkage.add_instance(instance)
        assert "inst1" in linkage.instances

    def test_add_instance_duplicate_raises(self, simple_topology):
        """Test that adding duplicate instance raises."""
        linkage = HierarchicalLinkage()
        linkage.add_instance(
            ComponentInstance(
                id="inst1",
                topology=simple_topology,
                ports={"input": "A", "output": "B"},
            )
        )
        with pytest.raises(ValueError, match="already exists"):
            linkage.add_instance(
                ComponentInstance(
                    id="inst1",
                    topology=simple_topology,
                    ports={"input": "A", "output": "B"},
                )
            )

    def test_add_connection(self, simple_topology):
        """Test adding connections."""
        linkage = HierarchicalLinkage()
        linkage.add_instance(
            ComponentInstance(
                id="inst1",
                topology=simple_topology,
                ports={"input": "A", "output": "B"},
            )
        )
        linkage.add_instance(
            ComponentInstance(
                id="inst2",
                topology=simple_topology,
                ports={"input": "A", "output": "B"},
            )
        )

        conn = Connection("inst1", "output", "inst2", "input")
        linkage.add_connection(conn)
        assert len(linkage.connections) == 1

    def test_add_connection_invalid_instance_raises(self, simple_topology):
        """Test that connection to invalid instance raises."""
        linkage = HierarchicalLinkage()
        linkage.add_instance(
            ComponentInstance(
                id="inst1",
                topology=simple_topology,
                ports={"input": "A", "output": "B"},
            )
        )

        with pytest.raises(ValueError, match="unknown instance"):
            linkage.add_connection(Connection("inst1", "output", "nonexistent", "input"))

    def test_add_connection_invalid_port_raises(self, simple_topology):
        """Test that connection to invalid port raises."""
        linkage = HierarchicalLinkage()
        linkage.add_instance(
            ComponentInstance(
                id="inst1",
                topology=simple_topology,
                ports={"input": "A", "output": "B"},
            )
        )
        linkage.add_instance(
            ComponentInstance(
                id="inst2",
                topology=simple_topology,
                ports={"input": "A", "output": "B"},
            )
        )

        with pytest.raises(ValueError, match="unknown port"):
            linkage.add_connection(Connection("inst1", "nonexistent_port", "inst2", "input"))

    def test_flatten_single_instance(self, simple_topology):
        """Test flattening with a single instance."""
        linkage = HierarchicalLinkage(name="Single")
        linkage.add_instance(
            ComponentInstance(
                id="inst1",
                topology=simple_topology,
                ports={"input": "A", "output": "B"},
            )
        )

        flat = linkage.flatten()
        assert flat.name == "Single"
        assert "inst1.A" in flat.nodes
        assert "inst1.B" in flat.nodes
        assert "inst1.AB" in flat.edges

    def test_flatten_with_connection(self, simple_topology):
        """Test flattening merges connected ports."""
        linkage = HierarchicalLinkage()
        linkage.add_instance(
            ComponentInstance(
                id="inst1",
                topology=simple_topology,
                ports={"input": "A", "output": "B"},
            )
        )
        linkage.add_instance(
            ComponentInstance(
                id="inst2",
                topology=simple_topology,
                ports={"input": "A", "output": "B"},
            )
        )
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

    def test_flatten_chain_of_three(self, simple_topology):
        """Test flattening a chain of three components."""
        linkage = HierarchicalLinkage()
        linkage.add_instance(
            ComponentInstance(
                id="inst1",
                topology=simple_topology,
                ports={"input": "A", "output": "B"},
            )
        )
        linkage.add_instance(
            ComponentInstance(
                id="inst2",
                topology=simple_topology,
                ports={"input": "A", "output": "B"},
            )
        )
        linkage.add_instance(
            ComponentInstance(
                id="inst3",
                topology=simple_topology,
                ports={"input": "A", "output": "B"},
            )
        )
        linkage.add_connection(Connection("inst1", "output", "inst2", "input"))
        linkage.add_connection(Connection("inst2", "output", "inst3", "input"))

        flat = linkage.flatten()

        # Should have 4 nodes: inst1.A, inst1.B(=inst2.A), inst2.B(=inst3.A), inst3.B
        assert len(flat.nodes) == 4
        assert "inst1.A" in flat.nodes
        assert "inst1.B" in flat.nodes
        assert "inst2.B" in flat.nodes
        assert "inst3.B" in flat.nodes

    def test_get_all_port_nodes(self, simple_topology):
        """Test getting all port nodes."""
        linkage = HierarchicalLinkage()
        linkage.add_instance(
            ComponentInstance(
                id="inst1",
                topology=simple_topology,
                ports={"input": "A", "output": "B"},
            )
        )
        linkage.add_instance(
            ComponentInstance(
                id="inst2",
                topology=simple_topology,
                ports={"input": "A", "output": "B"},
            )
        )

        ports = linkage.get_all_port_nodes()
        assert ports["inst1.input"] == "inst1.A"
        assert ports["inst1.output"] == "inst1.B"
        assert ports["inst2.input"] == "inst2.A"
        assert ports["inst2.output"] == "inst2.B"
