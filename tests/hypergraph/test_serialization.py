"""Tests for hypergraph serialization."""

import json
import tempfile
from pathlib import Path

import pytest

from pylinkage.hypergraph import (
    Component,
    ComponentInstance,
    Connection,
    Edge,
    HierarchicalLinkage,
    Hyperedge,
    HypergraphLinkage,
    JointType,
    Node,
    NodeRole,
    ParameterMapping,
    ParameterSpec,
    Port,
    component_from_dict,
    component_to_dict,
    graph_from_dict,
    graph_from_json,
    graph_to_dict,
    graph_to_json,
    hierarchical_from_dict,
    hierarchical_to_dict,
)


class TestNodeSerialization:
    """Tests for node serialization."""

    def test_node_roundtrip(self):
        """Test node serialization roundtrip."""
        from pylinkage.hypergraph.serialization import node_from_dict, node_to_dict

        node = Node(
            id="test",
            position=(1.0, 2.0),
            role=NodeRole.DRIVER,
            joint_type=JointType.PRISMATIC,
            angle=0.5,
            name="Test Node",
        )
        data = node_to_dict(node)
        restored = node_from_dict(data)

        assert restored.id == node.id
        assert restored.position == node.position
        assert restored.role == node.role
        assert restored.joint_type == node.joint_type
        assert restored.angle == node.angle
        assert restored.name == node.name


class TestEdgeSerialization:
    """Tests for edge serialization."""

    def test_edge_roundtrip(self):
        """Test edge serialization roundtrip."""
        from pylinkage.hypergraph.serialization import edge_from_dict, edge_to_dict

        edge = Edge("AB", "A", "B", 1.5)
        data = edge_to_dict(edge)
        restored = edge_from_dict(data)

        assert restored.id == edge.id
        assert restored.source == edge.source
        assert restored.target == edge.target
        assert restored.distance == edge.distance


class TestHyperedgeSerialization:
    """Tests for hyperedge serialization."""

    def test_hyperedge_roundtrip(self):
        """Test hyperedge serialization roundtrip."""
        from pylinkage.hypergraph.serialization import (
            hyperedge_from_dict,
            hyperedge_to_dict,
        )

        he = Hyperedge(
            id="tri",
            nodes=("A", "B", "C"),
            constraints={("A", "B"): 1.0, ("B", "C"): 2.0},
            name="Triangle",
        )
        data = hyperedge_to_dict(he)
        restored = hyperedge_from_dict(data)

        assert restored.id == he.id
        assert set(restored.nodes) == set(he.nodes)
        assert len(restored.constraints) == len(he.constraints)
        assert restored.name == he.name


class TestGraphSerialization:
    """Tests for HypergraphLinkage serialization."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        graph = HypergraphLinkage(name="Sample")
        graph.add_node(Node("A", position=(0, 0), role=NodeRole.GROUND))
        graph.add_node(Node("B", position=(1, 0), role=NodeRole.DRIVER, angle=0.1))
        graph.add_node(Node("C", position=(2, 0), role=NodeRole.DRIVEN))
        graph.add_edge(Edge("AB", "A", "B", 1.0))
        graph.add_hyperedge(
            Hyperedge("he", ("B", "C"), {("B", "C"): 1.5}, name="Link")
        )
        return graph

    def test_graph_to_dict(self, sample_graph):
        """Test converting graph to dict."""
        data = graph_to_dict(sample_graph)
        assert data["name"] == "Sample"
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 1
        assert len(data["hyperedges"]) == 1

    def test_graph_roundtrip(self, sample_graph):
        """Test graph serialization roundtrip."""
        data = graph_to_dict(sample_graph)
        restored = graph_from_dict(data)

        assert restored.name == sample_graph.name
        assert len(restored.nodes) == len(sample_graph.nodes)
        assert len(restored.edges) == len(sample_graph.edges)
        assert len(restored.hyperedges) == len(sample_graph.hyperedges)

    def test_graph_json_roundtrip(self, sample_graph):
        """Test graph JSON file roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            graph_to_json(sample_graph, path)

            restored = graph_from_json(path)

            assert restored.name == sample_graph.name
            assert len(restored.nodes) == len(sample_graph.nodes)


class TestComponentSerialization:
    """Tests for Component serialization."""

    @pytest.fixture
    def sample_component(self):
        """Create a sample component for testing."""
        graph = HypergraphLinkage(name="Internal")
        graph.add_node(Node("A", position=(0, 0), role=NodeRole.GROUND))
        graph.add_node(Node("B", position=(1, 0), role=NodeRole.DRIVEN))
        graph.add_edge(Edge("AB", "A", "B", 1.0))

        return Component(
            id="sample",
            internal_graph=graph,
            ports={"in": Port("in", "A", "Input"), "out": Port("out", "B", "Output")},
            parameters={"length": ParameterSpec("length", "Length", default=1.0)},
            parameter_mappings=[ParameterMapping("length", edge_ids=["AB"])],
            name="Sample Component",
        )

    def test_component_roundtrip(self, sample_component):
        """Test component serialization roundtrip."""
        data = component_to_dict(sample_component)
        restored = component_from_dict(data)

        assert restored.id == sample_component.id
        assert restored.name == sample_component.name
        assert len(restored.ports) == len(sample_component.ports)
        assert len(restored.parameters) == len(sample_component.parameters)
        assert len(restored.parameter_mappings) == len(sample_component.parameter_mappings)


class TestHierarchicalSerialization:
    """Tests for HierarchicalLinkage serialization."""

    @pytest.fixture
    def sample_hierarchical(self):
        """Create a sample hierarchical linkage."""
        graph = HypergraphLinkage()
        graph.add_node(Node("A", position=(0, 0), role=NodeRole.GROUND))
        graph.add_node(Node("B", position=(1, 0), role=NodeRole.DRIVEN))
        graph.add_edge(Edge("AB", "A", "B", 1.0))

        component = Component(
            id="simple",
            internal_graph=graph,
            ports={"in": Port("in", "A"), "out": Port("out", "B")},
        )

        linkage = HierarchicalLinkage(name="Test")
        linkage.add_instance(ComponentInstance("i1", component, {"length": 2.0}))
        linkage.add_instance(ComponentInstance("i2", component))
        linkage.add_connection(Connection("i1", "out", "i2", "in"))

        return linkage

    def test_hierarchical_roundtrip(self, sample_hierarchical):
        """Test hierarchical linkage serialization roundtrip."""
        data = hierarchical_to_dict(sample_hierarchical)
        restored = hierarchical_from_dict(data)

        assert restored.name == sample_hierarchical.name
        assert len(restored.instances) == len(sample_hierarchical.instances)
        assert len(restored.connections) == len(sample_hierarchical.connections)

    def test_hierarchical_json_compatibility(self, sample_hierarchical):
        """Test that hierarchical dict is JSON-compatible."""
        data = hierarchical_to_dict(sample_hierarchical)
        # Should not raise
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = hierarchical_from_dict(restored_data)
        assert restored.name == sample_hierarchical.name
