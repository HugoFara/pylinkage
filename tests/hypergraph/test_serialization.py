"""Tests for hypergraph serialization (topology only)."""

import json
import tempfile
from pathlib import Path

import pytest

from pylinkage.hypergraph import (
    ComponentInstance,
    Connection,
    Edge,
    HierarchicalLinkage,
    Hyperedge,
    HypergraphLinkage,
    JointType,
    Node,
    NodeRole,
    graph_from_dict,
    graph_from_json,
    graph_to_dict,
    graph_to_json,
    hierarchical_from_dict,
    hierarchical_to_dict,
)


class TestNodeSerialization:
    """Tests for node serialization (topology only)."""

    def test_node_roundtrip(self):
        """Test node serialization roundtrip."""
        from pylinkage.hypergraph.serialization import node_from_dict, node_to_dict

        node = Node(
            id="test",
            role=NodeRole.DRIVER,
            joint_type=JointType.PRISMATIC,
            name="Test Node",
        )
        data = node_to_dict(node)
        restored = node_from_dict(data)

        assert restored.id == node.id
        assert restored.role == node.role
        assert restored.joint_type == node.joint_type
        assert restored.name == node.name


class TestEdgeSerialization:
    """Tests for edge serialization (topology only)."""

    def test_edge_roundtrip(self):
        """Test edge serialization roundtrip."""
        from pylinkage.hypergraph.serialization import edge_from_dict, edge_to_dict

        edge = Edge("AB", "A", "B")
        data = edge_to_dict(edge)
        restored = edge_from_dict(data)

        assert restored.id == edge.id
        assert restored.source == edge.source
        assert restored.target == edge.target


class TestHyperedgeSerialization:
    """Tests for hyperedge serialization (topology only)."""

    def test_hyperedge_roundtrip(self):
        """Test hyperedge serialization roundtrip."""
        from pylinkage.hypergraph.serialization import (
            hyperedge_from_dict,
            hyperedge_to_dict,
        )

        he = Hyperedge(
            id="tri",
            nodes=("A", "B", "C"),
            name="Triangle",
        )
        data = hyperedge_to_dict(he)
        restored = hyperedge_from_dict(data)

        assert restored.id == he.id
        assert set(restored.nodes) == set(he.nodes)
        assert restored.name == he.name


class TestGraphSerialization:
    """Tests for HypergraphLinkage serialization (topology only)."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        graph = HypergraphLinkage(name="Sample")
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("B", role=NodeRole.DRIVER))
        graph.add_node(Node("C", role=NodeRole.DRIVEN))
        graph.add_edge(Edge("AB", "A", "B"))
        graph.add_hyperedge(Hyperedge("he", ("B", "C"), name="Link"))
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


class TestHierarchicalSerialization:
    """Tests for HierarchicalLinkage serialization (topology only)."""

    @pytest.fixture
    def sample_hierarchical(self):
        """Create a sample hierarchical linkage."""
        graph = HypergraphLinkage()
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("B", role=NodeRole.DRIVEN))
        graph.add_edge(Edge("AB", "A", "B"))

        linkage = HierarchicalLinkage(name="Test")
        linkage.add_instance(
            ComponentInstance(
                id="i1",
                topology=graph,
                ports={"in": "A", "out": "B"},
            )
        )
        linkage.add_instance(
            ComponentInstance(
                id="i2",
                topology=graph,
                ports={"in": "A", "out": "B"},
            )
        )
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
