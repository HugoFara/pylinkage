"""Tests for the serialization module."""

import json
import tempfile
from pathlib import Path

from pylinkage.assur import (
    Edge,
    JointType,
    LinkageGraph,
    Node,
    NodeRole,
    graph_from_dict,
    graph_from_json,
    graph_to_dict,
    graph_to_json,
)


class TestGraphSerialization:
    """Tests for graph serialization functions."""

    def test_graph_to_dict(self):
        """Test converting a graph to dictionary."""
        graph = LinkageGraph(name="Test")
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("B", role=NodeRole.DRIVER, position=(1.0, 0.0), angle=0.5))
        graph.add_edge(Edge("AB", source="A", target="B", distance=1.0))

        data = graph_to_dict(graph)

        assert data["name"] == "Test"
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

        # Check node data
        node_a = next(n for n in data["nodes"] if n["id"] == "A")
        assert node_a["role"] == "GROUND"
        assert node_a["position"] == [0.0, 0.0]

        node_b = next(n for n in data["nodes"] if n["id"] == "B")
        assert node_b["role"] == "DRIVER"
        assert node_b["angle"] == 0.5

        # Check edge data
        edge_ab = data["edges"][0]
        assert edge_ab["source"] == "A"
        assert edge_ab["target"] == "B"
        assert edge_ab["distance"] == 1.0

    def test_graph_from_dict(self):
        """Test creating a graph from dictionary."""
        data = {
            "name": "Restored",
            "nodes": [
                {
                    "id": "A",
                    "joint_type": "REVOLUTE",
                    "role": "GROUND",
                    "position": [0.0, 0.0],
                    "angle": None,
                    "name": "A"
                },
                {
                    "id": "B",
                    "joint_type": "REVOLUTE",
                    "role": "DRIVER",
                    "position": [1.0, 0.0],
                    "angle": 0.5,
                    "name": "B"
                }
            ],
            "edges": [
                {
                    "id": "AB",
                    "source": "A",
                    "target": "B",
                    "distance": 1.0,
                    "body_id": None
                }
            ]
        }

        graph = graph_from_dict(data)

        assert graph.name == "Restored"
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

        assert graph.nodes["A"].role == NodeRole.GROUND
        assert graph.nodes["B"].role == NodeRole.DRIVER
        assert graph.nodes["B"].angle == 0.5

    def test_roundtrip_dict(self):
        """Test that dict roundtrip preserves data."""
        original = LinkageGraph(name="Four-bar")
        original.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        original.add_node(Node("D", role=NodeRole.GROUND, position=(3.0, 0.0)))
        original.add_node(Node("B", role=NodeRole.DRIVER, position=(0.0, 1.0), angle=0.31))
        original.add_node(Node("C", role=NodeRole.DRIVEN, position=(3.0, 2.0)))
        original.add_edge(Edge("AB", source="A", target="B", distance=1.0))
        original.add_edge(Edge("BC", source="B", target="C", distance=3.0))
        original.add_edge(Edge("CD", source="C", target="D", distance=1.0))

        # Roundtrip
        data = graph_to_dict(original)
        restored = graph_from_dict(data)

        assert restored.name == original.name
        assert len(restored.nodes) == len(original.nodes)
        assert len(restored.edges) == len(original.edges)

        for node_id in original.nodes:
            assert node_id in restored.nodes
            assert restored.nodes[node_id].role == original.nodes[node_id].role
            assert restored.nodes[node_id].position == original.nodes[node_id].position

    def test_json_serialization(self):
        """Test JSON serialization to/from file."""
        graph = LinkageGraph(name="JSON-Test")
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("B", role=NodeRole.DRIVER, position=(1.0, 0.0), angle=0.5))
        graph.add_edge(Edge("AB", source="A", target="B", distance=1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_graph.json"

            # Save
            graph_to_json(graph, path)

            # Verify file exists and is valid JSON
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["name"] == "JSON-Test"

            # Load
            restored = graph_from_json(path)

            assert restored.name == graph.name
            assert len(restored.nodes) == len(graph.nodes)
            assert len(restored.edges) == len(graph.edges)

    def test_json_roundtrip(self):
        """Test complete JSON roundtrip preserves data."""
        original = LinkageGraph(name="Four-bar")
        original.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        original.add_node(Node("D", role=NodeRole.GROUND, position=(3.0, 0.0)))
        original.add_node(
            Node(
                "B",
                role=NodeRole.DRIVER,
                position=(0.0, 1.0),
                angle=0.31,
                joint_type=JointType.REVOLUTE
            )
        )
        original.add_node(Node("C", role=NodeRole.DRIVEN, position=(3.0, 2.0)))
        original.add_edge(Edge("AB", source="A", target="B", distance=1.0))
        original.add_edge(Edge("BC", source="B", target="C", distance=3.0))
        original.add_edge(Edge("CD", source="C", target="D", distance=1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "four_bar.json"

            graph_to_json(original, path)
            restored = graph_from_json(path)

            # Verify structure
            assert restored.name == original.name
            assert set(restored.nodes.keys()) == set(original.nodes.keys())
            assert set(restored.edges.keys()) == set(original.edges.keys())

            # Verify node properties
            for node_id in original.nodes:
                orig_node = original.nodes[node_id]
                rest_node = restored.nodes[node_id]
                assert rest_node.role == orig_node.role
                assert rest_node.joint_type == orig_node.joint_type
                assert rest_node.position == orig_node.position
                assert rest_node.angle == orig_node.angle

            # Verify edge properties
            for edge_id in original.edges:
                orig_edge = original.edges[edge_id]
                rest_edge = restored.edges[edge_id]
                assert rest_edge.source == orig_edge.source
                assert rest_edge.target == orig_edge.target
                assert rest_edge.distance == orig_edge.distance
