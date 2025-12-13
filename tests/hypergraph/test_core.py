"""Tests for hypergraph core elements (Node, Edge, Hyperedge)."""

import pytest

from pylinkage.hypergraph import Edge, Hyperedge, JointType, Node, NodeRole


class TestNode:
    """Tests for the Node class."""

    def test_node_creation_defaults(self):
        """Test creating a node with default values."""
        node = Node("A")
        assert node.id == "A"
        assert node.position == (None, None)
        assert node.role == NodeRole.DRIVEN
        assert node.joint_type == JointType.REVOLUTE
        assert node.name == "A"

    def test_node_creation_full(self):
        """Test creating a node with all parameters."""
        node = Node(
            id="B",
            position=(1.0, 2.0),
            role=NodeRole.DRIVER,
            joint_type=JointType.PRISMATIC,
            angle=0.5,
            initial_angle=0.0,
            name="Driver B",
        )
        assert node.id == "B"
        assert node.position == (1.0, 2.0)
        assert node.role == NodeRole.DRIVER
        assert node.joint_type == JointType.PRISMATIC
        assert node.angle == 0.5
        assert node.initial_angle == 0.0
        assert node.name == "Driver B"

    def test_node_hash_by_id(self):
        """Test that nodes hash by their ID."""
        node1 = Node("A", position=(0, 0))
        node2 = Node("A", position=(1, 1))  # Same ID, different position
        assert hash(node1) == hash(node2)

    def test_node_equality_by_id(self):
        """Test that nodes are equal if they have the same ID."""
        node1 = Node("A", position=(0, 0))
        node2 = Node("A", position=(1, 1))
        assert node1 == node2

    def test_node_inequality(self):
        """Test that nodes with different IDs are not equal."""
        node1 = Node("A")
        node2 = Node("B")
        assert node1 != node2


class TestEdge:
    """Tests for the Edge class."""

    def test_edge_creation(self):
        """Test creating an edge."""
        edge = Edge("AB", source="A", target="B", distance=1.5)
        assert edge.id == "AB"
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.distance == 1.5

    def test_edge_connects(self):
        """Test the connects method."""
        edge = Edge("AB", source="A", target="B")
        assert edge.connects("A")
        assert edge.connects("B")
        assert not edge.connects("C")

    def test_edge_other_node(self):
        """Test the other_node method."""
        edge = Edge("AB", source="A", target="B")
        assert edge.other_node("A") == "B"
        assert edge.other_node("B") == "A"

    def test_edge_other_node_invalid(self):
        """Test other_node raises for non-connected node."""
        edge = Edge("AB", source="A", target="B")
        with pytest.raises(ValueError, match="not connected"):
            edge.other_node("C")

    def test_edge_hash_by_id(self):
        """Test that edges hash by their ID."""
        edge1 = Edge("AB", source="A", target="B")
        edge2 = Edge("AB", source="X", target="Y")  # Same ID
        assert hash(edge1) == hash(edge2)


class TestHyperedge:
    """Tests for the Hyperedge class."""

    def test_hyperedge_creation(self):
        """Test creating a hyperedge."""
        he = Hyperedge(
            id="triangle",
            nodes=("A", "B", "C"),
            constraints={("A", "B"): 1.0, ("B", "C"): 1.5, ("A", "C"): 2.0},
        )
        assert he.id == "triangle"
        assert set(he.nodes) == {"A", "B", "C"}
        assert len(he.constraints) == 3

    def test_hyperedge_constraint_normalization(self):
        """Test that constraints are normalized to sorted tuples."""
        he = Hyperedge(
            id="test",
            nodes=("A", "B"),
            constraints={("B", "A"): 1.0},  # Reversed order
        )
        # Should be normalized to ("A", "B")
        assert ("A", "B") in he.constraints
        assert he.constraints[("A", "B")] == 1.0

    def test_hyperedge_duplicate_constraint_raises(self):
        """Test that duplicate constraints raise an error."""
        with pytest.raises(ValueError, match="Duplicate constraint"):
            Hyperedge(
                id="test",
                nodes=("A", "B"),
                constraints={("A", "B"): 1.0, ("B", "A"): 2.0},
            )

    def test_hyperedge_invalid_node_raises(self):
        """Test that constraints referencing non-existent nodes raise."""
        with pytest.raises(ValueError, match="not in hyperedge nodes"):
            Hyperedge(
                id="test",
                nodes=("A", "B"),
                constraints={("A", "C"): 1.0},  # C not in nodes
            )

    def test_hyperedge_to_edges(self):
        """Test converting hyperedge to edges."""
        he = Hyperedge(
            id="tri",
            nodes=("A", "B", "C"),
            constraints={("A", "B"): 1.0, ("B", "C"): 2.0},
        )
        edges = he.to_edges()
        assert len(edges) == 2
        edge_ids = {e.id for e in edges}
        assert "tri_A_B" in edge_ids
        assert "tri_B_C" in edge_ids

    def test_hyperedge_from_edges(self):
        """Test creating hyperedge from edges."""
        edges = [
            Edge("e1", "A", "B", 1.0),
            Edge("e2", "B", "C", 2.0),
        ]
        he = Hyperedge.from_edges(edges, "combined")
        assert he.id == "combined"
        assert set(he.nodes) == {"A", "B", "C"}
        assert len(he.constraints) == 2

    def test_hyperedge_from_edges_empty_raises(self):
        """Test that from_edges raises for empty list."""
        with pytest.raises(ValueError, match="empty edge list"):
            Hyperedge.from_edges([], "empty")

    def test_hyperedge_get_distance(self):
        """Test getting distance between nodes."""
        he = Hyperedge(
            id="test",
            nodes=("A", "B", "C"),
            constraints={("A", "B"): 1.0, ("B", "C"): 2.0},
        )
        assert he.get_distance("A", "B") == 1.0
        assert he.get_distance("B", "A") == 1.0  # Order doesn't matter
        assert he.get_distance("A", "C") is None  # Not specified
