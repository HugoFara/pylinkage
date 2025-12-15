"""Tests for hypergraph core elements (Node, Edge, Hyperedge) - topology only."""

import pytest

from pylinkage.hypergraph import Edge, Hyperedge, JointType, Node, NodeRole


class TestNode:
    """Tests for the Node class (topology only)."""

    def test_node_creation_defaults(self):
        """Test creating a node with default values."""
        node = Node("A")
        assert node.id == "A"
        assert node.role == NodeRole.DRIVEN
        assert node.joint_type == JointType.REVOLUTE
        assert node.name == "A"

    def test_node_creation_full(self):
        """Test creating a node with all parameters."""
        node = Node(
            id="B",
            role=NodeRole.DRIVER,
            joint_type=JointType.PRISMATIC,
            name="Driver B",
        )
        assert node.id == "B"
        assert node.role == NodeRole.DRIVER
        assert node.joint_type == JointType.PRISMATIC
        assert node.name == "Driver B"

    def test_node_hash_by_id(self):
        """Test that nodes hash by their ID."""
        node1 = Node("A", role=NodeRole.GROUND)
        node2 = Node("A", role=NodeRole.DRIVER)  # Same ID, different role
        assert hash(node1) == hash(node2)

    def test_node_equality_by_id(self):
        """Test that nodes are equal if they have the same ID."""
        node1 = Node("A", role=NodeRole.GROUND)
        node2 = Node("A", role=NodeRole.DRIVER)  # Same ID, different role
        assert node1 == node2

    def test_node_inequality(self):
        """Test that nodes with different IDs are not equal."""
        node1 = Node("A")
        node2 = Node("B")
        assert node1 != node2

    def test_node_default_name_from_id(self):
        """Test that default name is set from id."""
        node = Node("my_joint")
        assert node.name == "my_joint"

    def test_node_custom_name(self):
        """Test that custom name is used."""
        node = Node("A", name="Custom Name")
        assert node.name == "Custom Name"

    def test_node_ground_role(self):
        """Test creating a ground node."""
        node = Node("ground", role=NodeRole.GROUND)
        assert node.role == NodeRole.GROUND

    def test_node_driver_role(self):
        """Test creating a driver node."""
        node = Node("driver", role=NodeRole.DRIVER)
        assert node.role == NodeRole.DRIVER

    def test_node_prismatic_type(self):
        """Test creating a prismatic joint node."""
        node = Node("slider", joint_type=JointType.PRISMATIC)
        assert node.joint_type == JointType.PRISMATIC


class TestEdge:
    """Tests for the Edge class (topology only)."""

    def test_edge_creation(self):
        """Test creating an edge."""
        edge = Edge("AB", source="A", target="B")
        assert edge.id == "AB"
        assert edge.source == "A"
        assert edge.target == "B"

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

    def test_edge_equality_by_id(self):
        """Test that edges are equal if they have the same ID."""
        edge1 = Edge("AB", source="A", target="B")
        edge2 = Edge("AB", source="X", target="Y")  # Same ID
        assert edge1 == edge2

    def test_edge_inequality(self):
        """Test that edges with different IDs are not equal."""
        edge1 = Edge("AB", source="A", target="B")
        edge2 = Edge("CD", source="A", target="B")
        assert edge1 != edge2


class TestHyperedge:
    """Tests for the Hyperedge class (topology only)."""

    def test_hyperedge_creation(self):
        """Test creating a hyperedge."""
        he = Hyperedge(
            id="triangle",
            nodes=("A", "B", "C"),
        )
        assert he.id == "triangle"
        assert set(he.nodes) == {"A", "B", "C"}

    def test_hyperedge_with_name(self):
        """Test creating a hyperedge with a name."""
        he = Hyperedge(
            id="tri",
            nodes=("A", "B", "C"),
            name="Triangle Link",
        )
        assert he.name == "Triangle Link"

    def test_hyperedge_default_name(self):
        """Test that default name is None."""
        he = Hyperedge(id="test", nodes=("A", "B"))
        assert he.name is None

    def test_hyperedge_to_edges(self):
        """Test converting hyperedge to edges."""
        he = Hyperedge(
            id="tri",
            nodes=("A", "B", "C"),
        )
        edges = he.to_edges()
        assert len(edges) == 2
        edge_ids = {e.id for e in edges}
        assert "tri_A_B" in edge_ids
        assert "tri_B_C" in edge_ids

    def test_hyperedge_to_edges_with_prefix(self):
        """Test converting hyperedge to edges with prefix."""
        he = Hyperedge(id="tri", nodes=("A", "B"))
        edges = he.to_edges(prefix="leg_")
        assert len(edges) == 1
        assert edges[0].id == "leg_tri_A_B"

    def test_hyperedge_from_edges(self):
        """Test creating hyperedge from edges."""
        edges = [
            Edge("e1", "A", "B"),
            Edge("e2", "B", "C"),
        ]
        he = Hyperedge.from_edges(edges, "combined")
        assert he.id == "combined"
        assert set(he.nodes) == {"A", "B", "C"}

    def test_hyperedge_from_edges_with_name(self):
        """Test creating hyperedge from edges with name."""
        edges = [
            Edge("e1", "A", "B"),
        ]
        he = Hyperedge.from_edges(edges, "edge", name="My Edge")
        assert he.name == "My Edge"

    def test_hyperedge_from_edges_empty_raises(self):
        """Test that from_edges raises for empty list."""
        with pytest.raises(ValueError, match="empty edge list"):
            Hyperedge.from_edges([], "empty")

    def test_hyperedge_hash_by_id(self):
        """Test that hyperedges hash by their ID."""
        he1 = Hyperedge("tri", nodes=("A", "B", "C"))
        he2 = Hyperedge("tri", nodes=("X", "Y"))  # Same ID
        assert hash(he1) == hash(he2)

    def test_hyperedge_equality_by_id(self):
        """Test that hyperedges are equal if they have the same ID."""
        he1 = Hyperedge("tri", nodes=("A", "B", "C"))
        he2 = Hyperedge("tri", nodes=("X", "Y"))  # Same ID
        assert he1 == he2

    def test_hyperedge_inequality(self):
        """Test that hyperedges with different IDs are not equal."""
        he1 = Hyperedge("tri1", nodes=("A", "B"))
        he2 = Hyperedge("tri2", nodes=("A", "B"))
        assert he1 != he2
