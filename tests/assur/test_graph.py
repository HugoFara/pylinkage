"""Tests for the graph module."""

import pytest

from pylinkage.assur import Edge, JointType, LinkageGraph, Node, NodeRole


class TestNode:
    """Tests for the Node class."""

    def test_node_creation_defaults(self):
        """Test creating a node with default values."""
        node = Node("A")
        assert node.id == "A"
        assert node.joint_type == JointType.REVOLUTE
        assert node.role == NodeRole.DRIVEN
        assert node.position == (None, None)
        assert node.name == "A"  # Defaults to id

    def test_node_creation_full(self):
        """Test creating a node with all parameters."""
        node = Node(
            id="B",
            joint_type=JointType.PRISMATIC,
            role=NodeRole.DRIVER,
            position=(1.0, 2.0),
            angle=0.5,
            name="Motor",
        )
        assert node.id == "B"
        assert node.joint_type == JointType.PRISMATIC
        assert node.role == NodeRole.DRIVER
        assert node.position == (1.0, 2.0)
        assert node.angle == 0.5
        assert node.name == "Motor"

    def test_node_hash_and_equality(self):
        """Test node hashing and equality."""
        node1 = Node("A", position=(0, 0))
        node2 = Node("A", position=(1, 1))  # Same id, different position
        node3 = Node("B", position=(0, 0))

        assert node1 == node2  # Equal by id
        assert node1 != node3
        assert hash(node1) == hash(node2)

    def test_joint_type_string(self):
        """Test JointType string representation."""
        assert str(JointType.REVOLUTE) == "R"
        assert str(JointType.PRISMATIC) == "P"


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
        """Test edge connects method."""
        edge = Edge("AB", source="A", target="B")
        assert edge.connects("A")
        assert edge.connects("B")
        assert not edge.connects("C")

    def test_edge_other_node(self):
        """Test edge other_node method."""
        edge = Edge("AB", source="A", target="B")
        assert edge.other_node("A") == "B"
        assert edge.other_node("B") == "A"
        with pytest.raises(ValueError):
            edge.other_node("C")

    def test_edge_hash_and_equality(self):
        """Test edge hashing and equality."""
        edge1 = Edge("AB", source="A", target="B")
        edge2 = Edge("AB", source="X", target="Y")  # Same id
        edge3 = Edge("CD", source="A", target="B")

        assert edge1 == edge2  # Equal by id
        assert edge1 != edge3


class TestLinkageGraph:
    """Tests for the LinkageGraph class."""

    def test_graph_creation(self):
        """Test creating an empty graph."""
        graph = LinkageGraph(name="Test")
        assert graph.name == "Test"
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_node(self):
        """Test adding nodes."""
        graph = LinkageGraph()
        node = Node("A", position=(0, 0))
        graph.add_node(node)

        assert "A" in graph.nodes
        assert graph.nodes["A"] == node

    def test_add_node_duplicate_raises(self):
        """Test that adding duplicate node raises ValueError."""
        graph = LinkageGraph()
        graph.add_node(Node("A"))
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(Node("A"))

    def test_add_edge(self):
        """Test adding edges."""
        graph = LinkageGraph()
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))

        edge = Edge("AB", source="A", target="B", distance=1.0)
        graph.add_edge(edge)

        assert "AB" in graph.edges
        assert graph.edges["AB"] == edge

    def test_add_edge_missing_node_raises(self):
        """Test that adding edge with missing nodes raises ValueError."""
        graph = LinkageGraph()
        graph.add_node(Node("A"))

        with pytest.raises(ValueError, match="not found"):
            graph.add_edge(Edge("AB", source="A", target="B"))

    def test_neighbors(self):
        """Test getting node neighbors."""
        graph = LinkageGraph()
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        graph.add_node(Node("C"))
        graph.add_edge(Edge("AB", source="A", target="B"))
        graph.add_edge(Edge("AC", source="A", target="C"))

        neighbors = graph.neighbors("A")
        assert set(neighbors) == {"B", "C"}
        assert graph.neighbors("B") == ["A"]

    def test_get_edge_between(self):
        """Test getting edge between two nodes."""
        graph = LinkageGraph()
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        graph.add_node(Node("C"))
        edge = Edge("AB", source="A", target="B", distance=1.0)
        graph.add_edge(edge)

        assert graph.get_edge_between("A", "B") == edge
        assert graph.get_edge_between("B", "A") == edge  # Order independent
        assert graph.get_edge_between("A", "C") is None

    def test_node_role_filters(self):
        """Test filtering nodes by role."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("B", role=NodeRole.DRIVER))
        graph.add_node(Node("C", role=NodeRole.DRIVEN))
        graph.add_node(Node("D", role=NodeRole.GROUND))

        grounds = graph.ground_nodes()
        assert len(grounds) == 2
        assert all(n.role == NodeRole.GROUND for n in grounds)

        drivers = graph.driver_nodes()
        assert len(drivers) == 1
        assert drivers[0].id == "B"

        driven = graph.driven_nodes()
        assert len(driven) == 1
        assert driven[0].id == "C"

    def test_degree(self):
        """Test getting node degree."""
        graph = LinkageGraph()
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        graph.add_node(Node("C"))
        graph.add_edge(Edge("AB", source="A", target="B"))
        graph.add_edge(Edge("AC", source="A", target="C"))

        assert graph.degree("A") == 2
        assert graph.degree("B") == 1
        assert graph.degree("C") == 1

    def test_remove_node(self):
        """Test removing a node."""
        graph = LinkageGraph()
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        graph.add_edge(Edge("AB", source="A", target="B"))

        removed = graph.remove_node("A")
        assert removed.id == "A"
        assert "A" not in graph.nodes
        assert "AB" not in graph.edges  # Edge should be removed too

    def test_remove_edge(self):
        """Test removing an edge."""
        graph = LinkageGraph()
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        graph.add_edge(Edge("AB", source="A", target="B"))

        removed = graph.remove_edge("AB")
        assert removed.id == "AB"
        assert "AB" not in graph.edges
        assert "A" in graph.nodes  # Node should remain
        assert "B" in graph.nodes

    def test_copy(self):
        """Test deep copying a graph."""
        graph = LinkageGraph(name="Original")
        graph.add_node(Node("A", position=(0, 0)))
        graph.add_node(Node("B", position=(1, 0)))
        graph.add_edge(Edge("AB", source="A", target="B", distance=1.0))

        copy = graph.copy()

        assert copy.name == graph.name
        assert len(copy.nodes) == len(graph.nodes)
        assert len(copy.edges) == len(graph.edges)

        # Modify copy shouldn't affect original
        copy.nodes["A"].position = (5, 5)
        assert graph.nodes["A"].position == (0, 0)

    def test_contains(self):
        """Test __contains__ method."""
        graph = LinkageGraph()
        node = Node("A")
        graph.add_node(node)

        assert "A" in graph
        assert node in graph
        assert "B" not in graph

    def test_len(self):
        """Test __len__ method."""
        graph = LinkageGraph()
        assert len(graph) == 0

        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        assert len(graph) == 2
