"""Tests for HypergraphLinkage class."""

import pytest

from pylinkage.hypergraph import (
    Edge,
    Hyperedge,
    HypergraphLinkage,
    Node,
    NodeRole,
)


class TestHypergraphLinkage:
    """Tests for the HypergraphLinkage class."""

    def test_graph_creation_empty(self):
        """Test creating an empty graph."""
        graph = HypergraphLinkage(name="Test")
        assert graph.name == "Test"
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert len(graph.hyperedges) == 0

    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = HypergraphLinkage()
        node = Node("A", position=(0, 0), role=NodeRole.GROUND)
        graph.add_node(node)
        assert "A" in graph.nodes
        assert graph.nodes["A"] == node

    def test_add_node_duplicate_raises(self):
        """Test that adding duplicate node raises."""
        graph = HypergraphLinkage()
        graph.add_node(Node("A"))
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(Node("A"))

    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = HypergraphLinkage()
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        edge = Edge("AB", "A", "B", 1.0)
        graph.add_edge(edge)
        assert "AB" in graph.edges

    def test_add_edge_missing_node_raises(self):
        """Test that adding edge with missing node raises."""
        graph = HypergraphLinkage()
        graph.add_node(Node("A"))
        with pytest.raises(ValueError, match="not found"):
            graph.add_edge(Edge("AB", "A", "B"))

    def test_add_hyperedge(self):
        """Test adding hyperedges to graph."""
        graph = HypergraphLinkage()
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        graph.add_node(Node("C"))
        he = Hyperedge("tri", ("A", "B", "C"), {("A", "B"): 1.0})
        graph.add_hyperedge(he)
        assert "tri" in graph.hyperedges

    def test_add_hyperedge_missing_node_raises(self):
        """Test that adding hyperedge with missing node raises."""
        graph = HypergraphLinkage()
        graph.add_node(Node("A"))
        with pytest.raises(ValueError, match="not found"):
            graph.add_hyperedge(Hyperedge("test", ("A", "B"), {}))

    def test_remove_node(self):
        """Test removing nodes from graph."""
        graph = HypergraphLinkage()
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        graph.add_edge(Edge("AB", "A", "B"))

        removed = graph.remove_node("A")
        assert removed.id == "A"
        assert "A" not in graph.nodes
        assert "AB" not in graph.edges  # Edge should be removed too

    def test_remove_edge(self):
        """Test removing edges from graph."""
        graph = HypergraphLinkage()
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        graph.add_edge(Edge("AB", "A", "B"))

        removed = graph.remove_edge("AB")
        assert removed.id == "AB"
        assert "AB" not in graph.edges

    def test_remove_hyperedge(self):
        """Test removing hyperedges from graph."""
        graph = HypergraphLinkage()
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        graph.add_hyperedge(Hyperedge("he", ("A", "B"), {("A", "B"): 1.0}))

        removed = graph.remove_hyperedge("he")
        assert removed.id == "he"
        assert "he" not in graph.hyperedges

    def test_neighbors(self):
        """Test getting edge neighbors."""
        graph = HypergraphLinkage()
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        graph.add_node(Node("C"))
        graph.add_edge(Edge("AB", "A", "B"))
        graph.add_edge(Edge("AC", "A", "C"))

        neighbors = graph.neighbors("A")
        assert set(neighbors) == {"B", "C"}

    def test_hyperedge_neighbors(self):
        """Test getting hyperedge neighbors."""
        graph = HypergraphLinkage()
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        graph.add_node(Node("C"))
        graph.add_hyperedge(Hyperedge("he", ("A", "B", "C"), {}))

        neighbors = graph.hyperedge_neighbors("A")
        assert set(neighbors) == {"B", "C"}

    def test_get_edge_between(self):
        """Test getting edge between two nodes."""
        graph = HypergraphLinkage()
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        graph.add_node(Node("C"))
        graph.add_edge(Edge("AB", "A", "B", 1.0))

        edge = graph.get_edge_between("A", "B")
        assert edge is not None
        assert edge.id == "AB"

        edge = graph.get_edge_between("A", "C")
        assert edge is None

    def test_ground_driver_driven_nodes(self):
        """Test filtering nodes by role."""
        graph = HypergraphLinkage()
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("B", role=NodeRole.DRIVER))
        graph.add_node(Node("C", role=NodeRole.DRIVEN))

        ground = graph.ground_nodes()
        assert len(ground) == 1
        assert ground[0].id == "A"

        drivers = graph.driver_nodes()
        assert len(drivers) == 1
        assert drivers[0].id == "B"

        driven = graph.driven_nodes()
        assert len(driven) == 1
        assert driven[0].id == "C"

    def test_degree(self):
        """Test edge degree calculation."""
        graph = HypergraphLinkage()
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        graph.add_node(Node("C"))
        graph.add_edge(Edge("AB", "A", "B"))
        graph.add_edge(Edge("AC", "A", "C"))

        assert graph.degree("A") == 2
        assert graph.degree("B") == 1
        assert graph.degree("C") == 1

    def test_to_simple_graph(self):
        """Test converting hypergraph to simple graph."""
        graph = HypergraphLinkage(name="Test")
        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        graph.add_node(Node("C"))
        graph.add_hyperedge(
            Hyperedge("tri", ("A", "B", "C"), {("A", "B"): 1.0, ("B", "C"): 2.0})
        )

        simple = graph.to_simple_graph()
        assert len(simple.hyperedges) == 0
        assert len(simple.edges) == 2

    def test_copy(self):
        """Test deep copying graph."""
        graph = HypergraphLinkage(name="Original")
        graph.add_node(Node("A", position=(0, 0)))
        graph.add_node(Node("B", position=(1, 0)))
        graph.add_edge(Edge("AB", "A", "B", 1.0))

        copy = graph.copy()
        assert copy.name == "Original"
        assert "A" in copy.nodes
        assert "AB" in copy.edges

        # Modify original, verify copy is independent
        graph.nodes["A"].position = (5, 5)  # type: ignore
        assert copy.nodes["A"].position == (0, 0)

    def test_contains(self):
        """Test __contains__ method."""
        graph = HypergraphLinkage()
        node = Node("A")
        graph.add_node(node)

        assert "A" in graph
        assert node in graph
        assert "B" not in graph

    def test_len(self):
        """Test __len__ returns node count."""
        graph = HypergraphLinkage()
        assert len(graph) == 0

        graph.add_node(Node("A"))
        graph.add_node(Node("B"))
        assert len(graph) == 2
