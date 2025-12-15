"""Tests for the decomposition module (topology only)."""

import pytest

from pylinkage.assur import (
    Edge,
    LinkageGraph,
    Node,
    NodeRole,
    decompose_assur_groups,
    validate_decomposition,
)


class TestDecomposition:
    """Tests for the decomposition algorithm."""

    def test_decompose_four_bar(self):
        """Test decomposing a simple four-bar linkage."""
        graph = LinkageGraph(name="Four-bar")

        # Ground points (topology only - no positions)
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("D", role=NodeRole.GROUND))

        # Driver (crank)
        graph.add_node(Node("B", role=NodeRole.DRIVER))

        # Driven point (coupler)
        graph.add_node(Node("C", role=NodeRole.DRIVEN))

        # Edges (topology only - no distances)
        graph.add_edge(Edge("AB", source="A", target="B"))
        graph.add_edge(Edge("BC", source="B", target="C"))
        graph.add_edge(Edge("CD", source="C", target="D"))

        result = decompose_assur_groups(graph)

        assert result.ground == ["A", "D"]
        assert result.drivers == ["B"]
        assert len(result.groups) == 1
        assert result.groups[0].joint_signature == "RRR"
        assert result.groups[0].internal_nodes == ("C",)

    def test_decompose_with_multiple_driven(self):
        """Test decomposing a linkage with multiple driven joints."""
        graph = LinkageGraph(name="Six-bar")

        # Ground
        graph.add_node(Node("G1", role=NodeRole.GROUND))
        graph.add_node(Node("G2", role=NodeRole.GROUND))

        # Driver
        graph.add_node(Node("D", role=NodeRole.DRIVER))

        # First driven point
        graph.add_node(Node("P1", role=NodeRole.DRIVEN))

        # Second driven point
        graph.add_node(Node("P2", role=NodeRole.DRIVEN))

        # Edges (topology defines connectivity)
        graph.add_edge(Edge("G1-D", source="G1", target="D"))
        graph.add_edge(Edge("D-P1", source="D", target="P1"))
        graph.add_edge(Edge("G2-P1", source="G2", target="P1"))
        graph.add_edge(Edge("P1-P2", source="P1", target="P2"))
        graph.add_edge(Edge("G2-P2", source="G2", target="P2"))

        result = decompose_assur_groups(graph)

        assert len(result.ground) == 2
        assert len(result.drivers) == 1
        assert len(result.groups) == 2

        # First group should be P1 (depends on D and G2)
        # Second group should be P2 (depends on P1 and G2)
        assert "P1" in result.groups[0].internal_nodes
        assert "P2" in result.groups[1].internal_nodes

    def test_decompose_ground_only_raises(self):
        """Test that a graph with only ground nodes has no groups."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("B", role=NodeRole.GROUND))
        graph.add_edge(Edge("AB", source="A", target="B"))

        result = decompose_assur_groups(graph)
        assert len(result.groups) == 0

    def test_decompose_unsolvable_raises(self):
        """Test that unsolvable linkage raises ValueError."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND))
        # Isolated driven node with no connections to known nodes
        graph.add_node(Node("B", role=NodeRole.DRIVEN))
        graph.add_node(Node("C", role=NodeRole.DRIVEN))
        graph.add_edge(Edge("BC", source="B", target="C"))

        with pytest.raises(ValueError, match="Cannot decompose"):
            decompose_assur_groups(graph)


class TestValidateDecomposition:
    """Tests for the validate_decomposition function."""

    def test_validate_valid_decomposition(self):
        """Test validation of a valid decomposition."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("B", role=NodeRole.DRIVER))
        graph.add_edge(Edge("AB", source="A", target="B"))

        result = decompose_assur_groups(graph)
        messages = validate_decomposition(result)

        assert len(messages) == 0

    def test_validate_no_ground(self):
        """Test validation catches missing ground."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.DRIVER))

        result = decompose_assur_groups(graph)
        messages = validate_decomposition(result)

        assert any("ground" in m.lower() for m in messages)

    def test_validate_no_driver(self):
        """Test validation catches missing driver."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND))

        result = decompose_assur_groups(graph)
        messages = validate_decomposition(result)

        assert any("driver" in m.lower() for m in messages)


class TestDecompositionResult:
    """Tests for DecompositionResult properties."""

    def test_decomposition_result_has_graph(self):
        """Test that decomposition result stores the graph reference."""
        graph = LinkageGraph(name="Test")
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("B", role=NodeRole.DRIVER))
        graph.add_edge(Edge("AB", source="A", target="B"))

        result = decompose_assur_groups(graph)

        assert result.graph is graph
        assert result.graph.name == "Test"

    def test_decomposition_result_groups_have_edges(self):
        """Test that groups contain edge references."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("D", role=NodeRole.GROUND))
        graph.add_node(Node("B", role=NodeRole.DRIVER))
        graph.add_node(Node("C", role=NodeRole.DRIVEN))

        graph.add_edge(Edge("AB", source="A", target="B"))
        graph.add_edge(Edge("BC", source="B", target="C"))
        graph.add_edge(Edge("CD", source="C", target="D"))

        result = decompose_assur_groups(graph)

        # The RRR group for C should have edges BC and CD
        assert len(result.groups) == 1
        group = result.groups[0]
        assert len(group.internal_edges) == 2
        assert "BC" in group.internal_edges
        assert "CD" in group.internal_edges
