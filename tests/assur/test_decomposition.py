"""Tests for the decomposition module."""

import pytest

from pylinkage.assur import (
    Edge,
    LinkageGraph,
    Node,
    NodeRole,
    decompose_assur_groups,
    solve_decomposition,
    validate_decomposition,
)


class TestDecomposition:
    """Tests for the decomposition algorithm."""

    def test_decompose_four_bar(self):
        """Test decomposing a simple four-bar linkage."""
        graph = LinkageGraph(name="Four-bar")

        # Ground points
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("D", role=NodeRole.GROUND, position=(3.0, 0.0)))

        # Driver (crank)
        graph.add_node(Node("B", role=NodeRole.DRIVER, position=(0.0, 1.0), angle=0.31))

        # Driven point (coupler)
        graph.add_node(Node("C", role=NodeRole.DRIVEN, position=(3.0, 2.0)))

        # Edges
        graph.add_edge(Edge("AB", source="A", target="B", distance=1.0))
        graph.add_edge(Edge("BC", source="B", target="C", distance=3.0))
        graph.add_edge(Edge("CD", source="C", target="D", distance=1.0))

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
        graph.add_node(Node("G1", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("G2", role=NodeRole.GROUND, position=(4.0, 0.0)))

        # Driver
        graph.add_node(Node("D", role=NodeRole.DRIVER, position=(0.0, 1.0), angle=0.2))

        # First driven point
        graph.add_node(Node("P1", role=NodeRole.DRIVEN, position=(2.0, 2.0)))

        # Second driven point
        graph.add_node(Node("P2", role=NodeRole.DRIVEN, position=(4.0, 1.0)))

        # Edges
        graph.add_edge(Edge("G1-D", source="G1", target="D", distance=1.0))
        graph.add_edge(Edge("D-P1", source="D", target="P1", distance=2.0))
        graph.add_edge(Edge("G2-P1", source="G2", target="P1", distance=2.5))
        graph.add_edge(Edge("P1-P2", source="P1", target="P2", distance=2.0))
        graph.add_edge(Edge("G2-P2", source="G2", target="P2", distance=1.0))

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
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("B", role=NodeRole.GROUND, position=(1.0, 0.0)))
        graph.add_edge(Edge("AB", source="A", target="B", distance=1.0))

        result = decompose_assur_groups(graph)
        assert len(result.groups) == 0

    def test_decompose_unsolvable_raises(self):
        """Test that unsolvable linkage raises ValueError."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        # Isolated driven node with no connections to known nodes
        graph.add_node(Node("B", role=NodeRole.DRIVEN, position=(1.0, 1.0)))
        graph.add_node(Node("C", role=NodeRole.DRIVEN, position=(2.0, 1.0)))
        graph.add_edge(Edge("BC", source="B", target="C", distance=1.0))

        with pytest.raises(ValueError, match="Cannot decompose"):
            decompose_assur_groups(graph)


class TestValidateDecomposition:
    """Tests for the validate_decomposition function."""

    def test_validate_valid_decomposition(self):
        """Test validation of a valid decomposition."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("B", role=NodeRole.DRIVER, position=(0.0, 1.0), angle=0.1))
        graph.add_edge(Edge("AB", source="A", target="B", distance=1.0))

        result = decompose_assur_groups(graph)
        messages = validate_decomposition(result)

        assert len(messages) == 0

    def test_validate_no_ground(self):
        """Test validation catches missing ground."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.DRIVER, position=(0.0, 0.0), angle=0.1))

        result = decompose_assur_groups(graph)
        messages = validate_decomposition(result)

        assert any("ground" in m.lower() for m in messages)

    def test_validate_no_driver(self):
        """Test validation catches missing driver."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))

        result = decompose_assur_groups(graph)
        messages = validate_decomposition(result)

        assert any("driver" in m.lower() for m in messages)


class TestSolveDecomposition:
    """Tests for the solve_decomposition function."""

    def test_solve_four_bar(self):
        """Test solving a four-bar linkage."""
        graph = LinkageGraph(name="Four-bar")

        # Ground points
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("D", role=NodeRole.GROUND, position=(3.0, 0.0)))

        # Driver
        graph.add_node(Node("B", role=NodeRole.DRIVER, position=(0.0, 1.0), angle=0.31))

        # Driven
        graph.add_node(Node("C", role=NodeRole.DRIVEN, position=(3.0, 2.0)))

        # Edges
        graph.add_edge(Edge("AB", source="A", target="B", distance=1.0))
        graph.add_edge(Edge("BC", source="B", target="C", distance=3.0))
        graph.add_edge(Edge("CD", source="C", target="D", distance=1.0))

        result = decompose_assur_groups(graph)
        positions = solve_decomposition(result)

        # Check all positions are computed
        assert "A" in positions
        assert "B" in positions
        assert "C" in positions
        assert "D" in positions

        # Ground positions should be preserved
        assert positions["A"] == (0.0, 0.0)
        assert positions["D"] == (3.0, 0.0)

    def test_solve_with_initial_positions(self):
        """Test solving with custom initial positions."""
        graph = LinkageGraph(name="Four-bar")

        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("D", role=NodeRole.GROUND, position=(3.0, 0.0)))
        graph.add_node(Node("B", role=NodeRole.DRIVER, position=(0.0, 1.0), angle=0.31))
        graph.add_node(Node("C", role=NodeRole.DRIVEN, position=(3.0, 2.0)))

        graph.add_edge(Edge("AB", source="A", target="B", distance=1.0))
        graph.add_edge(Edge("BC", source="B", target="C", distance=3.0))
        graph.add_edge(Edge("CD", source="C", target="D", distance=1.0))

        result = decompose_assur_groups(graph)

        # Override ground positions
        initial = {"A": (0.0, 0.0), "D": (3.0, 0.0), "B": (1.0, 0.0)}
        positions = solve_decomposition(result, initial_positions=initial)

        assert positions["A"] == (0.0, 0.0)
        assert positions["B"] == (1.0, 0.0)
