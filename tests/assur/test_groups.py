"""Tests for the groups module."""

import math

import pytest

from pylinkage.assur import (
    DyadRRP,
    DyadRRR,
    Edge,
    JointType,
    LinkageGraph,
    Node,
    NodeRole,
)
from pylinkage.exceptions import UnbuildableError


class TestDyadRRR:
    """Tests for the DyadRRR class."""

    def test_dyad_rrr_properties(self):
        """Test DyadRRR basic properties."""
        dyad = DyadRRR(
            internal_nodes=("C",),
            anchor_nodes=("A", "B"),
            distance0=1.0,
            distance1=1.0,
        )
        assert dyad.group_class == 1
        assert dyad.joint_signature == "RRR"

    def test_dyad_rrr_solve_two_intersections(self):
        """Test solving RRR dyad with two circle intersections."""
        # Set up a simple case: equilateral triangle
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("B", role=NodeRole.GROUND, position=(1.0, 0.0)))
        graph.add_node(Node("C", role=NodeRole.DRIVEN, position=(0.5, 0.866)))
        graph.add_edge(Edge("AC", source="A", target="C", distance=1.0))
        graph.add_edge(Edge("BC", source="B", target="C", distance=1.0))

        dyad = DyadRRR(
            internal_nodes=("C",),
            anchor_nodes=("A", "B"),
            internal_edges=("AC", "BC"),
            distance0=1.0,
            distance1=1.0,
        )

        previous_positions = {
            "A": (0.0, 0.0),
            "B": (1.0, 0.0),
        }

        result = dyad.solve(graph, previous_positions)
        assert "C" in result
        x, y = result["C"]

        # Should be near (0.5, 0.866) since that's the current position
        # and it will choose the nearest solution
        assert abs(x - 0.5) < 0.01
        assert abs(y - 0.866) < 0.01

    def test_dyad_rrr_solve_tangent(self):
        """Test solving RRR dyad with tangent circles (one solution)."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("B", role=NodeRole.GROUND, position=(2.0, 0.0)))
        graph.add_node(Node("C", role=NodeRole.DRIVEN, position=(1.0, 0.0)))
        graph.add_edge(Edge("AC", source="A", target="C", distance=1.0))
        graph.add_edge(Edge("BC", source="B", target="C", distance=1.0))

        dyad = DyadRRR(
            internal_nodes=("C",),
            anchor_nodes=("A", "B"),
            distance0=1.0,
            distance1=1.0,
        )

        previous_positions = {
            "A": (0.0, 0.0),
            "B": (2.0, 0.0),
        }

        result = dyad.solve(graph, previous_positions)
        x, y = result["C"]

        # Tangent point is at (1, 0)
        assert abs(x - 1.0) < 0.001
        assert abs(y - 0.0) < 0.001

    def test_dyad_rrr_solve_no_intersection_raises(self):
        """Test that no intersection raises UnbuildableError."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("B", role=NodeRole.GROUND, position=(10.0, 0.0)))
        graph.add_node(Node("C", role=NodeRole.DRIVEN, position=(5.0, 0.0)))
        graph.add_edge(Edge("AC", source="A", target="C", distance=1.0))
        graph.add_edge(Edge("BC", source="B", target="C", distance=1.0))

        dyad = DyadRRR(
            internal_nodes=("C",),
            anchor_nodes=("A", "B"),
            distance0=1.0,  # Too short to reach
            distance1=1.0,
        )

        previous_positions = {
            "A": (0.0, 0.0),
            "B": (10.0, 0.0),
        }

        with pytest.raises(UnbuildableError):
            dyad.solve(graph, previous_positions)

    def test_dyad_rrr_solve_missing_anchor_raises(self):
        """Test that missing anchor position raises ValueError."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("C", role=NodeRole.DRIVEN, position=(0.5, 0.866)))

        dyad = DyadRRR(
            internal_nodes=("C",),
            anchor_nodes=("A", "B"),
            distance0=1.0,
            distance1=1.0,
        )

        previous_positions = {"A": (0.0, 0.0)}  # Missing B

        with pytest.raises(ValueError, match="position not found"):
            dyad.solve(graph, previous_positions)

    def test_dyad_rrr_can_form(self):
        """Test can_form method."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("B", role=NodeRole.GROUND, position=(1.0, 0.0)))
        graph.add_node(Node("C", role=NodeRole.DRIVEN, joint_type=JointType.REVOLUTE))
        graph.add_edge(Edge("AC", source="A", target="C", distance=1.0))
        graph.add_edge(Edge("BC", source="B", target="C", distance=1.0))

        # Valid RRR formation
        assert DyadRRR.can_form(["C"], ["A", "B"], graph)

        # Not enough internal nodes
        assert not DyadRRR.can_form([], ["A", "B"], graph)

        # Not enough anchors
        assert not DyadRRR.can_form(["C"], ["A"], graph)


class TestDyadRRP:
    """Tests for the DyadRRP class."""

    def test_dyad_rrp_properties(self):
        """Test DyadRRP basic properties."""
        dyad = DyadRRP(
            internal_nodes=("C",),
            anchor_nodes=("A",),
            revolute_distance=1.0,
            line_node1="L1",
            line_node2="L2",
        )
        assert dyad.group_class == 1
        assert dyad.joint_signature == "RRP"

    def test_dyad_rrp_solve_two_intersections(self):
        """Test solving RRP dyad with two circle-line intersections."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("L1", role=NodeRole.GROUND, position=(0.0, 1.0)))
        graph.add_node(Node("L2", role=NodeRole.GROUND, position=(2.0, 1.0)))
        graph.add_node(Node("C", role=NodeRole.DRIVEN, position=(1.0, 1.0)))
        graph.add_edge(Edge("AC", source="A", target="C", distance=math.sqrt(2)))

        dyad = DyadRRP(
            internal_nodes=("C",),
            anchor_nodes=("A",),
            revolute_distance=math.sqrt(2),
            line_node1="L1",
            line_node2="L2",
        )

        previous_positions = {
            "A": (0.0, 0.0),
            "L1": (0.0, 1.0),
            "L2": (2.0, 1.0),
        }

        result = dyad.solve(graph, previous_positions)
        x, y = result["C"]

        # Should be at (1, 1) - the nearest intersection to initial position
        assert abs(y - 1.0) < 0.01  # On the line y=1

    def test_dyad_rrp_solve_no_intersection_raises(self):
        """Test that no intersection raises UnbuildableError."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("L1", role=NodeRole.GROUND, position=(0.0, 10.0)))
        graph.add_node(Node("L2", role=NodeRole.GROUND, position=(2.0, 10.0)))
        graph.add_node(Node("C", role=NodeRole.DRIVEN, position=(1.0, 10.0)))

        dyad = DyadRRP(
            internal_nodes=("C",),
            anchor_nodes=("A",),
            revolute_distance=1.0,  # Too short to reach line
            line_node1="L1",
            line_node2="L2",
        )

        previous_positions = {
            "A": (0.0, 0.0),
            "L1": (0.0, 10.0),
            "L2": (2.0, 10.0),
        }

        with pytest.raises(UnbuildableError):
            dyad.solve(graph, previous_positions)
