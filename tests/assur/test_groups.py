"""Tests for the groups module (topology only)."""

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
from pylinkage.dimensions import Dimensions
from pylinkage.exceptions import UnbuildableError
from pylinkage.solver.solve import solve_group


class TestDyadRRR:
    """Tests for the DyadRRR class."""

    def test_dyad_rrr_properties(self):
        """Test DyadRRR basic properties."""
        dyad = DyadRRR(
            internal_nodes=("C",),
            anchor_nodes=("A", "B"),
            internal_edges=("AC", "BC"),
        )
        assert dyad.group_class == 1
        assert dyad.joint_signature == "RRR"

    def test_dyad_rrr_solve_two_intersections(self):
        """Test solving RRR dyad with two circle intersections."""
        # Set up a simple case: equilateral triangle
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("B", role=NodeRole.GROUND))
        graph.add_node(Node("C", role=NodeRole.DRIVEN))
        graph.add_edge(Edge("AC", source="A", target="C"))
        graph.add_edge(Edge("BC", source="B", target="C"))

        dyad = DyadRRR(
            internal_nodes=("C",),
            anchor_nodes=("A", "B"),
            internal_edges=("AC", "BC"),
        )

        # Dimensions separate from topology
        dimensions = Dimensions(
            node_positions={
                "A": (0.0, 0.0),
                "B": (1.0, 0.0),
                "C": (0.5, 0.866),  # Used as hint
            },
            edge_distances={"AC": 1.0, "BC": 1.0},
        )

        previous_positions = {
            "A": (0.0, 0.0),
            "B": (1.0, 0.0),
        }

        result = solve_group(dyad, previous_positions, dimensions)
        assert "C" in result
        x, y = result["C"]

        # Should be near (0.5, 0.866) since that's the hint position
        # and it will choose the nearest solution
        assert abs(x - 0.5) < 0.01
        assert abs(y - 0.866) < 0.01

    def test_dyad_rrr_solve_tangent(self):
        """Test solving RRR dyad with tangent circles (one solution)."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("B", role=NodeRole.GROUND))
        graph.add_node(Node("C", role=NodeRole.DRIVEN))
        graph.add_edge(Edge("AC", source="A", target="C"))
        graph.add_edge(Edge("BC", source="B", target="C"))

        dyad = DyadRRR(
            internal_nodes=("C",),
            anchor_nodes=("A", "B"),
            internal_edges=("AC", "BC"),
        )

        dimensions = Dimensions(
            node_positions={
                "A": (0.0, 0.0),
                "B": (2.0, 0.0),
                "C": (1.0, 0.0),  # Hint
            },
            edge_distances={"AC": 1.0, "BC": 1.0},
        )

        previous_positions = {
            "A": (0.0, 0.0),
            "B": (2.0, 0.0),
        }

        result = solve_group(dyad, previous_positions, dimensions)
        x, y = result["C"]

        # Tangent point is at (1, 0)
        assert abs(x - 1.0) < 0.001
        assert abs(y - 0.0) < 0.001

    def test_dyad_rrr_solve_no_intersection_raises(self):
        """Test that no intersection raises UnbuildableError."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("B", role=NodeRole.GROUND))
        graph.add_node(Node("C", role=NodeRole.DRIVEN))
        graph.add_edge(Edge("AC", source="A", target="C"))
        graph.add_edge(Edge("BC", source="B", target="C"))

        dyad = DyadRRR(
            internal_nodes=("C",),
            anchor_nodes=("A", "B"),
            internal_edges=("AC", "BC"),
        )

        dimensions = Dimensions(
            node_positions={
                "A": (0.0, 0.0),
                "B": (10.0, 0.0),
                "C": (5.0, 0.0),
            },
            edge_distances={"AC": 1.0, "BC": 1.0},  # Too short to reach
        )

        previous_positions = {
            "A": (0.0, 0.0),
            "B": (10.0, 0.0),
        }

        with pytest.raises(UnbuildableError):
            solve_group(dyad, previous_positions, dimensions)

    def test_dyad_rrr_solve_missing_anchor_raises(self):
        """Test that missing anchor position raises ValueError."""
        dyad = DyadRRR(
            internal_nodes=("C",),
            anchor_nodes=("A", "B"),
            internal_edges=("AC", "BC"),
        )

        dimensions = Dimensions(
            node_positions={"A": (0.0, 0.0), "C": (0.5, 0.866)},
            edge_distances={"AC": 1.0, "BC": 1.0},
        )

        previous_positions = {"A": (0.0, 0.0)}  # Missing B

        with pytest.raises(ValueError, match="position not found"):
            solve_group(dyad, previous_positions, dimensions)

    def test_dyad_rrr_can_form(self):
        """Test can_form method."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("B", role=NodeRole.GROUND))
        graph.add_node(Node("C", role=NodeRole.DRIVEN, joint_type=JointType.REVOLUTE))
        graph.add_edge(Edge("AC", source="A", target="C"))
        graph.add_edge(Edge("BC", source="B", target="C"))

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
            internal_edges=("AC",),
            line_node1="L1",
            line_node2="L2",
        )
        assert dyad.group_class == 1
        assert dyad.joint_signature == "RRP"

    def test_dyad_rrp_solve_two_intersections(self):
        """Test solving RRP dyad with two circle-line intersections."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("L1", role=NodeRole.GROUND))
        graph.add_node(Node("L2", role=NodeRole.GROUND))
        graph.add_node(Node("C", role=NodeRole.DRIVEN))
        graph.add_edge(Edge("AC", source="A", target="C"))

        dyad = DyadRRP(
            internal_nodes=("C",),
            anchor_nodes=("A",),
            internal_edges=("AC",),
            line_node1="L1",
            line_node2="L2",
        )

        dimensions = Dimensions(
            node_positions={
                "A": (0.0, 0.0),
                "L1": (0.0, 1.0),
                "L2": (2.0, 1.0),
                "C": (1.0, 1.0),  # Hint
            },
            edge_distances={"AC": math.sqrt(2)},
        )

        previous_positions = {
            "A": (0.0, 0.0),
            "L1": (0.0, 1.0),
            "L2": (2.0, 1.0),
        }

        result = solve_group(dyad, previous_positions, dimensions)
        x, y = result["C"]

        # Should be at (1, 1) - the nearest intersection to initial position
        assert abs(y - 1.0) < 0.01  # On the line y=1

    def test_dyad_rrp_solve_no_intersection_raises(self):
        """Test that no intersection raises UnbuildableError."""
        graph = LinkageGraph()
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("L1", role=NodeRole.GROUND))
        graph.add_node(Node("L2", role=NodeRole.GROUND))
        graph.add_node(Node("C", role=NodeRole.DRIVEN))

        dyad = DyadRRP(
            internal_nodes=("C",),
            anchor_nodes=("A",),
            internal_edges=("AC",),
            line_node1="L1",
            line_node2="L2",
        )

        dimensions = Dimensions(
            node_positions={
                "A": (0.0, 0.0),
                "L1": (0.0, 10.0),
                "L2": (2.0, 10.0),
                "C": (1.0, 10.0),
            },
            edge_distances={"AC": 1.0},  # Too short to reach line
        )

        previous_positions = {
            "A": (0.0, 0.0),
            "L1": (0.0, 10.0),
            "L2": (2.0, 10.0),
        }

        with pytest.raises(UnbuildableError):
            solve_group(dyad, previous_positions, dimensions)
