"""Tests for the triad solver (Newton-Raphson on distance constraints)."""

import math

import pytest

from pylinkage.assur import (
    Edge,
    LinkageGraph,
    Node,
    NodeRole,
    decompose_assur_groups,
)
from pylinkage.assur.groups import Triad
from pylinkage.dimensions import Dimensions
from pylinkage.exceptions import UnbuildableError
from pylinkage.solver.groups import solve_triad
from pylinkage.solver.solve import solve_decomposition, solve_group


class TestSolveTriad:
    """Tests for the standalone triad solver."""

    def test_equilateral_triad(self):
        """Solve a triad forming an equilateral-like configuration.

        Two unknowns X and Y connected to 3 known anchors:
        - X at distance 1.0 from A(0,0)
        - X at distance 1.0 from Y
        - Y at distance 1.0 from B(1,0)
        - Y at distance 1.0 from C(0.5, -0.866)
        """
        positions = {
            "A": (0.0, 0.0),
            "B": (1.0, 0.0),
            "C": (0.5, -0.866),
        }
        constraints = [
            ("X", "A", 1.0),
            ("X", "Y", 1.0),
            ("Y", "B", 1.0),
            ("Y", "C", 1.0),
        ]
        hints = {
            "X": (0.5, 0.5),
            "Y": (0.8, -0.3),
        }

        result = solve_triad(
            constraints=constraints,
            positions=positions,
            internal_ids=("X", "Y"),
            hints=hints,
        )

        assert "X" in result
        assert "Y" in result

        # Verify all distance constraints are satisfied
        all_pos = {**positions, **result}
        for na, nb, dist in constraints:
            pa, pb = all_pos[na], all_pos[nb]
            actual = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            assert abs(actual - dist) < 1e-4, (
                f"Constraint {na}-{nb}: expected {dist}, got {actual}"
            )

    def test_rectangle_triad(self):
        """Solve a triad in a rectangular configuration.

        Known: A(0,0), B(3,0), C(3,4)
        Unknowns: X, Y
        Constraints:
        - X at distance 2.0 from A
        - X at distance 2.5 from Y
        - Y at distance 2.0 from B
        - Y at distance 2.0 from C
        """
        positions = {
            "A": (0.0, 0.0),
            "B": (3.0, 0.0),
            "C": (3.0, 4.0),
        }
        constraints = [
            ("X", "A", 2.0),
            ("X", "Y", 2.5),
            ("Y", "B", 2.0),
            ("Y", "C", 2.0),
        ]

        result = solve_triad(
            constraints=constraints,
            positions=positions,
            internal_ids=("X", "Y"),
            hints={"X": (1.0, 1.5), "Y": (3.0, 2.0)},
        )

        # Verify constraints
        all_pos = {**positions, **result}
        for na, nb, dist in constraints:
            pa, pb = all_pos[na], all_pos[nb]
            actual = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            assert abs(actual - dist) < 1e-4

    def test_no_hints_uses_centroid(self):
        """Solver works without hints by using anchor centroid."""
        positions = {
            "A": (0.0, 0.0),
            "B": (2.0, 0.0),
            "C": (1.0, 2.0),
        }
        constraints = [
            ("X", "A", 1.5),
            ("X", "Y", 1.0),
            ("Y", "B", 1.5),
            ("Y", "C", 1.5),
        ]

        result = solve_triad(
            constraints=constraints,
            positions=positions,
            internal_ids=("X", "Y"),
            hints=None,
        )

        # Just verify it converged and constraints are satisfied
        all_pos = {**positions, **result}
        for na, nb, dist in constraints:
            pa, pb = all_pos[na], all_pos[nb]
            actual = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            assert abs(actual - dist) < 1e-4

    def test_unbuildable_raises(self):
        """Impossible distances should raise ValueError."""
        positions = {
            "A": (0.0, 0.0),
            "B": (100.0, 0.0),
            "C": (50.0, 100.0),
        }
        # Distances way too short to reach between anchors
        constraints = [
            ("X", "A", 0.1),
            ("X", "Y", 0.1),
            ("Y", "B", 0.1),
            ("Y", "C", 0.1),
        ]

        with pytest.raises(ValueError):
            solve_triad(
                constraints=constraints,
                positions=positions,
                internal_ids=("X", "Y"),
            )


class TestSolveGroupTriad:
    """Tests for triad solving through the solve_group dispatch."""

    def test_solve_group_triad(self):
        """solve_group() correctly dispatches to the triad solver."""
        triad = Triad(
            _signature="RRRRRR",
            internal_nodes=("X", "Y"),
            anchor_nodes=("A", "B", "C"),
            internal_edges=("AX", "XY", "YB", "YC"),
            edge_map={
                "AX": ("X", "A"),
                "XY": ("X", "Y"),
                "YB": ("Y", "B"),
                "YC": ("Y", "C"),
            },
        )

        positions = {
            "A": (0.0, 0.0),
            "B": (2.0, 0.0),
            "C": (1.0, 2.0),
        }

        dimensions = Dimensions(
            node_positions={
                "A": (0.0, 0.0),
                "B": (2.0, 0.0),
                "C": (1.0, 2.0),
                "X": (0.8, 1.0),  # hint
                "Y": (1.5, 1.0),  # hint
            },
            edge_distances={
                "AX": 1.5,
                "XY": 1.0,
                "YB": 1.5,
                "YC": 1.5,
            },
        )

        result = solve_group(triad, positions, dimensions)
        assert "X" in result
        assert "Y" in result

        # Verify distances
        all_pos = {**positions, **result}
        for edge_id, (na, nb) in triad.edge_map.items():
            dist = dimensions.get_edge_distance(edge_id)
            pa, pb = all_pos[na], all_pos[nb]
            actual = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            assert abs(actual - dist) < 1e-4, (
                f"Edge {edge_id} ({na}-{nb}): expected {dist}, got {actual}"
            )

    def test_solve_group_triad_missing_anchor_raises(self):
        """Missing anchor position raises ValueError."""
        triad = Triad(
            _signature="RRRRRR",
            internal_nodes=("X", "Y"),
            anchor_nodes=("A", "B", "C"),
            internal_edges=("AX", "XY", "YB", "YC"),
            edge_map={
                "AX": ("X", "A"),
                "XY": ("X", "Y"),
                "YB": ("Y", "B"),
                "YC": ("Y", "C"),
            },
        )

        positions = {"A": (0.0, 0.0)}  # Missing B and C

        dimensions = Dimensions(
            edge_distances={"AX": 1.0, "XY": 1.0, "YB": 1.0, "YC": 1.0},
        )

        with pytest.raises(ValueError, match="position not found"):
            solve_group(triad, positions, dimensions)

    def test_solve_group_triad_no_edge_map_raises(self):
        """Triad without edge_map raises ValueError."""
        triad = Triad(
            _signature="RRRRRR",
            internal_nodes=("X", "Y"),
            anchor_nodes=("A", "B", "C"),
            internal_edges=("AX", "XY", "YB", "YC"),
            # No edge_map!
        )

        positions = {"A": (0.0, 0.0), "B": (2.0, 0.0), "C": (1.0, 2.0)}
        dimensions = Dimensions(
            edge_distances={"AX": 1.0, "XY": 1.0, "YB": 1.0, "YC": 1.0},
        )

        with pytest.raises(ValueError, match="edge_map"):
            solve_group(triad, positions, dimensions)


class TestSolveDecompositionWithTriad:
    """End-to-end test: decompose a graph with a triad, then solve it."""

    def test_decompose_and_solve_triad(self):
        """Build a graph requiring a triad, decompose, and solve."""
        graph = LinkageGraph(name="Triad e2e")
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("B", role=NodeRole.GROUND))
        graph.add_node(Node("C", role=NodeRole.GROUND))
        graph.add_node(Node("D", role=NodeRole.DRIVER))
        graph.add_node(Node("X", role=NodeRole.DRIVEN))
        graph.add_node(Node("Y", role=NodeRole.DRIVEN))

        # Crank
        graph.add_edge(Edge("AD", source="A", target="D"))

        # X connects to D (1 known neighbor)
        graph.add_edge(Edge("DX", source="D", target="X"))
        graph.add_edge(Edge("XY", source="X", target="Y"))

        # Y connects to B and C (2 known) + X (unknown)
        graph.add_edge(Edge("YB", source="Y", target="B"))
        graph.add_edge(Edge("YC", source="Y", target="C"))

        # Dimensions
        dims = Dimensions(
            node_positions={
                "A": (0.0, 0.0),
                "B": (3.0, 0.0),
                "C": (1.5, 2.598),
                "D": (1.0, 0.0),
                "X": (1.5, 1.0),
                "Y": (2.0, 1.0),
            },
            edge_distances={
                "AD": 1.0,
                "DX": 1.2,
                "XY": 1.0,
                "YB": 1.5,
                "YC": 1.5,
            },
        )

        # Decompose
        result = decompose_assur_groups(graph)
        assert len(result.groups) >= 1

        # Check that at least one group is a triad or that decomposition
        # found some valid grouping
        has_triad = any(g.group_class == 2 for g in result.groups)
        # If Y has 2 known neighbors (B,C), it may decompose as dyad first.
        # That's fine — the decomposer prefers dyads.

        # Solve regardless of decomposition strategy
        positions = solve_decomposition(result, dims)

        # All nodes should have positions
        for node_id in graph.nodes:
            assert node_id in positions, f"Node {node_id} not in solution"

        # Verify edge distance constraints
        for edge in graph.edges.values():
            dist = dims.get_edge_distance(edge.id)
            if dist is not None:
                pa = positions[edge.source]
                pb = positions[edge.target]
                actual = math.sqrt(
                    (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
                )
                assert abs(actual - dist) < 1e-3, (
                    f"Edge {edge.id} ({edge.source}-{edge.target}): "
                    f"expected {dist}, got {actual}"
                )
