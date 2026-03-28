"""Tests for the high-level solve module (solver/solve.py).

Covers solve_group dispatch, RRR/RRP/PP dyad solving, triad solving,
and solve_decomposition.
"""

import pytest

from pylinkage.assur.decomposition import DecompositionResult
from pylinkage.assur.groups import Dyad, Triad
from pylinkage.dimensions import Dimensions
from pylinkage.exceptions import UnbuildableError
from pylinkage.solver.solve import solve_decomposition, solve_group

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rrr_dyad(anchor_nodes=("A0", "A1"), internal_nodes=("P",), edges=("e0", "e1")):
    """Create a minimal RRR dyad."""
    return Dyad(
        _signature="RRR",
        internal_nodes=tuple(internal_nodes),
        anchor_nodes=tuple(anchor_nodes),
        internal_edges=tuple(edges),
    )


def _make_rrp_dyad(
    anchor_nodes=("A0",),
    internal_nodes=("P",),
    edges=("e0",),
    line_node1="L1",
    line_node2="L2",
):
    """Create a minimal RRP dyad."""
    return Dyad(
        _signature="RRP",
        internal_nodes=tuple(internal_nodes),
        anchor_nodes=tuple(anchor_nodes),
        internal_edges=tuple(edges),
        line_node1=line_node1,
        line_node2=line_node2,
    )


def _make_pp_dyad(
    internal_nodes=("P",),
    anchor_nodes=(),
    line_node1="L1",
    line_node2="L2",
    line2_node1="L3",
    line2_node2="L4",
):
    """Create a minimal PP dyad."""
    return Dyad(
        _signature="PP",
        internal_nodes=tuple(internal_nodes),
        anchor_nodes=tuple(anchor_nodes),
        internal_edges=(),
        line_node1=line_node1,
        line_node2=line_node2,
        line2_node1=line2_node1,
        line2_node2=line2_node2,
    )


# ---------------------------------------------------------------------------
# solve_group dispatch
# ---------------------------------------------------------------------------


class TestSolveGroupDispatch:
    """Test that solve_group dispatches to the correct solver."""

    def test_unsupported_category_raises(self):
        """Group with unknown solver_category raises NotImplementedError."""

        class FakeGroup(Dyad):
            @property
            def solver_category(self):
                return "unknown_solver"

        group = FakeGroup(
            _signature="RRR",
            internal_nodes=("P",),
            anchor_nodes=("A0", "A1"),
            internal_edges=("e0", "e1"),
        )
        dims = Dimensions()
        with pytest.raises(NotImplementedError, match="Solver not implemented"):
            solve_group(group, {}, dims)


# ---------------------------------------------------------------------------
# RRR dyad solving
# ---------------------------------------------------------------------------


class TestSolveRRRDyad:
    """Tests for _solve_dyad_rrr via solve_group."""

    def test_basic_rrr(self):
        """Solve a simple RRR dyad: two anchors at (0,0) and (3,0), distances 2 and 2."""
        group = _make_rrr_dyad()
        dims = Dimensions(
            edge_distances={"e0": 2.0, "e1": 2.0},
            node_positions={"P": (1.5, 1.5)},  # hint toward upper intersection
        )
        positions = {"A0": (0.0, 0.0), "A1": (3.0, 0.0)}
        result = solve_group(group, positions, dims)
        assert "P" in result
        x, y = result["P"]
        assert y > 0  # upper intersection
        assert abs(x - 1.5) < 0.1

    def test_rrr_wrong_internal_count_raises(self):
        """Dyad with != 1 internal node raises ValueError."""
        group = Dyad(
            _signature="RRR",
            internal_nodes=("P1", "P2"),
            anchor_nodes=("A0", "A1"),
            internal_edges=("e0", "e1"),
        )
        dims = Dimensions(edge_distances={"e0": 1.0, "e1": 1.0})
        with pytest.raises(ValueError, match="exactly 1 internal"):
            solve_group(group, {"A0": (0, 0), "A1": (1, 0)}, dims)

    def test_rrr_wrong_anchor_count_raises(self):
        """Dyad with != 2 anchor nodes raises ValueError."""
        group = Dyad(
            _signature="RRR",
            internal_nodes=("P",),
            anchor_nodes=("A0",),
            internal_edges=("e0", "e1"),
        )
        dims = Dimensions(edge_distances={"e0": 1.0, "e1": 1.0})
        with pytest.raises(ValueError, match="exactly 2 anchor"):
            solve_group(group, {"A0": (0, 0)}, dims)

    def test_rrr_missing_anchor_position_raises(self):
        """Missing anchor position raises ValueError."""
        group = _make_rrr_dyad()
        dims = Dimensions(edge_distances={"e0": 1.0, "e1": 1.0})
        # Missing A1 position
        with pytest.raises(ValueError, match="position not found"):
            solve_group(group, {"A0": (0, 0)}, dims)

    def test_rrr_missing_distance_raises(self):
        """Missing edge distance in dimensions raises ValueError."""
        group = _make_rrr_dyad()
        dims = Dimensions(edge_distances={"e0": 1.0})  # missing e1
        with pytest.raises(ValueError, match="distances must be set"):
            solve_group(group, {"A0": (0, 0), "A1": (3, 0)}, dims)

    def test_rrr_only_one_edge(self):
        """Single internal edge means only one distance; should fail."""
        group = Dyad(
            _signature="RRR",
            internal_nodes=("P",),
            anchor_nodes=("A0", "A1"),
            internal_edges=("e0",),
        )
        dims = Dimensions(edge_distances={"e0": 1.0})
        with pytest.raises(ValueError, match="distances must be set"):
            solve_group(group, {"A0": (0, 0), "A1": (3, 0)}, dims)

    def test_rrr_unbuildable_raises(self):
        """Circles too far apart raises UnbuildableError."""
        group = _make_rrr_dyad()
        dims = Dimensions(
            edge_distances={"e0": 1.0, "e1": 1.0},
            node_positions={"P": (5.0, 0.0)},
        )
        with pytest.raises(UnbuildableError):
            solve_group(group, {"A0": (0, 0), "A1": (10, 0)}, dims)

    def test_rrr_hint_from_hint_positions(self):
        """Hint from hint_positions should be preferred."""
        group = _make_rrr_dyad()
        dims = Dimensions(
            edge_distances={"e0": 2.0, "e1": 2.0},
            node_positions={"P": (1.5, -1.5)},  # hint toward lower
        )
        positions = {"A0": (0.0, 0.0), "A1": (3.0, 0.0)}
        # hint_positions overrides to upper
        result = solve_group(group, positions, dims, hint_positions={"P": (1.5, 1.5)})
        assert result["P"][1] > 0


# ---------------------------------------------------------------------------
# RRP dyad solving
# ---------------------------------------------------------------------------


class TestSolveRRPDyad:
    """Tests for _solve_dyad_rrp via solve_group."""

    def test_basic_rrp(self):
        """Solve a circle-line intersection."""
        group = _make_rrp_dyad()
        dims = Dimensions(
            edge_distances={"e0": 2.0},
            node_positions={"P": (0.0, 1.0)},  # hint
        )
        # Anchor at origin, line y=1
        positions = {"A0": (0.0, 0.0), "L1": (-5.0, 1.0), "L2": (5.0, 1.0)}
        result = solve_group(group, positions, dims)
        assert "P" in result
        # Should be on line y=1 at distance 2 from origin
        assert abs(result["P"][1] - 1.0) < 1e-5

    def test_rrp_wrong_internal_count_raises(self):
        """Dyad with != 1 internal node raises ValueError."""
        group = Dyad(
            _signature="RRP",
            internal_nodes=("P1", "P2"),
            anchor_nodes=("A0",),
            internal_edges=("e0",),
            line_node1="L1",
            line_node2="L2",
        )
        dims = Dimensions(edge_distances={"e0": 1.0})
        with pytest.raises(ValueError, match="exactly 1 internal"):
            solve_group(group, {"A0": (0, 0), "L1": (0, 1), "L2": (1, 1)}, dims)

    def test_rrp_no_anchor_raises(self):
        """Dyad with no anchor nodes raises ValueError."""
        group = Dyad(
            _signature="RRP",
            internal_nodes=("P",),
            anchor_nodes=(),
            internal_edges=("e0",),
            line_node1="L1",
            line_node2="L2",
        )
        dims = Dimensions(edge_distances={"e0": 1.0})
        with pytest.raises(ValueError, match="at least 1 anchor"):
            solve_group(group, {"L1": (0, 1), "L2": (1, 1)}, dims)

    def test_rrp_line_nodes_none_raises(self):
        """Line nodes not set raises ValueError."""
        group = Dyad(
            _signature="RRP",
            internal_nodes=("P",),
            anchor_nodes=("A0",),
            internal_edges=("e0",),
            line_node1=None,
            line_node2=None,
        )
        dims = Dimensions(edge_distances={"e0": 1.0})
        with pytest.raises(ValueError, match="line nodes must be set"):
            solve_group(group, {"A0": (0, 0)}, dims)

    def test_rrp_missing_anchor_position_raises(self):
        """Missing anchor position raises ValueError."""
        group = _make_rrp_dyad()
        dims = Dimensions(edge_distances={"e0": 1.0})
        with pytest.raises(ValueError, match="position not found"):
            solve_group(group, {"L1": (0, 1), "L2": (1, 1)}, dims)

    def test_rrp_missing_line_node_position_raises(self):
        """Missing line node position raises ValueError."""
        group = _make_rrp_dyad()
        dims = Dimensions(edge_distances={"e0": 1.0})
        with pytest.raises(ValueError, match="position not found"):
            solve_group(group, {"A0": (0, 0), "L1": (0, 1)}, dims)

    def test_rrp_missing_distance_raises(self):
        """Missing edge distance raises ValueError."""
        group = _make_rrp_dyad(edges=())
        dims = Dimensions()
        with pytest.raises(ValueError, match="revolute_distance must be set"):
            solve_group(group, {"A0": (0, 0), "L1": (0, 1), "L2": (1, 1)}, dims)

    def test_rrp_unbuildable_raises(self):
        """Circle-line no intersection raises UnbuildableError."""
        group = _make_rrp_dyad()
        dims = Dimensions(
            edge_distances={"e0": 1.0},
            node_positions={"P": (0, 10)},
        )
        # Line at y=10, circle radius 1 at origin
        with pytest.raises(UnbuildableError):
            solve_group(
                group,
                {"A0": (0, 0), "L1": (-5, 10), "L2": (5, 10)},
                dims,
            )

    def test_rrp_hint_from_hint_positions(self):
        """Hint from hint_positions is used."""
        group = _make_rrp_dyad()
        dims = Dimensions(
            edge_distances={"e0": 2.0},
            node_positions={"P": (-5.0, 1.0)},  # hint toward left
        )
        positions = {"A0": (0.0, 0.0), "L1": (-5.0, 1.0), "L2": (5.0, 1.0)}
        result = solve_group(group, positions, dims, hint_positions={"P": (5.0, 1.0)})
        # Should pick the solution closer to the hint (right side)
        assert result["P"][0] > 0


# ---------------------------------------------------------------------------
# PP dyad solving
# ---------------------------------------------------------------------------


class TestSolvePPDyad:
    """Tests for _solve_dyad_pp via solve_group."""

    def test_basic_pp(self):
        """Solve a line-line intersection."""
        group = _make_pp_dyad()
        positions = {
            "L1": (0.0, 0.0),
            "L2": (1.0, 0.0),  # horizontal line y=0
            "L3": (0.0, 0.0),
            "L4": (0.0, 1.0),  # vertical line x=0
        }
        dims = Dimensions()
        result = solve_group(group, positions, dims)
        assert "P" in result
        # Intersection of y=0 and x=0 is (0,0)
        assert abs(result["P"][0]) < 1e-5
        assert abs(result["P"][1]) < 1e-5

    def test_pp_wrong_internal_count_raises(self):
        """Dyad with != 1 internal node raises ValueError."""
        group = Dyad(
            _signature="PP",
            internal_nodes=("P1", "P2"),
            anchor_nodes=(),
            internal_edges=(),
            line_node1="L1",
            line_node2="L2",
            line2_node1="L3",
            line2_node2="L4",
        )
        dims = Dimensions()
        with pytest.raises(ValueError, match="exactly 1 internal"):
            solve_group(
                group,
                {"L1": (0, 0), "L2": (1, 0), "L3": (0, 0), "L4": (0, 1)},
                dims,
            )

    def test_pp_missing_line_nodes_raises(self):
        """All four line nodes must be set."""
        group = Dyad(
            _signature="PP",
            internal_nodes=("P",),
            anchor_nodes=(),
            internal_edges=(),
            line_node1="L1",
            line_node2="L2",
            line2_node1=None,
            line2_node2=None,
        )
        dims = Dimensions()
        with pytest.raises(ValueError, match="all four line nodes"):
            solve_group(group, {"L1": (0, 0), "L2": (1, 0)}, dims)

    def test_pp_missing_position_raises(self):
        """Missing line node position raises ValueError."""
        group = _make_pp_dyad()
        dims = Dimensions()
        with pytest.raises(ValueError, match="position not found"):
            solve_group(
                group,
                {"L1": (0, 0), "L2": (1, 0), "L3": (0, 0)},
                dims,
            )

    def test_pp_parallel_lines_raises(self):
        """Parallel lines raise UnbuildableError."""
        group = _make_pp_dyad()
        dims = Dimensions()
        # Two horizontal parallel lines
        with pytest.raises(UnbuildableError):
            solve_group(
                group,
                {
                    "L1": (0.0, 0.0),
                    "L2": (1.0, 0.0),
                    "L3": (0.0, 1.0),
                    "L4": (1.0, 1.0),
                },
                dims,
            )

    def test_pp_non_trivial_intersection(self):
        """Two non-axis-aligned lines should intersect correctly."""
        group = _make_pp_dyad()
        # Line 1: y=x (from (0,0) to (1,1))
        # Line 2: y=-x+2 (from (0,2) to (2,0))
        # Intersection at (1, 1)
        positions = {
            "L1": (0.0, 0.0),
            "L2": (1.0, 1.0),
            "L3": (0.0, 2.0),
            "L4": (2.0, 0.0),
        }
        dims = Dimensions()
        result = solve_group(group, positions, dims)
        assert abs(result["P"][0] - 1.0) < 1e-5
        assert abs(result["P"][1] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Triad solving
# ---------------------------------------------------------------------------


class TestSolveTriad:
    """Tests for _solve_triad via solve_group."""

    def test_triad_wrong_internal_count_raises(self):
        """Triad with != 2 internal nodes raises ValueError."""
        group = Triad(
            _signature="RRRRRR",
            internal_nodes=("P1",),
            anchor_nodes=("A0", "A1", "A2"),
            internal_edges=("e0", "e1", "e2", "e3"),
        )
        dims = Dimensions()
        with pytest.raises(ValueError, match="exactly 2 internal"):
            solve_group(group, {"A0": (0, 0), "A1": (1, 0), "A2": (0.5, 1)}, dims)

    def test_triad_missing_anchor_raises(self):
        """Missing anchor position raises ValueError."""
        group = Triad(
            _signature="RRRRRR",
            internal_nodes=("P1", "P2"),
            anchor_nodes=("A0", "A1", "A2"),
            internal_edges=("e0", "e1", "e2", "e3"),
            edge_map={
                "e0": ("A0", "P1"),
                "e1": ("A1", "P1"),
                "e2": ("P1", "P2"),
                "e3": ("A2", "P2"),
            },
        )
        dims = Dimensions(
            edge_distances={"e0": 1.0, "e1": 1.0, "e2": 1.0, "e3": 1.0},
        )
        with pytest.raises(ValueError, match="position not found"):
            solve_group(group, {"A0": (0, 0), "A1": (1, 0)}, dims)

    def test_triad_no_edge_map_raises(self):
        """Triad with no edge_map raises ValueError."""
        group = Triad(
            _signature="RRRRRR",
            internal_nodes=("P1", "P2"),
            anchor_nodes=("A0", "A1", "A2"),
            internal_edges=("e0", "e1", "e2", "e3"),
        )
        dims = Dimensions()
        with pytest.raises(ValueError, match="no edge_map"):
            solve_group(
                group,
                {"A0": (0, 0), "A1": (1, 0), "A2": (0.5, 1)},
                dims,
            )

    def test_triad_missing_edge_distance_raises(self):
        """Missing edge distance in dimensions raises ValueError."""
        group = Triad(
            _signature="RRRRRR",
            internal_nodes=("P1", "P2"),
            anchor_nodes=("A0", "A1", "A2"),
            internal_edges=("e0", "e1", "e2", "e3"),
            edge_map={
                "e0": ("A0", "P1"),
                "e1": ("A1", "P1"),
                "e2": ("P1", "P2"),
                "e3": ("A2", "P2"),
            },
        )
        dims = Dimensions(
            edge_distances={"e0": 1.0, "e1": 1.0, "e2": 1.0},  # missing e3
        )
        with pytest.raises(ValueError, match="no distance"):
            solve_group(
                group,
                {"A0": (0, 0), "A1": (1, 0), "A2": (0.5, 1)},
                dims,
            )

    def test_triad_too_few_constraints_raises(self):
        """Triad with < 4 constraints raises ValueError."""
        group = Triad(
            _signature="RRRRRR",
            internal_nodes=("P1", "P2"),
            anchor_nodes=("A0", "A1", "A2"),
            internal_edges=("e0", "e1", "e2"),
            edge_map={
                "e0": ("A0", "P1"),
                "e1": ("A1", "P1"),
                "e2": ("P1", "P2"),
            },
        )
        dims = Dimensions(
            edge_distances={"e0": 1.0, "e1": 1.0, "e2": 1.0},
        )
        with pytest.raises(ValueError, match="at least 4"):
            solve_group(
                group,
                {"A0": (0, 0), "A1": (1, 0), "A2": (0.5, 1)},
                dims,
            )


# ---------------------------------------------------------------------------
# solve_decomposition
# ---------------------------------------------------------------------------


class TestSolveDecomposition:
    """Tests for solve_decomposition."""

    def test_no_graph_raises(self):
        """DecompositionResult with no graph raises ValueError."""
        result = DecompositionResult(graph=None)
        dims = Dimensions()
        with pytest.raises(ValueError, match="no graph"):
            solve_decomposition(result, dims)

    def test_ground_only(self):
        """Decomposition with only ground nodes returns their positions."""
        from pylinkage.assur._types import NodeRole
        from pylinkage.assur.graph import LinkageGraph, Node

        graph = LinkageGraph()
        graph.add_node(Node("G1", role=NodeRole.GROUND))
        graph.add_node(Node("G2", role=NodeRole.GROUND))

        result = DecompositionResult(
            ground=["G1", "G2"],
            drivers=[],
            groups=[],
            graph=graph,
        )
        dims = Dimensions(
            node_positions={"G1": (0.0, 0.0), "G2": (3.0, 0.0)},
        )
        positions = solve_decomposition(result, dims)
        assert positions["G1"] == (0.0, 0.0)
        assert positions["G2"] == (3.0, 0.0)

    def test_ground_missing_position_raises(self):
        """Ground node without position raises ValueError."""
        from pylinkage.assur._types import NodeRole
        from pylinkage.assur.graph import LinkageGraph, Node

        graph = LinkageGraph()
        graph.add_node(Node("G1", role=NodeRole.GROUND))

        result = DecompositionResult(
            ground=["G1"],
            drivers=[],
            groups=[],
            graph=graph,
        )
        dims = Dimensions()  # no positions
        with pytest.raises(ValueError, match="Ground node"):
            solve_decomposition(result, dims)

    def test_driver_missing_position_raises(self):
        """Driver node without position raises ValueError."""
        from pylinkage.assur._types import NodeRole
        from pylinkage.assur.graph import LinkageGraph, Node

        graph = LinkageGraph()
        graph.add_node(Node("G1", role=NodeRole.GROUND))
        graph.add_node(Node("D1", role=NodeRole.DRIVER))

        result = DecompositionResult(
            ground=["G1"],
            drivers=["D1"],
            groups=[],
            graph=graph,
        )
        dims = Dimensions(node_positions={"G1": (0.0, 0.0)})
        with pytest.raises(ValueError, match="Driver node"):
            solve_decomposition(result, dims)

    def test_initial_positions_override(self):
        """initial_positions should override dimensions positions."""
        from pylinkage.assur._types import NodeRole
        from pylinkage.assur.graph import LinkageGraph, Node

        graph = LinkageGraph()
        graph.add_node(Node("G1", role=NodeRole.GROUND))

        result = DecompositionResult(
            ground=["G1"],
            drivers=[],
            groups=[],
            graph=graph,
        )
        dims = Dimensions(node_positions={"G1": (0.0, 0.0)})
        positions = solve_decomposition(
            result, dims, initial_positions={"G1": (5.0, 5.0)}
        )
        assert positions["G1"] == (5.0, 5.0)

    def test_driver_from_initial_positions(self):
        """Driver position taken from initial_positions when provided."""
        from pylinkage.assur._types import NodeRole
        from pylinkage.assur.graph import LinkageGraph, Node

        graph = LinkageGraph()
        graph.add_node(Node("G1", role=NodeRole.GROUND))
        graph.add_node(Node("D1", role=NodeRole.DRIVER))

        result = DecompositionResult(
            ground=["G1"],
            drivers=["D1"],
            groups=[],
            graph=graph,
        )
        dims = Dimensions(node_positions={"G1": (0.0, 0.0)})
        positions = solve_decomposition(
            result, dims, initial_positions={"G1": (0.0, 0.0), "D1": (1.0, 0.0)}
        )
        assert positions["D1"] == (1.0, 0.0)

    def test_with_rrr_group(self):
        """Full decomposition with a ground + driver + one RRR dyad."""
        from pylinkage.assur._types import NodeRole
        from pylinkage.assur.graph import LinkageGraph, Node

        graph = LinkageGraph()
        graph.add_node(Node("G1", role=NodeRole.GROUND))
        graph.add_node(Node("D1", role=NodeRole.DRIVER))
        graph.add_node(Node("P", role=NodeRole.DRIVEN))

        group = _make_rrr_dyad(anchor_nodes=("D1", "G1"), internal_nodes=("P",))
        result = DecompositionResult(
            ground=["G1"],
            drivers=["D1"],
            groups=[group],
            graph=graph,
        )
        dims = Dimensions(
            node_positions={
                "G1": (3.0, 0.0),
                "D1": (1.0, 0.0),
                "P": (2.0, 1.0),
            },
            edge_distances={"e0": 2.0, "e1": 2.0},
        )
        positions = solve_decomposition(result, dims)
        assert "P" in positions
        assert "G1" in positions
        assert "D1" in positions
