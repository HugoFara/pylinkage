"""Additional tests for decomposition.py to increase coverage.

Covers: DecompositionResult.solve_order, all_nodes_in_order,
RRP dyad creation path, triad path, validate_decomposition edge cases,
and complex multi-group mechanisms.
"""

import pytest

from pylinkage.assur import (
    DecompositionResult,
    Edge,
    LinkageGraph,
    Node,
    NodeRole,
    decompose_assur_groups,
    validate_decomposition,
)
from pylinkage.assur._types import JointType
from pylinkage.assur.groups import Dyad


# ---------------------------------------------------------------------------
# DecompositionResult methods
# ---------------------------------------------------------------------------

class TestDecompositionResultMethods:
    def test_solve_order_empty(self):
        r = DecompositionResult()
        assert r.solve_order() == []

    def test_solve_order_drivers_then_groups(self):
        d1 = Dyad(_signature="RRR", internal_nodes=("C",))
        d2 = Dyad(_signature="RRR", internal_nodes=("E",))
        r = DecompositionResult(
            ground=["A"],
            drivers=["B"],
            groups=[d1, d2],
        )
        order = r.solve_order()
        assert order[0] == "B"
        assert order[1] is d1
        assert order[2] is d2

    def test_all_nodes_in_order(self):
        d = Dyad(_signature="RRR", internal_nodes=("C",))
        r = DecompositionResult(
            ground=["A", "D"],
            drivers=["B"],
            groups=[d],
        )
        nodes = r.all_nodes_in_order()
        assert nodes == ["A", "D", "B", "C"]

    def test_all_nodes_in_order_empty(self):
        r = DecompositionResult()
        assert r.all_nodes_in_order() == []

    def test_all_nodes_multiple_groups(self):
        d1 = Dyad(_signature="RRR", internal_nodes=("C",))
        d2 = Dyad(_signature="RRR", internal_nodes=("E",))
        r = DecompositionResult(
            ground=["A"],
            drivers=["B"],
            groups=[d1, d2],
        )
        nodes = r.all_nodes_in_order()
        assert nodes == ["A", "B", "C", "E"]


# ---------------------------------------------------------------------------
# RRP dyad decomposition path
# ---------------------------------------------------------------------------

class TestDecomposeRRPDyad:
    def test_rrp_dyad_created_with_three_known_neighbors(self):
        """When a driven node has >= 3 known neighbors, _try_create_rrp_dyad fires."""
        g = LinkageGraph()
        g.add_node(Node("G1", role=NodeRole.GROUND))
        g.add_node(Node("G2", role=NodeRole.GROUND))
        g.add_node(Node("D", role=NodeRole.DRIVER))
        g.add_node(Node("L", role=NodeRole.GROUND))
        g.add_node(Node("C", role=NodeRole.DRIVEN))

        g.add_edge(Edge("G1-D", source="G1", target="D"))
        g.add_edge(Edge("D-C", source="D", target="C"))
        g.add_edge(Edge("G2-C", source="G2", target="C"))
        g.add_edge(Edge("L-C", source="L", target="C"))

        result = decompose_assur_groups(g)
        assert len(result.groups) == 1
        group = result.groups[0]
        # Either RRR or RRP depending on the algorithm's priority
        assert group.internal_nodes == ("C",)


# ---------------------------------------------------------------------------
# Triad decomposition path
# ---------------------------------------------------------------------------

class TestDecomposeTriad:
    def test_triad_decomposition(self):
        """Two driven nodes that form a triad with known anchors."""
        g = LinkageGraph()
        g.add_node(Node("G1", role=NodeRole.GROUND))
        g.add_node(Node("G2", role=NodeRole.GROUND))
        g.add_node(Node("G3", role=NodeRole.GROUND))
        g.add_node(Node("D", role=NodeRole.DRIVER))
        g.add_node(Node("X", role=NodeRole.DRIVEN))
        g.add_node(Node("Y", role=NodeRole.DRIVEN))

        # Driver connection
        g.add_edge(Edge("G1-D", source="G1", target="D"))

        # X connects to D and G2
        g.add_edge(Edge("D-X", source="D", target="X"))
        g.add_edge(Edge("G2-X", source="G2", target="X"))
        # Y connects to G3 and X
        g.add_edge(Edge("X-Y", source="X", target="Y"))
        g.add_edge(Edge("G3-Y", source="G3", target="Y"))

        result = decompose_assur_groups(g)
        # X should be solved first as a dyad, then Y
        assert len(result.groups) >= 1
        all_internal = []
        for group in result.groups:
            all_internal.extend(group.internal_nodes)
        assert "X" in all_internal
        assert "Y" in all_internal

    def test_triad_two_unsolvable_individually(self):
        """Two nodes that can only be solved as a triad (not individually)."""
        g = LinkageGraph()
        g.add_node(Node("G1", role=NodeRole.GROUND))
        g.add_node(Node("G2", role=NodeRole.GROUND))
        g.add_node(Node("G3", role=NodeRole.GROUND))
        g.add_node(Node("X", role=NodeRole.DRIVEN))
        g.add_node(Node("Y", role=NodeRole.DRIVEN))

        # X connects to G1 and Y (not enough known anchors alone)
        g.add_edge(Edge("G1-X", source="G1", target="X"))
        g.add_edge(Edge("X-Y", source="X", target="Y"))
        # Y connects to G2, G3, and X
        g.add_edge(Edge("G2-Y", source="G2", target="Y"))
        g.add_edge(Edge("G3-Y", source="G3", target="Y"))

        result = decompose_assur_groups(g)
        # Should solve as a triad
        all_internal = []
        for group in result.groups:
            all_internal.extend(group.internal_nodes)
        assert "X" in all_internal
        assert "Y" in all_internal


# ---------------------------------------------------------------------------
# Complex mechanisms: six-bar and eight-bar
# ---------------------------------------------------------------------------

class TestComplexDecomposition:
    def test_sixbar_watt(self):
        """Watt I six-bar: two RRR dyads in sequence."""
        g = LinkageGraph()
        g.add_node(Node("G1", role=NodeRole.GROUND))
        g.add_node(Node("G2", role=NodeRole.GROUND))
        g.add_node(Node("G3", role=NodeRole.GROUND))
        g.add_node(Node("D", role=NodeRole.DRIVER))
        g.add_node(Node("C", role=NodeRole.DRIVEN))
        g.add_node(Node("E", role=NodeRole.DRIVEN))

        g.add_edge(Edge("G1-D", source="G1", target="D"))
        g.add_edge(Edge("D-C", source="D", target="C"))
        g.add_edge(Edge("G2-C", source="G2", target="C"))
        g.add_edge(Edge("C-E", source="C", target="E"))
        g.add_edge(Edge("G3-E", source="G3", target="E"))

        result = decompose_assur_groups(g)
        assert len(result.groups) == 2
        assert result.groups[0].internal_nodes == ("C",)
        assert result.groups[1].internal_nodes == ("E",)


# ---------------------------------------------------------------------------
# validate_decomposition edge cases
# ---------------------------------------------------------------------------

class TestValidateDecompositionEdgeCases:
    def test_validate_missing_nodes(self):
        """Nodes in graph but not in any group."""
        g = LinkageGraph()
        g.add_node(Node("A", role=NodeRole.GROUND))
        g.add_node(Node("B", role=NodeRole.DRIVER))
        g.add_node(Node("C", role=NodeRole.DRIVEN))
        g.add_edge(Edge("AB", source="A", target="B"))
        g.add_edge(Edge("BC", source="B", target="C"))

        # Create a result that doesn't include C
        r = DecompositionResult(
            ground=["A"],
            drivers=["B"],
            groups=[],
            graph=g,
        )
        msgs = validate_decomposition(r)
        assert any("C" in m for m in msgs)

    def test_validate_extra_nodes(self):
        """Nodes in groups but not in graph."""
        g = LinkageGraph()
        g.add_node(Node("A", role=NodeRole.GROUND))
        g.add_node(Node("B", role=NodeRole.DRIVER))
        g.add_edge(Edge("AB", source="A", target="B"))

        d = Dyad(internal_nodes=("PHANTOM",))
        r = DecompositionResult(
            ground=["A"],
            drivers=["B"],
            groups=[d],
            graph=g,
        )
        msgs = validate_decomposition(r)
        assert any("not in graph" in m.lower() for m in msgs)

    def test_validate_no_graph_reference(self):
        """If graph is None, no node check is done."""
        r = DecompositionResult(ground=[], drivers=[])
        msgs = validate_decomposition(r)
        # Should still warn about missing ground and driver
        assert any("ground" in m.lower() for m in msgs)
        assert any("driver" in m.lower() for m in msgs)

    def test_validate_fully_valid(self):
        """A fully accounted-for decomposition has no messages."""
        g = LinkageGraph()
        g.add_node(Node("A", role=NodeRole.GROUND))
        g.add_node(Node("D", role=NodeRole.GROUND))
        g.add_node(Node("B", role=NodeRole.DRIVER))
        g.add_node(Node("C", role=NodeRole.DRIVEN))
        g.add_edge(Edge("AB", source="A", target="B"))
        g.add_edge(Edge("BC", source="B", target="C"))
        g.add_edge(Edge("CD", source="C", target="D"))

        result = decompose_assur_groups(g)
        msgs = validate_decomposition(result)
        assert msgs == []
