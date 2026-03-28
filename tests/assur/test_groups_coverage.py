"""Additional tests for groups.py to increase coverage.

Covers: Dyad/Triad creation with various signatures, solver_category,
_check_dyad_signature branches, identify_group_type, identify_dyad_type,
backwards-compatible aliases (DyadRRR, DyadRRP, etc.), and DYAD_TYPES registry.
"""


from pylinkage.assur import (
    DYAD_TYPES,
    Dyad,
    DyadPRR,
    DyadRPR,
    DyadRRP,
    DyadRRR,
    Triad,
    identify_dyad_type,
    identify_group_type,
)
from pylinkage.assur._types import JointType, NodeRole
from pylinkage.assur.graph import Edge, LinkageGraph, Node
from pylinkage.assur.groups import DyadPP, _count_prismatic

# ---------------------------------------------------------------------------
# _count_prismatic
# ---------------------------------------------------------------------------

class TestCountPrismatic:
    def test_no_prismatic(self):
        assert _count_prismatic("RRR") == 0

    def test_one_prismatic(self):
        assert _count_prismatic("RRP") == 1
        assert _count_prismatic("RPR") == 1
        assert _count_prismatic("PRR") == 1

    def test_two_prismatic(self):
        assert _count_prismatic("PP") == 2
        assert _count_prismatic("PPR") == 2
        assert _count_prismatic("RPP") == 2

    def test_all_prismatic(self):
        assert _count_prismatic("PPP") == 3

    def test_empty(self):
        assert _count_prismatic("") == 0


# ---------------------------------------------------------------------------
# Dyad creation with various signatures
# ---------------------------------------------------------------------------

class TestDyadCreation:
    def test_default_signature_is_rrr(self):
        d = Dyad()
        assert d.joint_signature == "RRR"

    def test_custom_signature(self):
        d = Dyad(_signature="RRP")
        assert d.joint_signature == "RRP"

    def test_group_class_always_1(self):
        for sig in ("RRR", "RRP", "RPR", "PRR", "PP"):
            d = Dyad(_signature=sig)
            assert d.group_class == 1

    def test_line_nodes_default_none(self):
        d = Dyad()
        assert d.line_node1 is None
        assert d.line_node2 is None
        assert d.line2_node1 is None
        assert d.line2_node2 is None

    def test_line_nodes_set(self):
        d = Dyad(
            _signature="RRP",
            line_node1="L1",
            line_node2="L2",
        )
        assert d.line_node1 == "L1"
        assert d.line_node2 == "L2"

    def test_pp_dyad_second_line_nodes(self):
        d = Dyad(
            _signature="PP",
            line_node1="L1",
            line_node2="L2",
            line2_node1="L3",
            line2_node2="L4",
        )
        assert d.line2_node1 == "L3"
        assert d.line2_node2 == "L4"


# ---------------------------------------------------------------------------
# Triad creation
# ---------------------------------------------------------------------------

class TestTriadCreation:
    def test_default_signature(self):
        t = Triad()
        assert t.joint_signature == "RRRRRR"

    def test_custom_signature(self):
        t = Triad(_signature="RRRRRP")
        assert t.joint_signature == "RRRRRP"

    def test_group_class_is_2(self):
        t = Triad()
        assert t.group_class == 2

    def test_edge_map_default_empty(self):
        t = Triad()
        assert t.edge_map == {}

    def test_edge_map_stored(self):
        em = {"e1": ("A", "B"), "e2": ("B", "C")}
        t = Triad(edge_map=em)
        assert t.edge_map == em


# ---------------------------------------------------------------------------
# solver_category property
# ---------------------------------------------------------------------------

class TestSolverCategory:
    def test_rrr_is_circle_circle(self):
        d = Dyad(_signature="RRR")
        assert d.solver_category == "circle_circle"

    def test_rrp_is_circle_line(self):
        d = Dyad(_signature="RRP")
        assert d.solver_category == "circle_line"

    def test_rpr_is_circle_line(self):
        d = Dyad(_signature="RPR")
        assert d.solver_category == "circle_line"

    def test_prr_is_circle_line(self):
        d = Dyad(_signature="PRR")
        assert d.solver_category == "circle_line"

    def test_pp_is_line_line(self):
        d = Dyad(_signature="PP")
        assert d.solver_category == "line_line"

    def test_ppr_is_line_line(self):
        d = Dyad(_signature="PPR")
        assert d.solver_category == "line_line"

    def test_ppp_is_line_line(self):
        d = Dyad(_signature="PPP")
        assert d.solver_category == "line_line"

    def test_triad_is_newton_raphson(self):
        t = Triad()
        assert t.solver_category == "newton_raphson"


# ---------------------------------------------------------------------------
# _check_dyad_signature via Dyad.can_form (all branches)
# ---------------------------------------------------------------------------

def _make_rrr_graph():
    """Graph with one revolute internal node and two revolute anchors."""
    g = LinkageGraph()
    g.add_node(Node("A", role=NodeRole.GROUND, joint_type=JointType.REVOLUTE))
    g.add_node(Node("B", role=NodeRole.GROUND, joint_type=JointType.REVOLUTE))
    g.add_node(Node("C", role=NodeRole.DRIVEN, joint_type=JointType.REVOLUTE))
    g.add_edge(Edge("AC", source="A", target="C"))
    g.add_edge(Edge("BC", source="B", target="C"))
    return g


def _make_rpr_graph():
    """Graph with a prismatic internal node and two revolute anchors."""
    g = LinkageGraph()
    g.add_node(Node("A", role=NodeRole.GROUND, joint_type=JointType.REVOLUTE))
    g.add_node(Node("B", role=NodeRole.GROUND, joint_type=JointType.REVOLUTE))
    g.add_node(Node("C", role=NodeRole.DRIVEN, joint_type=JointType.PRISMATIC))
    g.add_edge(Edge("AC", source="A", target="C"))
    g.add_edge(Edge("BC", source="B", target="C"))
    return g


def _make_prr_graph():
    """Graph: revolute internal, one prismatic anchor, one revolute anchor."""
    g = LinkageGraph()
    g.add_node(Node("A", role=NodeRole.GROUND, joint_type=JointType.PRISMATIC))
    g.add_node(Node("B", role=NodeRole.GROUND, joint_type=JointType.REVOLUTE))
    g.add_node(Node("C", role=NodeRole.DRIVEN, joint_type=JointType.REVOLUTE))
    g.add_edge(Edge("AC", source="A", target="C"))
    g.add_edge(Edge("BC", source="B", target="C"))
    return g


class TestCheckDyadSignatureRRR:
    def test_valid_rrr(self):
        g = _make_rrr_graph()
        assert Dyad.can_form(["C"], ["A", "B"], g, signature="RRR")

    def test_rrr_prismatic_internal_fails(self):
        g = _make_rpr_graph()
        assert not Dyad.can_form(["C"], ["A", "B"], g, signature="RRR")

    def test_rrr_insufficient_anchors(self):
        g = _make_rrr_graph()
        assert not Dyad.can_form(["C"], ["A"], g, signature="RRR")

    def test_rrr_no_edge_to_anchor_fails(self):
        g = LinkageGraph()
        g.add_node(Node("A", role=NodeRole.GROUND))
        g.add_node(Node("B", role=NodeRole.GROUND))
        g.add_node(Node("C", role=NodeRole.DRIVEN, joint_type=JointType.REVOLUTE))
        g.add_edge(Edge("AC", source="A", target="C"))
        # No edge BC
        assert not Dyad.can_form(["C"], ["A", "B"], g, signature="RRR")


class TestCheckDyadSignatureRPR:
    def test_valid_rpr(self):
        g = _make_rpr_graph()
        assert Dyad.can_form(["C"], ["A", "B"], g, signature="RPR")

    def test_rpr_revolute_internal_fails(self):
        g = _make_rrr_graph()
        assert not Dyad.can_form(["C"], ["A", "B"], g, signature="RPR")

    def test_rpr_no_revolute_anchor_with_edge_fails(self):
        """RPR requires at least one revolute anchor connected via edge."""
        g = LinkageGraph()
        g.add_node(Node("A", role=NodeRole.GROUND, joint_type=JointType.PRISMATIC))
        g.add_node(Node("C", role=NodeRole.DRIVEN, joint_type=JointType.PRISMATIC))
        # No edge at all
        assert not Dyad.can_form(["C"], ["A"], g, signature="RPR")


class TestCheckDyadSignaturePRR:
    def test_valid_prr(self):
        g = _make_prr_graph()
        assert Dyad.can_form(["C"], ["A", "B"], g, signature="PRR")

    def test_prr_prismatic_internal_fails(self):
        g = _make_rpr_graph()
        assert not Dyad.can_form(["C"], ["A", "B"], g, signature="PRR")

    def test_prr_no_prismatic_anchor_fails(self):
        """PRR needs at least one prismatic + one revolute anchor."""
        g = _make_rrr_graph()
        assert not Dyad.can_form(["C"], ["A", "B"], g, signature="PRR")

    def test_prr_no_revolute_anchor_fails(self):
        """PRR needs both prismatic and revolute anchors."""
        g = LinkageGraph()
        g.add_node(Node("A", role=NodeRole.GROUND, joint_type=JointType.PRISMATIC))
        g.add_node(Node("B", role=NodeRole.GROUND, joint_type=JointType.PRISMATIC))
        g.add_node(Node("C", role=NodeRole.DRIVEN, joint_type=JointType.REVOLUTE))
        g.add_edge(Edge("AC", source="A", target="C"))
        g.add_edge(Edge("BC", source="B", target="C"))
        assert not Dyad.can_form(["C"], ["A", "B"], g, signature="PRR")


class TestCheckDyadSignatureRRP:
    def test_rrp_with_enough_edges(self):
        """RRP branch: need >= 3 edges from internal node."""
        g = LinkageGraph()
        g.add_node(Node("A", role=NodeRole.GROUND))
        g.add_node(Node("B", role=NodeRole.GROUND))
        g.add_node(Node("L", role=NodeRole.GROUND))
        g.add_node(Node("C", role=NodeRole.DRIVEN, joint_type=JointType.REVOLUTE))
        g.add_edge(Edge("AC", source="A", target="C"))
        g.add_edge(Edge("BC", source="B", target="C"))
        g.add_edge(Edge("LC", source="L", target="C"))
        assert Dyad.can_form(["C"], ["A", "B", "L"], g, signature="RRP")

    def test_rrp_insufficient_edges_fails(self):
        g = _make_rrr_graph()
        # Only 2 edges, RRP needs >= 3
        assert not Dyad.can_form(["C"], ["A", "B"], g, signature="RRP")


class TestCheckDyadSignaturePP:
    def test_pp_with_enough_prismatic_anchors(self):
        g = LinkageGraph()
        g.add_node(Node("A", joint_type=JointType.PRISMATIC))
        g.add_node(Node("B", joint_type=JointType.PRISMATIC))
        g.add_node(Node("D", joint_type=JointType.REVOLUTE))
        g.add_node(Node("E", joint_type=JointType.REVOLUTE))
        g.add_node(Node("C", role=NodeRole.DRIVEN, joint_type=JointType.REVOLUTE))
        g.add_edge(Edge("AC", source="A", target="C"))
        g.add_edge(Edge("BC", source="B", target="C"))
        g.add_edge(Edge("DC", source="D", target="C"))
        g.add_edge(Edge("EC", source="E", target="C"))
        assert Dyad.can_form(["C"], ["A", "B", "D", "E"], g, signature="PP")

    def test_pp_insufficient_anchors_fails(self):
        g = _make_rrr_graph()
        assert not Dyad.can_form(["C"], ["A", "B"], g, signature="PP")

    def test_pp_insufficient_prismatic_anchors_fails(self):
        g = LinkageGraph()
        g.add_node(Node("A", joint_type=JointType.REVOLUTE))
        g.add_node(Node("B", joint_type=JointType.REVOLUTE))
        g.add_node(Node("D", joint_type=JointType.REVOLUTE))
        g.add_node(Node("E", joint_type=JointType.REVOLUTE))
        g.add_node(Node("C", role=NodeRole.DRIVEN))
        g.add_edge(Edge("AC", source="A", target="C"))
        g.add_edge(Edge("BC", source="B", target="C"))
        g.add_edge(Edge("DC", source="D", target="C"))
        g.add_edge(Edge("EC", source="E", target="C"))
        assert not Dyad.can_form(["C"], ["A", "B", "D", "E"], g, signature="PP")


# ---------------------------------------------------------------------------
# Dyad.can_form without signature (tries all)
# ---------------------------------------------------------------------------

class TestDyadCanFormAutoDetect:
    def test_auto_detects_rrr(self):
        g = _make_rrr_graph()
        assert Dyad.can_form(["C"], ["A", "B"], g)

    def test_auto_detects_rpr(self):
        g = _make_rpr_graph()
        assert Dyad.can_form(["C"], ["A", "B"], g)

    def test_no_internal_node_fails(self):
        g = _make_rrr_graph()
        assert not Dyad.can_form([], ["A", "B"], g)

    def test_two_internal_nodes_fails(self):
        g = _make_rrr_graph()
        assert not Dyad.can_form(["A", "C"], ["B"], g)

    def test_nonexistent_internal_node_fails(self):
        g = _make_rrr_graph()
        assert not Dyad.can_form(["Z"], ["A", "B"], g)

    def test_returns_false_when_nothing_matches(self):
        g = LinkageGraph()
        g.add_node(Node("C", role=NodeRole.DRIVEN, joint_type=JointType.REVOLUTE))
        assert not Dyad.can_form(["C"], [], g)


# ---------------------------------------------------------------------------
# Triad.can_form
# ---------------------------------------------------------------------------

def _make_triad_graph():
    """Create a graph that can form a triad: 2 internals, 3 anchors."""
    g = LinkageGraph()
    for nid in ("A", "B", "C"):
        g.add_node(Node(nid, role=NodeRole.GROUND))
    g.add_node(Node("X", role=NodeRole.DRIVEN))
    g.add_node(Node("Y", role=NodeRole.DRIVEN))
    g.add_edge(Edge("AX", source="A", target="X"))
    g.add_edge(Edge("BX", source="B", target="X"))
    g.add_edge(Edge("XY", source="X", target="Y"))
    g.add_edge(Edge("CY", source="C", target="Y"))
    return g


class TestTriadCanForm:
    def test_valid_triad(self):
        g = _make_triad_graph()
        assert Triad.can_form(["X", "Y"], ["A", "B", "C"], g)

    def test_wrong_internal_count(self):
        g = _make_triad_graph()
        assert not Triad.can_form(["X"], ["A", "B", "C"], g)
        assert not Triad.can_form(["X", "Y", "A"], ["B", "C"], g)

    def test_insufficient_anchors(self):
        g = _make_triad_graph()
        assert not Triad.can_form(["X", "Y"], ["A", "B"], g)

    def test_nonexistent_internal(self):
        g = _make_triad_graph()
        assert not Triad.can_form(["X", "MISSING"], ["A", "B", "C"], g)

    def test_insufficient_edges(self):
        g = LinkageGraph()
        for nid in ("A", "B", "C"):
            g.add_node(Node(nid, role=NodeRole.GROUND))
        g.add_node(Node("X", role=NodeRole.DRIVEN))
        g.add_node(Node("Y", role=NodeRole.DRIVEN))
        g.add_edge(Edge("AX", source="A", target="X"))
        # Only 1 edge -- needs >= 4
        assert not Triad.can_form(["X", "Y"], ["A", "B", "C"], g)


# ---------------------------------------------------------------------------
# identify_group_type
# ---------------------------------------------------------------------------

class TestIdentifyGroupType:
    def test_identifies_rrr_dyad(self):
        g = _make_rrr_graph()
        result = identify_group_type(["C"], ["A", "B"], g)
        assert isinstance(result, Dyad)
        assert result.joint_signature == "RRR"

    def test_identifies_rpr_dyad(self):
        g = _make_rpr_graph()
        result = identify_group_type(["C"], ["A", "B"], g)
        assert isinstance(result, Dyad)
        assert result.joint_signature == "RPR"

    def test_identifies_prr_graph_as_dyad(self):
        """PRR graph matches RRR first since RRR only checks revolute internal + edges."""
        g = _make_prr_graph()
        result = identify_group_type(["C"], ["A", "B"], g)
        assert isinstance(result, Dyad)
        # RRR is tried first and matches (revolute internal, 2 edges)
        assert result.joint_signature == "RRR"

    def test_identifies_triad(self):
        g = _make_triad_graph()
        result = identify_group_type(["X", "Y"], ["A", "B", "C"], g)
        assert isinstance(result, Triad)

    def test_returns_none_for_no_match_single(self):
        g = LinkageGraph()
        g.add_node(Node("C", role=NodeRole.DRIVEN))
        result = identify_group_type(["C"], [], g)
        assert result is None

    def test_returns_none_for_nonexistent_node(self):
        g = LinkageGraph()
        g.add_node(Node("A", role=NodeRole.GROUND))
        result = identify_group_type(["MISSING"], ["A"], g)
        assert result is None

    def test_returns_none_for_no_match_pair(self):
        """Two internal nodes that can't form a triad."""
        g = LinkageGraph()
        g.add_node(Node("X", role=NodeRole.DRIVEN))
        g.add_node(Node("Y", role=NodeRole.DRIVEN))
        g.add_node(Node("A", role=NodeRole.GROUND))
        result = identify_group_type(["X", "Y"], ["A"], g)
        assert result is None

    def test_returns_none_for_three_internal(self):
        """Three internal nodes -- not handled."""
        g = LinkageGraph()
        for nid in ("X", "Y", "Z"):
            g.add_node(Node(nid, role=NodeRole.DRIVEN))
        result = identify_group_type(["X", "Y", "Z"], [], g)
        assert result is None


# ---------------------------------------------------------------------------
# identify_dyad_type
# ---------------------------------------------------------------------------

class TestIdentifyDyadType:
    def test_identifies_rrr(self):
        g = _make_rrr_graph()
        cls = identify_dyad_type(["C"], ["A", "B"], g)
        assert cls is DyadRRR

    def test_identifies_rpr(self):
        g = _make_rpr_graph()
        cls = identify_dyad_type(["C"], ["A", "B"], g)
        assert cls is DyadRPR

    def test_identifies_prr_graph_matches_rrr_first(self):
        """PRR graph matches RRR first in DYAD_TYPES iteration order."""
        g = _make_prr_graph()
        cls = identify_dyad_type(["C"], ["A", "B"], g)
        # RRR is iterated first and matches (revolute internal, 2 edges)
        assert cls is DyadRRR

    def test_wrong_internal_count(self):
        g = _make_rrr_graph()
        assert identify_dyad_type([], ["A", "B"], g) is None
        assert identify_dyad_type(["A", "C"], ["B"], g) is None

    def test_nonexistent_node(self):
        g = _make_rrr_graph()
        assert identify_dyad_type(["MISSING"], ["A", "B"], g) is None

    def test_no_match(self):
        g = LinkageGraph()
        g.add_node(Node("C", role=NodeRole.DRIVEN))
        assert identify_dyad_type(["C"], [], g) is None


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

class TestBackwardsCompatAliases:
    def test_dyad_rrr_creates_correct_signature(self):
        d = DyadRRR()
        assert d.joint_signature == "RRR"
        assert isinstance(d, Dyad)

    def test_dyad_rrp_creates_correct_signature(self):
        d = DyadRRP()
        assert d.joint_signature == "RRP"

    def test_dyad_rpr_creates_correct_signature(self):
        d = DyadRPR()
        assert d.joint_signature == "RPR"

    def test_dyad_prr_creates_correct_signature(self):
        d = DyadPRR()
        assert d.joint_signature == "PRR"

    def test_dyad_pp_creates_correct_signature(self):
        d = DyadPP()
        assert d.joint_signature == "PP"

    def test_alias_class_names(self):
        assert DyadRRR.__name__ == "DyadRRR"
        assert DyadRRP.__name__ == "DyadRRP"
        assert DyadRPR.__name__ == "DyadRPR"
        assert DyadPRR.__name__ == "DyadPRR"
        assert DyadPP.__name__ == "DyadPP"

    def test_alias_accepts_kwargs(self):
        d = DyadRRR(internal_nodes=("C",), anchor_nodes=("A", "B"))
        assert d.internal_nodes == ("C",)
        assert d.anchor_nodes == ("A", "B")
        assert d.joint_signature == "RRR"

    def test_alias_signature_override(self):
        """If you pass _signature kwarg, it overrides the default only if not set."""
        d = DyadRRR()
        # The factory uses setdefault, so the alias signature wins
        assert d.joint_signature == "RRR"


# ---------------------------------------------------------------------------
# DYAD_TYPES registry
# ---------------------------------------------------------------------------

class TestDyadTypesRegistry:
    def test_all_expected_keys(self):
        expected = {"RRR", "RRP", "RPR", "PRR", "PP", "PRP", "PPR"}
        assert set(DYAD_TYPES.keys()) == expected

    def test_pp_aliases(self):
        """PP, PRP, PPR all map to DyadPP."""
        assert DYAD_TYPES["PP"] is DyadPP
        assert DYAD_TYPES["PRP"] is DyadPP
        assert DYAD_TYPES["PPR"] is DyadPP
