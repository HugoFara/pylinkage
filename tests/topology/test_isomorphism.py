"""Tests for graph isomorphism detection."""

from pylinkage.hypergraph import (
    Edge,
    Hyperedge,
    HypergraphLinkage,
    JointType,
    Node,
    NodeRole,
)
from pylinkage.topology.isomorphism import are_isomorphic, canonical_form, canonical_hash


def _make_four_bar(prefix: str = "") -> HypergraphLinkage:
    """Create a four-bar linkage topology."""
    hg = HypergraphLinkage(name=f"{prefix}Four-bar")
    hg.add_node(Node(f"{prefix}A", role=NodeRole.GROUND))
    hg.add_node(Node(f"{prefix}B", role=NodeRole.DRIVER))
    hg.add_node(Node(f"{prefix}C", role=NodeRole.DRIVEN))
    hg.add_node(Node(f"{prefix}D", role=NodeRole.GROUND))
    hg.add_edge(Edge(f"{prefix}AB", f"{prefix}A", f"{prefix}B"))
    hg.add_edge(Edge(f"{prefix}BC", f"{prefix}B", f"{prefix}C"))
    hg.add_edge(Edge(f"{prefix}CD", f"{prefix}C", f"{prefix}D"))
    return hg


def _make_watt_six_bar() -> HypergraphLinkage:
    """Create a Watt-I six-bar: ternary coupler (B,C,E)."""
    hg = HypergraphLinkage(name="Watt-I")
    hg.add_node(Node("A", role=NodeRole.GROUND))
    hg.add_node(Node("B", role=NodeRole.DRIVER))
    hg.add_node(Node("C", role=NodeRole.DRIVEN))
    hg.add_node(Node("D", role=NodeRole.GROUND))
    hg.add_node(Node("E", role=NodeRole.DRIVEN))
    hg.add_node(Node("F", role=NodeRole.DRIVEN))
    hg.add_node(Node("G", role=NodeRole.GROUND))
    hg.add_edge(Edge("L1", "A", "B"))
    hg.add_edge(Edge("L3", "C", "D"))
    hg.add_edge(Edge("L4", "E", "F"))
    hg.add_edge(Edge("L5", "F", "G"))
    hg.add_hyperedge(Hyperedge("L2", nodes=("B", "C", "E")))
    return hg


def _make_stephenson_six_bar() -> HypergraphLinkage:
    """Create a Stephenson-I six-bar: ternary link on non-adjacent links.

    Stephenson differs from Watt: the two ternary links in the link-adjacency
    graph ARE adjacent (share a joint), whereas in Watt they are NOT.

    Here: ground link (A, D, G) is ternary, coupler (B, C, E) is ternary,
    and they share joint... actually, let's build the Stephenson topology:

    Link adjacency (Stephenson I):
      L0(ground, ternary) - L1 - L2(ternary) - L3 - L4 - L5 - L0
                            L0 - L2

    So L0 and L2 are both ternary AND adjacent.
    """
    hg = HypergraphLinkage(name="Stephenson-I")
    # Joints: A(L0-L1), B(L1-L2), C(L2-L3), D(L3-L4), E(L4-L5), F(L5-L0), G(L0-L2)
    hg.add_node(Node("A", role=NodeRole.GROUND))   # L0-L1
    hg.add_node(Node("B", role=NodeRole.DRIVER))    # L1-L2
    hg.add_node(Node("C", role=NodeRole.DRIVEN))    # L2-L3
    hg.add_node(Node("D", role=NodeRole.DRIVEN))    # L3-L4
    hg.add_node(Node("E", role=NodeRole.DRIVEN))    # L4-L5
    hg.add_node(Node("F", role=NodeRole.GROUND))    # L5-L0
    hg.add_node(Node("G", role=NodeRole.GROUND))    # L0-L2
    # Binary links
    hg.add_edge(Edge("L1", "A", "B"))   # L1: A-B
    hg.add_edge(Edge("L3", "C", "D"))   # L3: C-D
    hg.add_edge(Edge("L4", "D", "E"))   # L4: D-E
    hg.add_edge(Edge("L5", "E", "F"))   # L5: E-F
    # Ternary link L2: joints B, C, G
    hg.add_hyperedge(Hyperedge("L2", nodes=("B", "C", "G")))
    # Ground is ternary (A, F, G) — implicit, not an edge/hyperedge
    return hg


class TestCanonicalHash:
    """Tests for canonical_hash."""

    def test_identical_graphs_same_hash(self):
        g1 = _make_four_bar()
        g2 = _make_four_bar()
        assert canonical_hash(g1) == canonical_hash(g2)

    def test_relabeled_graph_same_hash(self):
        g1 = _make_four_bar()
        g2 = _make_four_bar(prefix="X_")
        assert canonical_hash(g1) == canonical_hash(g2)

    def test_different_topology_different_hash(self):
        g1 = _make_four_bar()
        g2 = _make_watt_six_bar()
        assert canonical_hash(g1) != canonical_hash(g2)

    def test_empty_graph(self):
        g = HypergraphLinkage()
        h = canonical_hash(g)
        assert isinstance(h, int)


class TestCanonicalForm:
    """Tests for canonical_form."""

    def test_identical_graphs_same_form(self):
        g1 = _make_four_bar()
        g2 = _make_four_bar()
        assert canonical_form(g1) == canonical_form(g2)

    def test_relabeled_graph_same_form(self):
        g1 = _make_four_bar()
        g2 = _make_four_bar(prefix="X_")
        assert canonical_form(g1) == canonical_form(g2)

    def test_different_topology_different_form(self):
        g1 = _make_watt_six_bar()
        g2 = _make_stephenson_six_bar()
        assert canonical_form(g1) != canonical_form(g2)

    def test_form_is_hashable(self):
        form = canonical_form(_make_four_bar())
        # Should be usable as dict key / set element
        s = {form}
        assert form in s

    def test_empty_graph(self):
        g = HypergraphLinkage()
        assert canonical_form(g) == ()

    def test_different_joint_type_different_form(self):
        """A graph with a prismatic joint differs from all-revolute."""
        g1 = _make_four_bar()
        g2 = _make_four_bar(prefix="P_")
        # Change one joint to prismatic
        g2.nodes["P_C"].joint_type = JointType.PRISMATIC
        assert canonical_form(g1) != canonical_form(g2)


class TestAreIsomorphic:
    """Tests for are_isomorphic."""

    def test_identical_four_bars(self):
        assert are_isomorphic(_make_four_bar(), _make_four_bar())

    def test_relabeled_four_bars(self):
        assert are_isomorphic(_make_four_bar(), _make_four_bar(prefix="Y_"))

    def test_four_bar_vs_six_bar(self):
        assert not are_isomorphic(_make_four_bar(), _make_watt_six_bar())

    def test_watt_vs_stephenson(self):
        assert not are_isomorphic(_make_watt_six_bar(), _make_stephenson_six_bar())

    def test_watt_self_isomorphic(self):
        assert are_isomorphic(_make_watt_six_bar(), _make_watt_six_bar())

    def test_different_node_count(self):
        g1 = _make_four_bar()
        g2 = _make_four_bar()
        g2.add_node(Node("extra", role=NodeRole.DRIVEN))
        assert not are_isomorphic(g1, g2)

    def test_different_role_not_isomorphic(self):
        """Same adjacency but different role assignment."""
        g1 = _make_four_bar()
        g2 = HypergraphLinkage()
        # Same structure but B is DRIVEN instead of DRIVER
        g2.add_node(Node("A", role=NodeRole.GROUND))
        g2.add_node(Node("B", role=NodeRole.DRIVEN))
        g2.add_node(Node("C", role=NodeRole.DRIVEN))
        g2.add_node(Node("D", role=NodeRole.GROUND))
        g2.add_edge(Edge("AB", "A", "B"))
        g2.add_edge(Edge("BC", "B", "C"))
        g2.add_edge(Edge("CD", "C", "D"))
        assert not are_isomorphic(g1, g2)
