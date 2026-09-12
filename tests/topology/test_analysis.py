"""Tests for the topology analysis module (DOF calculator)."""

from pylinkage.hypergraph import (
    Edge,
    Hyperedge,
    HypergraphLinkage,
    JointType,
    Node,
    NodeRole,
)
from pylinkage.topology import compute_dof, compute_mobility


def _make_four_bar() -> HypergraphLinkage:
    """Create a standard four-bar linkage topology (DOF=1).

    Topology:
        A (ground) -- AB -- B (driver) -- BC -- C (driven) -- CD -- D (ground)

    4 nodes (joints), 3 edges + 1 ground link = 4 links.
    DOF = 3*(4-1) - 2*4 = 9 - 8 = 1
    """
    hg = HypergraphLinkage(name="Four-bar")
    hg.add_node(Node("A", role=NodeRole.GROUND))
    hg.add_node(Node("B", role=NodeRole.DRIVER))
    hg.add_node(Node("C", role=NodeRole.DRIVEN))
    hg.add_node(Node("D", role=NodeRole.GROUND))
    hg.add_edge(Edge("AB", "A", "B"))
    hg.add_edge(Edge("BC", "B", "C"))
    hg.add_edge(Edge("CD", "C", "D"))
    return hg


def _make_six_bar_watt() -> HypergraphLinkage:
    """Create a Watt-I six-bar linkage topology (DOF=1).

    6 links, 7 revolute joints. Two four-bar loops sharing a ternary link.

    Link-adjacency (links as vertices, joints as edges):
        L0(ground) -- L1(crank) -- L2(ternary coupler) -- L3 -- L0
                                    L2 -- L4 -- L5 -- L0

    Joint-first (HypergraphLinkage) representation:
        7 nodes (joints), 4 binary-link edges + 1 ternary-link hyperedge + ground
        = 6 links total.
        DOF = 3*(6-1) - 2*7 = 15 - 14 = 1
    """
    hg = HypergraphLinkage(name="Watt-I six-bar")
    # 7 joints: 3 on ground link (A, D, G), 1 driver (B), 3 driven (C, E, F)
    hg.add_node(Node("A", role=NodeRole.GROUND))   # L0-L1
    hg.add_node(Node("B", role=NodeRole.DRIVER))    # L1-L2
    hg.add_node(Node("C", role=NodeRole.DRIVEN))    # L2-L3
    hg.add_node(Node("D", role=NodeRole.GROUND))    # L3-L0
    hg.add_node(Node("E", role=NodeRole.DRIVEN))    # L2-L4
    hg.add_node(Node("F", role=NodeRole.DRIVEN))    # L4-L5
    hg.add_node(Node("G", role=NodeRole.GROUND))    # L5-L0
    # 4 binary links (edges)
    hg.add_edge(Edge("L1", "A", "B"))   # crank
    hg.add_edge(Edge("L3", "C", "D"))   # connecting rod 1
    hg.add_edge(Edge("L4", "E", "F"))   # connecting rod 2
    hg.add_edge(Edge("L5", "F", "G"))   # rocker
    # 1 ternary link (hyperedge): coupler connecting joints B, C, E
    hg.add_hyperedge(Hyperedge("L2", nodes=("B", "C", "E")))
    return hg


class TestComputeDof:
    """Tests for the compute_dof function."""

    def test_four_bar_dof_is_1(self):
        """A standard four-bar linkage has DOF=1."""
        hg = _make_four_bar()
        assert compute_dof(hg) == 1

    def test_single_crank_dof_is_1(self):
        """A crank pinned to the ground and free at its tip has DOF=1.

        2 links (ground, crank arm) and 1 joint: the tip ``B`` belongs to
        the crank alone, so it is not a joint.
        DOF = 3*(2-1) - 2*1 = 1
        """
        hg = HypergraphLinkage()
        hg.add_node(Node("A", role=NodeRole.GROUND))
        hg.add_node(Node("B", role=NodeRole.DRIVER))
        hg.add_edge(Edge("AB", "A", "B"))
        assert compute_dof(hg) == 1

    def test_empty_graph_dof_is_0(self):
        """An empty graph has DOF=0 (just ground, no joints)."""
        hg = HypergraphLinkage()
        # 0 nodes, 0 edges, 1 ground link
        # DOF = 3*(1-1) - 0 = 0
        assert compute_dof(hg) == 0

    def test_watt_six_bar_dof_is_1(self):
        """A Watt-I six-bar (6 links, 7 joints, ternary coupler) has DOF=1."""
        hg = _make_six_bar_watt()
        assert compute_dof(hg) == 1

    def test_slider_crank_dof_is_1(self):
        """A slider-crank mechanism has DOF=1.

        3 nodes, 2 edges + ground = 3 links, 3 joints.
        But one joint is prismatic — still 1-DOF in Grübler.
        DOF = 3*(3-1) - 2*3 = 6 - 6 = 0?

        Actually a slider-crank has 4 links and 4 joints:
        ground, crank, connecting rod, slider block.
        Let me model it properly.
        """
        hg = HypergraphLinkage()
        hg.add_node(Node("A", role=NodeRole.GROUND))
        hg.add_node(Node("B", role=NodeRole.DRIVER))
        hg.add_node(Node("C", role=NodeRole.DRIVEN))
        hg.add_node(Node("D", role=NodeRole.GROUND))  # slider ground
        hg.add_edge(Edge("AB", "A", "B"))
        hg.add_edge(Edge("BC", "B", "C"))
        hg.add_edge(Edge("CD", "C", "D"))
        # Same topology as four-bar: 4 nodes, 3 edges + ground = 4 links
        # DOF = 3*(4-1) - 2*4 = 9 - 8 = 1
        assert compute_dof(hg) == 1


class TestComputeMobility:
    """Tests for the compute_mobility function."""

    def test_four_bar_counts(self):
        """Check link and joint counts for a four-bar."""
        hg = _make_four_bar()
        info = compute_mobility(hg)
        assert info.dof == 1
        assert info.num_links == 4  # 3 edges + 1 ground
        assert info.num_full_joints == 4
        assert info.num_half_joints == 0

    def test_watt_six_bar_counts(self):
        """Check link and joint counts for a Watt-I six-bar."""
        hg = _make_six_bar_watt()
        info = compute_mobility(hg)
        assert info.dof == 1
        assert info.num_links == 6  # 4 edges + 1 hyperedge + 1 ground
        assert info.num_full_joints == 7

    def test_triangle_is_rigid(self):
        """A triangle with one side on the ground is a structure (DOF=0).

        The edge between the two ground nodes is part of the ground
        link: 3 links (ground, BC, CA) and 3 joints.
        DOF = 3*(3-1) - 2*3 = 0
        """
        hg = HypergraphLinkage()
        hg.add_node(Node("A", role=NodeRole.GROUND))
        hg.add_node(Node("B", role=NodeRole.GROUND))
        hg.add_node(Node("C", role=NodeRole.DRIVEN))
        hg.add_edge(Edge("AB", "A", "B"))
        hg.add_edge(Edge("BC", "B", "C"))
        hg.add_edge(Edge("CA", "C", "A"))
        info = compute_mobility(hg)
        assert info.num_links == 3
        assert info.num_full_joints == 3
        assert info.dof == 0


def _make_coupler_four_bar(inner_edges: bool, hyperedge: bool) -> HypergraphLinkage:
    """Four-bar with a coupler point ``P`` rigid with the coupler B-C.

    The triangle B-C-P can be written as its three edges, as a hyperedge,
    or as both (what ``Linkage.to_hypergraph`` emits for a ``FixedDyad``).
    """
    hg = HypergraphLinkage(name="Coupler four-bar")
    for node_id, role in [
        ("A", NodeRole.GROUND), ("B", NodeRole.DRIVER), ("C", NodeRole.DRIVEN),
        ("P", NodeRole.DRIVEN), ("D", NodeRole.GROUND),
    ]:
        hg.add_node(Node(node_id, role=role))
    hg.add_edge(Edge("AB", "A", "B"))
    hg.add_edge(Edge("CD", "C", "D"))
    if inner_edges or not hyperedge:
        hg.add_edge(Edge("BC", "B", "C"))
    if inner_edges:
        hg.add_edge(Edge("BP", "B", "P"))
        hg.add_edge(Edge("CP", "C", "P"))
    if hyperedge:
        hg.add_hyperedge(Hyperedge("coupler", ("B", "C", "P")))
    return hg


class TestRigidBodies:
    """Links are rigid bodies, whichever way the graph spells them."""

    def test_hyperedge_over_its_own_edges_is_one_link(self):
        """A hyperedge labelling a triangle of edges does not add links."""
        info = compute_mobility(_make_coupler_four_bar(inner_edges=True, hyperedge=True))
        assert info.num_links == 4
        assert info.num_full_joints == 4
        assert info.dof == 1

    def test_hyperedge_only_triangle(self):
        """A hyperedge with no edges among its nodes is one link."""
        info = compute_mobility(_make_coupler_four_bar(inner_edges=False, hyperedge=True))
        assert info.num_links == 4
        assert info.num_full_joints == 4
        assert info.dof == 1

    def test_edge_triangle_has_same_dof(self):
        """Three bars closing a triangle count as bars, with the same DOF."""
        info = compute_mobility(_make_coupler_four_bar(inner_edges=True, hyperedge=False))
        assert info.num_links == 6
        assert info.num_full_joints == 7
        assert info.dof == 1

    def test_coupler_point_is_not_a_joint(self):
        """A node in a single body adds no joint."""
        plain = compute_mobility(_make_four_bar())
        with_point = compute_mobility(_make_coupler_four_bar(inner_edges=False, hyperedge=True))
        assert with_point.num_full_joints == plain.num_full_joints

    def test_pin_shared_by_three_links_is_two_joints(self):
        """Two ternary links and a bar meeting at one pin: a multiple joint."""
        hg = HypergraphLinkage()
        for node_id, role in [
            ("A", NodeRole.GROUND), ("B", NodeRole.DRIVER), ("C", NodeRole.DRIVEN),
            ("D", NodeRole.GROUND), ("E", NodeRole.DRIVEN), ("F", NodeRole.GROUND),
        ]:
            hg.add_node(Node(node_id, role=role))
        hg.add_edge(Edge("AB", "A", "B"))
        hg.add_edge(Edge("CD", "C", "D"))
        hg.add_edge(Edge("CE", "C", "E"))
        hg.add_edge(Edge("EF", "E", "F"))
        hg.add_hyperedge(Hyperedge("coupler", ("B", "C")))
        info = compute_mobility(hg)
        # ground, AB, coupler, CD, CE, EF
        assert info.num_links == 6
        # A, B, D, E, F once; C joins coupler, CD and CE: twice
        assert info.num_full_joints == 7
        assert info.dof == 1

    def test_slider_crank_with_prismatic_node(self):
        """A slider block on a ground guide: 4 links, 4 joints, DOF=1."""
        hg = HypergraphLinkage()
        hg.add_node(Node("A", role=NodeRole.GROUND))
        hg.add_node(Node("B", role=NodeRole.DRIVER))
        hg.add_node(Node("S", role=NodeRole.DRIVEN, joint_type=JointType.PRISMATIC))
        hg.add_node(Node("D", role=NodeRole.GROUND))
        hg.add_edge(Edge("AB", "A", "B"))
        hg.add_edge(Edge("BS", "B", "S"))
        # The guide line A-D with the slider on it
        hg.add_hyperedge(Hyperedge("guide", ("A", "D", "S")))
        info = compute_mobility(hg)
        assert info.num_links == 4
        assert info.num_full_joints == 4
        assert info.dof == 1

    def test_catalog_counts_are_unchanged(self):
        """The built-in catalog records the counts it was generated with."""
        from pylinkage.topology import load_catalog

        for entry in load_catalog():
            info = compute_mobility(entry.to_graph())
            assert (info.num_links, info.dof) == (entry.num_links, entry.dof), entry.id

    def test_fixed_dyad_round_trip(self):
        """``Linkage.to_hypergraph`` of a coupler point stays a four-bar."""
        from pylinkage.actuators import Crank
        from pylinkage.components import Ground
        from pylinkage.dyads import FixedDyad, RRRDyad
        from pylinkage.simulation import Linkage

        a = Ground(0, 0, name="A")
        d = Ground(3, 0, name="D")
        crank = Crank(anchor=a, radius=1, angular_velocity=0.1, name="B")
        c = RRRDyad(anchor1=crank.output, anchor2=d, distance1=2.5, distance2=2.0, name="C")
        p = FixedDyad(anchor1=crank.output, anchor2=c, distance=1.5, angle=0.8, name="P")
        linkage = Linkage([a, d, crank, c, p])
        linkage.rebuild()
        hg, _ = linkage.to_hypergraph()
        info = compute_mobility(hg)
        assert info.num_links == 4
        assert info.dof == 1
