"""Tests for assur/mechanism_conversion.py (graph <-> Mechanism conversion)."""

import math

import pytest

from pylinkage._types import NodeRole
from pylinkage.assur.graph import Edge, LinkageGraph, Node
from pylinkage.assur.mechanism_conversion import graph_to_mechanism, mechanism_to_graph
from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.mechanism import (
    DriverLink,
    GroundJoint,
    GroundLink,
    Link,
    Mechanism,
    RevoluteJoint,
)


def _make_fourbar_graph() -> tuple[LinkageGraph, Dimensions]:
    """Build a simple four-bar as a LinkageGraph + Dimensions."""
    graph = LinkageGraph(name="Four-bar")

    graph.add_node(Node("A", role=NodeRole.GROUND))
    graph.add_node(Node("D", role=NodeRole.GROUND))
    graph.add_node(Node("B", role=NodeRole.DRIVER))
    graph.add_node(Node("C", role=NodeRole.DRIVEN))

    graph.add_edge(Edge("AB", source="A", target="B"))
    graph.add_edge(Edge("BC", source="B", target="C"))
    graph.add_edge(Edge("CD", source="C", target="D"))

    dims = Dimensions(
        node_positions={
            "A": (0.0, 0.0),
            "D": (3.0, 0.0),
            "B": (0.0, 1.0),
            "C": (3.0, 2.0),
        },
        driver_angles={"B": DriverAngle(angular_velocity=0.1, initial_angle=math.pi / 2)},
        edge_distances={"AB": 1.0, "BC": 3.0, "CD": 1.0},
        name="Four-bar",
    )
    return graph, dims


class TestGraphToMechanism:
    """Tests for graph_to_mechanism()."""

    def test_fourbar_basic_structure(self):
        """Conversion produces a Mechanism with expected joints and links."""
        graph, dims = _make_fourbar_graph()
        mech = graph_to_mechanism(graph, dims)

        assert isinstance(mech, Mechanism)
        assert mech.name == "Four-bar"

        # Should have joints for A, D (ground), B (driver output), C (driven)
        joint_ids = {j.id for j in mech.joints}
        assert "A" in joint_ids
        assert "D" in joint_ids
        assert "B" in joint_ids
        assert "C" in joint_ids

    def test_fourbar_ground_joints(self):
        """Ground nodes become GroundJoint instances."""
        graph, dims = _make_fourbar_graph()
        mech = graph_to_mechanism(graph, dims)

        ground_joints = [j for j in mech.joints if isinstance(j, GroundJoint)]
        ground_ids = {j.id for j in ground_joints}
        assert "A" in ground_ids
        assert "D" in ground_ids

    def test_fourbar_ground_link_exists(self):
        """A GroundLink is created from ground joints."""
        graph, dims = _make_fourbar_graph()
        mech = graph_to_mechanism(graph, dims)

        ground_links = [lk for lk in mech.links if isinstance(lk, GroundLink)]
        assert len(ground_links) == 1
        assert len(ground_links[0].joints) == 2

    def test_fourbar_driver_link(self):
        """A DriverLink is created for the driver node."""
        graph, dims = _make_fourbar_graph()
        mech = graph_to_mechanism(graph, dims)

        driver_links = [lk for lk in mech.links if isinstance(lk, DriverLink)]
        assert len(driver_links) == 1
        dl = driver_links[0]
        assert dl.angular_velocity == pytest.approx(0.1)

    def test_fourbar_positions(self):
        """Joint positions come from Dimensions."""
        graph, dims = _make_fourbar_graph()
        mech = graph_to_mechanism(graph, dims)

        joint_map = {j.id: j for j in mech.joints}
        assert joint_map["A"].position == pytest.approx((0.0, 0.0))
        assert joint_map["D"].position == pytest.approx((3.0, 0.0))
        assert joint_map["B"].position == pytest.approx((0.0, 1.0))
        assert joint_map["C"].position == pytest.approx((3.0, 2.0))

    def test_fourbar_driven_links(self):
        """The driven node C gets Link objects connecting to its anchors."""
        graph, dims = _make_fourbar_graph()
        mech = graph_to_mechanism(graph, dims)

        # Exclude ground link and driver link; remaining are driven links
        plain_links = [lk for lk in mech.links if not isinstance(lk, (GroundLink, DriverLink))]
        # C should be connected to B and D via two links
        assert len(plain_links) == 2

    def test_no_ground_nodes(self):
        """Graph with no ground nodes produces a Mechanism with no GroundLink."""
        graph = LinkageGraph(name="no-ground")
        graph.add_node(Node("X", role=NodeRole.DRIVER))
        dims = Dimensions(node_positions={"X": (0.0, 0.0)})

        mech = graph_to_mechanism(graph, dims)
        ground_links = [lk for lk in mech.links if isinstance(lk, GroundLink)]
        assert len(ground_links) == 0

    def test_missing_position_defaults_to_zero(self):
        """Nodes without positions in Dimensions default to (0, 0)."""
        graph = LinkageGraph(name="test")
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("B", role=NodeRole.DRIVER))
        graph.add_edge(Edge("AB", source="A", target="B"))

        dims = Dimensions(
            node_positions={"A": (1.0, 2.0)},  # B has no position
            driver_angles={"B": DriverAngle(angular_velocity=0.2)},
            edge_distances={"AB": 1.0},
        )
        mech = graph_to_mechanism(graph, dims)
        joint_map = {j.id: j for j in mech.joints}
        # B should default to (0,0)
        assert joint_map["B"].position == pytest.approx((0.0, 0.0))


class TestMechanismToGraph:
    """Tests for mechanism_to_graph()."""

    def _make_fourbar_mechanism(self) -> Mechanism:
        """Build a four-bar Mechanism directly."""
        O1 = GroundJoint("O1", position=(0.0, 0.0), name="O1")
        O2 = GroundJoint("O2", position=(3.0, 0.0), name="O2")
        A = RevoluteJoint("A", position=(0.0, 1.0), name="A")
        B = RevoluteJoint("B", position=(3.0, 2.0), name="B")

        ground = GroundLink("ground", joints=[O1, O2], name="ground")
        driver = DriverLink(
            "crank",
            joints=[O1, A],
            name="crank",
            motor_joint=O1,
            angular_velocity=0.1,
            initial_angle=math.pi / 2,
        )
        link1 = Link("coupler", joints=[A, B], name="coupler")
        link2 = Link("rocker", joints=[B, O2], name="rocker")

        return Mechanism(
            name="Four-bar",
            joints=[O1, O2, A, B],
            links=[ground, driver, link1, link2],
            ground=ground,
        )

    def test_roundtrip_node_count(self):
        """mechanism_to_graph preserves the number of joints as nodes."""
        mech = self._make_fourbar_mechanism()
        graph, dims = mechanism_to_graph(mech)

        assert len(graph.nodes) == 4

    def test_roundtrip_edge_count(self):
        """mechanism_to_graph creates edges for each link pair."""
        mech = self._make_fourbar_mechanism()
        graph, dims = mechanism_to_graph(mech)

        # ground link: O1-O2 edge, driver: O1-A, coupler: A-B, rocker: B-O2
        assert len(graph.edges) == 4

    def test_roundtrip_roles(self):
        """Roles are correctly assigned: GROUND, DRIVER, DRIVEN."""
        mech = self._make_fourbar_mechanism()
        graph, dims = mechanism_to_graph(mech)

        roles = {nid: n.role for nid, n in graph.nodes.items()}
        assert roles["O1"] == NodeRole.GROUND
        assert roles["O2"] == NodeRole.GROUND
        assert roles["A"] == NodeRole.DRIVER
        assert roles["B"] == NodeRole.DRIVEN

    def test_roundtrip_positions(self):
        """Positions are stored in the returned Dimensions."""
        mech = self._make_fourbar_mechanism()
        graph, dims = mechanism_to_graph(mech)

        assert dims.node_positions["O1"] == pytest.approx((0.0, 0.0))
        assert dims.node_positions["O2"] == pytest.approx((3.0, 0.0))
        assert dims.node_positions["A"] == pytest.approx((0.0, 1.0))

    def test_roundtrip_driver_angle(self):
        """Driver angles are preserved in Dimensions."""
        mech = self._make_fourbar_mechanism()
        graph, dims = mechanism_to_graph(mech)

        assert "A" in dims.driver_angles
        assert dims.driver_angles["A"].angular_velocity == pytest.approx(0.1)
        assert dims.driver_angles["A"].initial_angle == pytest.approx(math.pi / 2)

    def test_roundtrip_name(self):
        """Mechanism name is preserved in graph."""
        mech = self._make_fourbar_mechanism()
        graph, dims = mechanism_to_graph(mech)

        assert graph.name == "Four-bar"
        assert dims.name == "Four-bar"

    def test_no_duplicate_edges(self):
        """Edges between the same pair of nodes are not duplicated."""
        mech = self._make_fourbar_mechanism()
        graph, dims = mechanism_to_graph(mech)

        edge_pairs = set()
        for e in graph.edges.values():
            pair = frozenset([e.source, e.target])
            assert pair not in edge_pairs, f"Duplicate edge: {e.source}-{e.target}"
            edge_pairs.add(pair)


class TestRoundTrip:
    """Test graph -> mechanism -> graph round-trip."""

    def test_fourbar_roundtrip_preserves_structure(self):
        """Converting graph->mechanism->graph preserves node count and roles."""
        graph, dims = _make_fourbar_graph()
        mech = graph_to_mechanism(graph, dims)
        graph2, dims2 = mechanism_to_graph(mech)

        # Node count should be preserved
        assert len(graph2.nodes) == len(graph.nodes)

        # Roles should be preserved
        original_roles = {nid: n.role for nid, n in graph.nodes.items()}
        restored_roles = {nid: n.role for nid, n in graph2.nodes.items()}
        for nid in original_roles:
            assert nid in restored_roles
            assert restored_roles[nid] == original_roles[nid]

    def test_fourbar_roundtrip_preserves_positions(self):
        """Positions survive the round-trip."""
        graph, dims = _make_fourbar_graph()
        mech = graph_to_mechanism(graph, dims)
        graph2, dims2 = mechanism_to_graph(mech)

        for nid in dims.node_positions:
            assert nid in dims2.node_positions
            assert dims2.node_positions[nid] == pytest.approx(dims.node_positions[nid])
