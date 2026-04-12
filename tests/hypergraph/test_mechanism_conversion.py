"""Tests for hypergraph/mechanism_conversion.py (HypergraphLinkage <-> Mechanism)."""

import math

import pytest

from pylinkage._types import NodeRole
from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph.core import Edge, Node
from pylinkage.hypergraph.graph import HypergraphLinkage
from pylinkage.hypergraph.mechanism_conversion import from_mechanism, to_mechanism
from pylinkage.mechanism import (
    DriverLink,
    GroundJoint,
    GroundLink,
    Link,
    Mechanism,
    RevoluteJoint,
)


def _make_fourbar_hypergraph() -> tuple[HypergraphLinkage, Dimensions]:
    """Build a simple four-bar as HypergraphLinkage + Dimensions."""
    hg = HypergraphLinkage(name="Four-bar")
    hg.add_node(Node(id="G1", role=NodeRole.GROUND, name="G1"))
    hg.add_node(Node(id="G2", role=NodeRole.GROUND, name="G2"))
    hg.add_node(Node(id="B", role=NodeRole.DRIVER, name="B"))
    hg.add_node(Node(id="C", role=NodeRole.DRIVEN, name="C"))

    hg.add_edge(Edge(id="e1", source="G1", target="B"))
    hg.add_edge(Edge(id="e2", source="B", target="C"))
    hg.add_edge(Edge(id="e3", source="G2", target="C"))

    dims = Dimensions(
        node_positions={
            "G1": (0.0, 0.0),
            "G2": (3.0, 0.0),
            "B": (0.0, 1.0),
            "C": (3.0, 2.0),
        },
        driver_angles={"B": DriverAngle(angular_velocity=0.1, initial_angle=math.pi / 2)},
        edge_distances={"e1": 1.0, "e2": 3.0, "e3": 1.0},
        name="Four-bar",
    )
    return hg, dims


class TestToMechanism:
    """Tests for to_mechanism()."""

    def test_fourbar_returns_mechanism(self):
        """Conversion produces a Mechanism instance."""
        hg, dims = _make_fourbar_hypergraph()
        mech = to_mechanism(hg, dims)
        assert isinstance(mech, Mechanism)
        assert mech.name == "Four-bar"

    def test_fourbar_joint_ids(self):
        """All node IDs appear as joint IDs in the Mechanism."""
        hg, dims = _make_fourbar_hypergraph()
        mech = to_mechanism(hg, dims)
        joint_ids = {j.id for j in mech.joints}
        assert {"G1", "G2", "B", "C"}.issubset(joint_ids)

    def test_fourbar_ground_joints(self):
        """Ground nodes become GroundJoint instances."""
        hg, dims = _make_fourbar_hypergraph()
        mech = to_mechanism(hg, dims)
        ground_ids = {j.id for j in mech.joints if isinstance(j, GroundJoint)}
        assert "G1" in ground_ids
        assert "G2" in ground_ids

    def test_fourbar_ground_link(self):
        """A GroundLink is created containing ground joints."""
        hg, dims = _make_fourbar_hypergraph()
        mech = to_mechanism(hg, dims)
        ground_links = [lk for lk in mech.links if isinstance(lk, GroundLink)]
        assert len(ground_links) == 1
        assert len(ground_links[0].joints) == 2

    def test_fourbar_driver_link(self):
        """A DriverLink is created for the driver node."""
        hg, dims = _make_fourbar_hypergraph()
        mech = to_mechanism(hg, dims)
        driver_links = [lk for lk in mech.links if isinstance(lk, DriverLink)]
        assert len(driver_links) == 1
        assert driver_links[0].angular_velocity == pytest.approx(0.1)

    def test_fourbar_positions(self):
        """Joint positions come from Dimensions."""
        hg, dims = _make_fourbar_hypergraph()
        mech = to_mechanism(hg, dims)
        jmap = {j.id: j for j in mech.joints}
        assert jmap["G1"].position == pytest.approx((0.0, 0.0))
        assert jmap["G2"].position == pytest.approx((3.0, 0.0))
        assert jmap["B"].position == pytest.approx((0.0, 1.0))
        assert jmap["C"].position == pytest.approx((3.0, 2.0))

    def test_fourbar_driven_links(self):
        """The driven node C gets two Link objects connecting to its parents."""
        hg, dims = _make_fourbar_hypergraph()
        mech = to_mechanism(hg, dims)
        plain_links = [lk for lk in mech.links if not isinstance(lk, (GroundLink, DriverLink))]
        # C connected to B and G2 => 2 plain links
        assert len(plain_links) == 2

    def test_single_crank(self):
        """A single driver with ground produces a driver link."""
        hg = HypergraphLinkage(name="Crank")
        hg.add_node(Node(id="G", role=NodeRole.GROUND))
        hg.add_node(Node(id="A", role=NodeRole.DRIVER))
        hg.add_edge(Edge(id="e1", source="G", target="A"))

        dims = Dimensions(
            node_positions={"G": (0.0, 0.0), "A": (1.0, 0.0)},
            driver_angles={"A": DriverAngle(angular_velocity=0.2, initial_angle=0.0)},
            edge_distances={"e1": 1.0},
        )
        mech = to_mechanism(hg, dims)
        assert len([j for j in mech.joints if isinstance(j, GroundJoint)]) == 1
        assert len([lk for lk in mech.links if isinstance(lk, DriverLink)]) == 1

    def test_unsolvable_raises(self):
        """A disconnected driven node raises ValueError."""
        hg = HypergraphLinkage(name="Bad")
        hg.add_node(Node(id="G", role=NodeRole.GROUND))
        hg.add_node(Node(id="X", role=NodeRole.DRIVEN))
        # No edges connecting X to anything solved

        dims = Dimensions(
            node_positions={"G": (0.0, 0.0), "X": (1.0, 1.0)},
        )
        with pytest.raises(ValueError, match="Could not determine solve order"):
            to_mechanism(hg, dims)

    def test_driver_without_ground_neighbor(self):
        """Driver not connected to ground still creates a joint (no driver link)."""
        hg = HypergraphLinkage(name="Orphan driver")
        hg.add_node(Node(id="G", role=NodeRole.GROUND))
        hg.add_node(Node(id="D", role=NodeRole.DRIVER))
        # D is not connected to G
        dims = Dimensions(
            node_positions={"G": (0.0, 0.0), "D": (2.0, 0.0)},
            driver_angles={"D": DriverAngle(angular_velocity=0.1)},
        )
        mech = to_mechanism(hg, dims)
        # No DriverLink should be created since there's no ground neighbor
        driver_links = [lk for lk in mech.links if isinstance(lk, DriverLink)]
        assert len(driver_links) == 0

    def test_missing_driver_angle_defaults_velocity(self):
        """When Dimensions has no driver_angle entry, angular_velocity defaults to tau/360."""
        hg = HypergraphLinkage(name="default-vel")
        hg.add_node(Node(id="G", role=NodeRole.GROUND))
        hg.add_node(Node(id="A", role=NodeRole.DRIVER))
        hg.add_edge(Edge(id="e1", source="G", target="A"))

        dims = Dimensions(
            node_positions={"G": (0.0, 0.0), "A": (1.0, 0.0)},
            # No driver_angles entry
        )
        mech = to_mechanism(hg, dims)
        dl = [lk for lk in mech.links if isinstance(lk, DriverLink)][0]
        assert dl.angular_velocity == pytest.approx(math.tau / 360)


class TestFromMechanism:
    """Tests for from_mechanism()."""

    def _make_fourbar_mechanism(self) -> Mechanism:
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

    def test_node_count(self):
        """from_mechanism produces a node for each joint."""
        mech = self._make_fourbar_mechanism()
        hg, dims = from_mechanism(mech)
        assert len(hg.nodes) == 4

    def test_edge_count(self):
        """from_mechanism creates edges from link joint pairs."""
        mech = self._make_fourbar_mechanism()
        hg, dims = from_mechanism(mech)
        # ground: O1-O2, driver: O1-A, coupler: A-B, rocker: B-O2
        assert len(hg.edges) == 4

    def test_roles(self):
        """Roles are correctly inferred from the mechanism."""
        mech = self._make_fourbar_mechanism()
        hg, dims = from_mechanism(mech)
        roles = {nid: n.role for nid, n in hg.nodes.items()}
        assert roles["O1"] == NodeRole.GROUND
        assert roles["O2"] == NodeRole.GROUND
        assert roles["A"] == NodeRole.DRIVER
        assert roles["B"] == NodeRole.DRIVEN

    def test_positions_preserved(self):
        """Node positions are stored in Dimensions."""
        mech = self._make_fourbar_mechanism()
        hg, dims = from_mechanism(mech)
        assert dims.node_positions["O1"] == pytest.approx((0.0, 0.0))
        assert dims.node_positions["A"] == pytest.approx((0.0, 1.0))

    def test_driver_angles_preserved(self):
        """Driver angle information is stored in Dimensions."""
        mech = self._make_fourbar_mechanism()
        hg, dims = from_mechanism(mech)
        assert "A" in dims.driver_angles
        assert dims.driver_angles["A"].angular_velocity == pytest.approx(0.1)
        assert dims.driver_angles["A"].initial_angle == pytest.approx(math.pi / 2)

    def test_name_preserved(self):
        """Mechanism name is preserved."""
        mech = self._make_fourbar_mechanism()
        hg, dims = from_mechanism(mech)
        assert hg.name == "Four-bar"
        assert dims.name == "Four-bar"

    def test_no_duplicate_edges(self):
        """No duplicate edges are created."""
        mech = self._make_fourbar_mechanism()
        hg, dims = from_mechanism(mech)
        edge_pairs = set()
        for e in hg.edges.values():
            pair = frozenset([e.source, e.target])
            assert pair not in edge_pairs
            edge_pairs.add(pair)


class TestRoundTrip:
    """Test hypergraph -> mechanism -> hypergraph round-trip."""

    def test_roundtrip_preserves_node_count(self):
        """Node count is preserved through round-trip."""
        hg, dims = _make_fourbar_hypergraph()
        mech = to_mechanism(hg, dims)
        hg2, dims2 = from_mechanism(mech)
        assert len(hg2.nodes) == len(hg.nodes)

    def test_roundtrip_preserves_positions(self):
        """Positions survive the round-trip."""
        hg, dims = _make_fourbar_hypergraph()
        mech = to_mechanism(hg, dims)
        hg2, dims2 = from_mechanism(mech)
        for nid in dims.node_positions:
            assert nid in dims2.node_positions
            assert dims2.node_positions[nid] == pytest.approx(dims.node_positions[nid])

    def test_roundtrip_preserves_roles(self):
        """Node roles survive the round-trip."""
        hg, dims = _make_fourbar_hypergraph()
        mech = to_mechanism(hg, dims)
        hg2, dims2 = from_mechanism(mech)
        for nid, node in hg.nodes.items():
            assert hg2.nodes[nid].role == node.role
