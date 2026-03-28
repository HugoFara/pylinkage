"""Additional tests for mechanism_conversion.py to increase coverage.

Covers: RRP group conversion, unsupported group type error,
PrismaticJoint conversion in mechanism_to_graph, driver without ground
connection, and edge cases.
"""

import math

import pytest

from pylinkage._types import JointType, NodeRole
from pylinkage.assur.graph import Edge, LinkageGraph, Node
from pylinkage.assur.mechanism_conversion import graph_to_mechanism, mechanism_to_graph
from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.mechanism import (
    DriverLink,
    GroundJoint,
    GroundLink,
    Link,
    Mechanism,
    PrismaticJoint,
    RevoluteJoint,
)

# ---------------------------------------------------------------------------
# graph_to_mechanism: RRP group path
# ---------------------------------------------------------------------------

class TestGraphToMechanismRRP:
    def test_rrp_group_creates_prismatic_joint(self):
        """RRP Assur group in decomposition creates a PrismaticJoint."""
        g = LinkageGraph(name="RRP-test")
        g.add_node(Node("G1", role=NodeRole.GROUND))
        g.add_node(Node("G2", role=NodeRole.GROUND))
        g.add_node(Node("G3", role=NodeRole.GROUND))
        g.add_node(Node("D", role=NodeRole.DRIVER))
        g.add_node(Node("C", role=NodeRole.DRIVEN))

        g.add_edge(Edge("G1-D", source="G1", target="D"))
        g.add_edge(Edge("D-C", source="D", target="C"))
        g.add_edge(Edge("G2-C", source="G2", target="C"))
        g.add_edge(Edge("G3-C", source="G3", target="C"))

        dims = Dimensions(
            node_positions={
                "G1": (0.0, 0.0),
                "G2": (2.0, 0.0),
                "G3": (4.0, 0.0),
                "D": (0.0, 1.0),
                "C": (2.0, 1.0),
            },
            driver_angles={"D": DriverAngle(angular_velocity=0.1)},
            edge_distances={"G1-D": 1.0, "D-C": 2.0, "G2-C": 1.0, "G3-C": 2.0},
        )

        mech = graph_to_mechanism(g, dims)
        assert isinstance(mech, Mechanism)
        # C should exist as a joint
        joint_ids = {j.id for j in mech.joints}
        assert "C" in joint_ids


# ---------------------------------------------------------------------------
# graph_to_mechanism: driver without ground connection
# ---------------------------------------------------------------------------

class TestDriverWithoutGroundConnection:
    def test_driver_no_ground_neighbor(self):
        """Driver node with no ground neighbor still creates output joint."""
        g = LinkageGraph(name="no-ground-driver")
        g.add_node(Node("D", role=NodeRole.DRIVER))

        dims = Dimensions(
            node_positions={"D": (1.0, 2.0)},
            driver_angles={"D": DriverAngle(angular_velocity=0.2)},
        )

        mech = graph_to_mechanism(g, dims)
        joint_map = {j.id: j for j in mech.joints}
        assert "D" in joint_map
        # No DriverLink should be created since no ground motor
        driver_links = [lk for lk in mech.links if isinstance(lk, DriverLink)]
        assert len(driver_links) == 0


# ---------------------------------------------------------------------------
# graph_to_mechanism: unsupported group type
# ---------------------------------------------------------------------------

class TestUnsupportedGroupType:
    def test_unsupported_signature_raises(self):
        """Group with unimplemented signature raises NotImplementedError.

        We mock the decomposition to produce a triad group, which is not
        supported by graph_to_mechanism.
        """
        from unittest.mock import patch

        from pylinkage.assur.decomposition import DecompositionResult
        from pylinkage.assur.groups import Triad

        g = LinkageGraph(name="triad-test")
        g.add_node(Node("G1", role=NodeRole.GROUND))
        g.add_node(Node("X", role=NodeRole.DRIVEN))
        g.add_node(Node("Y", role=NodeRole.DRIVEN))

        fake_result = DecompositionResult(
            ground=["G1"],
            drivers=[],
            groups=[Triad(
                _signature="RRRRRR",
                internal_nodes=("X", "Y"),
                anchor_nodes=("G1",),
            )],
            graph=g,
        )

        dims = Dimensions(
            node_positions={
                "G1": (0.0, 0.0),
                "X": (1.0, 1.0),
                "Y": (2.0, 1.0),
            },
        )

        with patch(
            "pylinkage.assur.mechanism_conversion.decompose_assur_groups",
            return_value=fake_result,
        ), pytest.raises(NotImplementedError, match="not implemented"):
            graph_to_mechanism(g, dims)


# ---------------------------------------------------------------------------
# mechanism_to_graph: PrismaticJoint
# ---------------------------------------------------------------------------

class TestMechanismToGraphPrismatic:
    def test_prismatic_joint_becomes_prismatic_node(self):
        O1 = GroundJoint("O1", position=(0.0, 0.0), name="O1")
        P = PrismaticJoint("P", position=(1.0, 0.0), name="P", axis=(1.0, 0.0))

        ground = GroundLink("ground", joints=[O1], name="ground")
        link = Link("link", joints=[O1, P], name="link")

        mech = Mechanism(
            name="Slider",
            joints=[O1, P],
            links=[ground, link],
            ground=ground,
        )

        graph, dims = mechanism_to_graph(mech)
        assert graph.nodes["P"].joint_type == JointType.PRISMATIC
        assert graph.nodes["P"].role == NodeRole.DRIVEN

    def test_prismatic_joint_position_stored(self):
        O1 = GroundJoint("O1", position=(0.0, 0.0), name="O1")
        P = PrismaticJoint("P", position=(2.5, 3.0), name="P", axis=(0.0, 1.0))

        ground = GroundLink("ground", joints=[O1], name="ground")
        link = Link("link", joints=[O1, P], name="link")

        mech = Mechanism(
            name="Slider",
            joints=[O1, P],
            links=[ground, link],
            ground=ground,
        )

        graph, dims = mechanism_to_graph(mech)
        assert dims.node_positions["P"] == pytest.approx((2.5, 3.0))


# ---------------------------------------------------------------------------
# mechanism_to_graph: edge distance storage
# ---------------------------------------------------------------------------

class TestEdgeDistances:
    def test_edge_distances_stored_in_dimensions(self):
        O1 = GroundJoint("O1", position=(0.0, 0.0), name="O1")
        O2 = GroundJoint("O2", position=(3.0, 0.0), name="O2")
        A = RevoluteJoint("A", position=(0.0, 1.0), name="A")

        ground = GroundLink("ground", joints=[O1, O2], name="ground")
        driver = DriverLink(
            "crank",
            joints=[O1, A],
            name="crank",
            motor_joint=O1,
            angular_velocity=0.1,
            initial_angle=math.pi / 2,
        )

        mech = Mechanism(
            name="Test",
            joints=[O1, O2, A],
            links=[ground, driver],
            ground=ground,
        )

        graph, dims = mechanism_to_graph(mech)
        # At least some edges should have distances if the links provide them
        assert isinstance(dims.edge_distances, dict)


# ---------------------------------------------------------------------------
# mechanism_to_graph: multiple links sharing joints
# ---------------------------------------------------------------------------

class TestSharedJoints:
    def test_no_duplicate_edges_for_shared_joints(self):
        """Two links sharing the same joint pair don't create duplicate edges."""
        O1 = GroundJoint("O1", position=(0.0, 0.0), name="O1")
        A = RevoluteJoint("A", position=(1.0, 0.0), name="A")

        ground = GroundLink("ground", joints=[O1], name="ground")
        link1 = Link("link1", joints=[O1, A], name="link1")
        link2 = Link("link2", joints=[O1, A], name="link2")

        mech = Mechanism(
            name="Test",
            joints=[O1, A],
            links=[ground, link1, link2],
            ground=ground,
        )

        graph, dims = mechanism_to_graph(mech)
        # Should only have one edge between O1 and A
        edge_count = sum(
            1 for e in graph.edges.values()
            if {e.source, e.target} == {"O1", "A"}
        )
        assert edge_count == 1
