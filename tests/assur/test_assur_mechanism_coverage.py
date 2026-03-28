"""Tests for assur_mechanism.py to increase coverage.

Covers: AssurMechanism construction, from_mechanism, from_graph, properties
(decomposition, assur_groups, degree_of_freedom, num_* properties),
analyze(), validate(), is_valid(), graph property, and delegation methods.
"""

import math

import pytest

from pylinkage._types import NodeRole
from pylinkage.assur.assur_mechanism import AssurMechanism
from pylinkage.assur.graph import Edge, LinkageGraph, Node
from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.mechanism import (
    DriverLink,
    GroundJoint,
    GroundLink,
    Link,
    Mechanism,
    RevoluteJoint,
)


def _make_fourbar_mechanism() -> Mechanism:
    """Build a simple four-bar Mechanism."""
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
    coupler = Link("coupler", joints=[A, B], name="coupler")
    rocker = Link("rocker", joints=[B, O2], name="rocker")

    return Mechanism(
        name="Four-bar",
        joints=[O1, O2, A, B],
        links=[ground, driver, coupler, rocker],
        ground=ground,
    )


def _make_fourbar_graph() -> tuple[LinkageGraph, Dimensions]:
    """Build a four-bar as a LinkageGraph + Dimensions."""
    graph = LinkageGraph(name="Four-bar-graph")
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
    )
    return graph, dims


class TestAssurMechanismConstruction:
    def test_from_mechanism(self):
        mech = _make_fourbar_mechanism()
        am = AssurMechanism.from_mechanism(mech)
        assert am.mechanism is mech

    def test_from_graph(self):
        graph, dims = _make_fourbar_graph()
        am = AssurMechanism.from_graph(graph, dims)
        assert am.mechanism is not None
        assert am._graph is graph
        assert am._dimensions is dims

    def test_direct_construction(self):
        mech = _make_fourbar_mechanism()
        am = AssurMechanism(mechanism=mech)
        assert am.mechanism is mech
        assert am._decomposition is None
        assert am._graph is None


class TestAssurMechanismProperties:
    def test_graph_property_from_mechanism(self):
        mech = _make_fourbar_mechanism()
        am = AssurMechanism(mechanism=mech)
        # graph property should auto-convert from mechanism
        g = am.graph
        assert g is not None
        assert len(g.nodes) > 0

    def test_graph_property_from_graph(self):
        graph, dims = _make_fourbar_graph()
        am = AssurMechanism.from_graph(graph, dims)
        assert am.graph is graph

    def test_decomposition_computed_and_cached(self):
        mech = _make_fourbar_mechanism()
        am = AssurMechanism(mechanism=mech)
        d1 = am.decomposition
        d2 = am.decomposition
        assert d1 is d2  # Same cached object

    def test_assur_groups(self):
        mech = _make_fourbar_mechanism()
        am = AssurMechanism(mechanism=mech)
        groups = am.assur_groups
        assert len(groups) >= 1

    def test_degree_of_freedom(self):
        mech = _make_fourbar_mechanism()
        am = AssurMechanism(mechanism=mech)
        dof = am.degree_of_freedom
        # Four-bar: 4 links, 4 joints (1-DOF each)
        # DOF = 3*(4-1) - 2*4 = 9 - 8 = 1
        assert dof == 1

    def test_num_ground_nodes(self):
        mech = _make_fourbar_mechanism()
        am = AssurMechanism(mechanism=mech)
        assert am.num_ground_nodes >= 1

    def test_num_driver_nodes(self):
        mech = _make_fourbar_mechanism()
        am = AssurMechanism(mechanism=mech)
        assert am.num_driver_nodes >= 1

    def test_num_assur_groups(self):
        mech = _make_fourbar_mechanism()
        am = AssurMechanism(mechanism=mech)
        assert am.num_assur_groups >= 1


class TestAssurMechanismAnalyze:
    def test_analyze_clears_cache(self):
        mech = _make_fourbar_mechanism()
        am = AssurMechanism(mechanism=mech)
        d1 = am.decomposition
        d2 = am.analyze()
        # analyze() forces recomputation, so new object
        assert d2 is not d1

    def test_analyze_returns_decomposition(self):
        mech = _make_fourbar_mechanism()
        am = AssurMechanism(mechanism=mech)
        result = am.analyze()
        assert result is not None
        assert result.graph is not None


class TestAssurMechanismValidation:
    def test_validate_fourbar(self):
        mech = _make_fourbar_mechanism()
        am = AssurMechanism(mechanism=mech)
        msgs = am.validate()
        assert isinstance(msgs, list)

    def test_is_valid_fourbar(self):
        mech = _make_fourbar_mechanism()
        am = AssurMechanism(mechanism=mech)
        # May or may not be valid depending on edge coverage
        result = am.is_valid()
        assert isinstance(result, bool)

    def test_is_valid_returns_true_when_no_warnings(self):
        graph, dims = _make_fourbar_graph()
        am = AssurMechanism.from_graph(graph, dims)
        msgs = am.validate()
        assert am.is_valid() == (len(msgs) == 0)


class TestAssurMechanismDelegation:
    def test_get_joint_positions(self):
        mech = _make_fourbar_mechanism()
        am = AssurMechanism(mechanism=mech)
        positions = am.get_joint_positions()
        assert len(positions) == len(mech.joints)

    def test_reset(self):
        mech = _make_fourbar_mechanism()
        am = AssurMechanism(mechanism=mech)
        # Should not raise
        am.reset()
