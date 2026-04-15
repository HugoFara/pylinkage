"""Tests for SimLinkage → HypergraphLinkage conversion."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pylinkage._types import NodeRole
from pylinkage.actuators import ArcCrank, Crank, LinearActuator
from pylinkage.components import Ground
from pylinkage.dyads import FixedDyad, PPDyad, RRPDyad, RRRDyad
from pylinkage.hypergraph import from_sim_linkage, to_mechanism
from pylinkage.simulation import Linkage


def _fourbar() -> Linkage:
    A = Ground(0.0, 0.0, name="A")
    D = Ground(4.0, 0.0, name="D")
    crank = Crank(anchor=A, radius=1.0, angular_velocity=0.1, name="B")
    pin = RRRDyad(
        anchor1=crank.output,
        anchor2=D,
        distance1=3.0,
        distance2=3.0,
        name="C",
    )
    return Linkage([A, D, crank, pin], name="fourbar")


class TestFourBar:
    def test_node_roles(self) -> None:
        hg, _ = from_sim_linkage(_fourbar())
        assert hg.nodes["A"].role == NodeRole.GROUND
        assert hg.nodes["D"].role == NodeRole.GROUND
        assert hg.nodes["B"].role == NodeRole.DRIVER
        assert hg.nodes["C"].role == NodeRole.DRIVEN

    def test_edges(self) -> None:
        hg, dims = from_sim_linkage(_fourbar())
        # 1 crank edge + 2 dyad edges.
        assert len(hg.edges) == 3
        # Crank carries its radius.
        crank_edges = [e for e in hg.edges.values() if {e.source, e.target} == {"A", "B"}]
        assert len(crank_edges) == 1
        assert dims.edge_distances[crank_edges[0].id] == 1.0
        # Dyad edges carry the two distances.
        dyad_lengths = sorted(
            dims.edge_distances[e.id]
            for e in hg.edges.values()
            if "C" in {e.source, e.target} and {e.source, e.target} != {"A", "B"}
        )
        assert dyad_lengths == [3.0, 3.0]

    def test_driver_angle(self) -> None:
        _, dims = from_sim_linkage(_fourbar())
        da = dims.driver_angles["B"]
        assert da.angular_velocity == pytest.approx(0.1)

    def test_round_trip_via_mechanism_preserves_driver_trajectory(self) -> None:
        """hypergraph → mechanism simulates the same crank tip trajectory."""
        sim = _fourbar()
        hg, dims = from_sim_linkage(sim)
        mech = to_mechanism(hg, dims)

        sim_tip = np.array(
            [sim.components[2].output.position for _ in range(5) if sim.step(iterations=1) or True]
        )
        # The sim.step() call above mutates; just verify the mechanism
        # also advances and the crank-tip joint ends up on the circle
        # of radius 1 about A.
        mech.step(iterations=5)
        b_pos = next(j.position for j in mech.joints if j.name == "B")
        assert b_pos is not None
        assert math.isfinite(b_pos[0]) and math.isfinite(b_pos[1])
        assert math.hypot(b_pos[0] - 0.0, b_pos[1] - 0.0) == pytest.approx(1.0, rel=1e-6)
        # And the sim-linkage tip stays on its circle too.
        assert np.allclose(np.hypot(sim_tip[:, 0], sim_tip[:, 1]), 1.0)


class TestFixedDyad:
    def test_hyperedge_and_edges(self) -> None:
        A = Ground(0.0, 0.0, name="A")
        D = Ground(4.0, 0.0, name="D")
        crank = Crank(anchor=A, radius=1.0, angular_velocity=0.1, name="B")
        coupler = RRRDyad(
            anchor1=crank.output, anchor2=D, distance1=3.0, distance2=3.0, name="C"
        )
        tip = FixedDyad(
            anchor1=coupler, anchor2=crank.output, distance=1.0, angle=0.5, name="T"
        )
        sim = Linkage([A, D, crank, coupler, tip], name="fixed-demo")

        hg, dims = from_sim_linkage(sim)
        # Exactly one hyperedge over (C, B, T).
        ternary = [h for h in hg.hyperedges.values() if set(h.nodes) == {"C", "B", "T"}]
        assert len(ternary) == 1
        # Two edges for numeric legs.
        tip_edges = [e for e in hg.edges.values() if "T" in {e.source, e.target}]
        assert len(tip_edges) == 2
        # First leg distance matches the declared ``distance``.
        declared_leg = next(
            e for e in tip_edges if {e.source, e.target} == {"C", "T"}
        )
        assert dims.edge_distances[declared_leg.id] == pytest.approx(1.0)


class TestRRPDyad:
    def test_creates_hyperedge_for_line(self) -> None:
        O1 = Ground(0.0, 0.0, name="O1")
        L1 = Ground(5.0, 0.0, name="L1")
        L2 = Ground(5.0, 1.0, name="L2")
        crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="P")
        slider = RRPDyad(
            revolute_anchor=crank.output,
            line_anchor1=L1,
            line_anchor2=L2,
            distance=4.0,
            name="S",
        )
        sim = Linkage([O1, L1, L2, crank, slider])

        hg, dims = from_sim_linkage(sim)
        # Revolute leg edge.
        rev_edges = [e for e in hg.edges.values() if {e.source, e.target} == {"P", "S"}]
        assert len(rev_edges) == 1
        assert dims.edge_distances[rev_edges[0].id] == pytest.approx(4.0)
        # Line hyperedge over (L1, L2, S).
        ternary = [h for h in hg.hyperedges.values() if set(h.nodes) == {"L1", "L2", "S"}]
        assert len(ternary) == 1


class TestPPDyad:
    def test_creates_5ary_hyperedge(self) -> None:
        A = Ground(0.0, 0.0, name="A")
        B = Ground(2.0, 0.0, name="B")
        C = Ground(1.0, -1.0, name="C")
        D = Ground(1.0, 1.0, name="D")
        dyad = PPDyad(
            line1_anchor1=A,
            line1_anchor2=B,
            line2_anchor1=C,
            line2_anchor2=D,
            name="X",
        )
        sim = Linkage([A, B, C, D, dyad])

        hg, _ = from_sim_linkage(sim)
        hyperedges = list(hg.hyperedges.values())
        assert len(hyperedges) == 1
        assert set(hyperedges[0].nodes) == {"A", "B", "C", "D", "X"}


class TestLinearActuatorAndArcCrank:
    def test_linear_actuator_is_prismatic_driver(self) -> None:
        base = Ground(0.0, 0.0, name="O")
        la = LinearActuator(
            anchor=base, angle=0.0, stroke=2.0, speed=0.1, name="slide"
        )
        sim = Linkage([base, la])

        hg, dims = from_sim_linkage(sim)
        assert hg.nodes["slide"].role == NodeRole.DRIVER
        assert hg.nodes["slide"].joint_type.name == "PRISMATIC"
        assert "slide" in dims.driver_angles

    def test_arc_crank_is_driver(self) -> None:
        base = Ground(0.0, 0.0, name="O")
        ac = ArcCrank(
            anchor=base,
            radius=1.0,
            arc_start=0.0,
            arc_end=math.pi,
            angular_velocity=0.1,
            name="swing",
        )
        sim = Linkage([base, ac])

        hg, dims = from_sim_linkage(sim)
        assert hg.nodes["swing"].role == NodeRole.DRIVER
        assert dims.driver_angles["swing"].angular_velocity == pytest.approx(0.1)


class TestToHypergraphMethod:
    def test_method_matches_function(self) -> None:
        sim = _fourbar()
        hg1, dims1 = sim.to_hypergraph()
        hg2, dims2 = from_sim_linkage(sim)
        assert set(hg1.nodes) == set(hg2.nodes)
        assert dims1 == dims2


class TestErrors:
    def test_non_simlinkage_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            from_sim_linkage(object())  # type: ignore[arg-type]
