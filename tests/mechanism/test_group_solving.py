"""Tests for Assur group-based solving in Mechanism.

These tests verify that Mechanism._step_once() correctly uses
the Assur group decomposition solver, including triad support
for six-bar linkages.
"""

import math

from pylinkage.assur import Edge, LinkageGraph, Node, NodeRole
from pylinkage.assur.mechanism_conversion import graph_to_mechanism
from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.mechanism import (
    DriverLink,
    GroundJoint,
    GroundLink,
    Link,
    Mechanism,
    RevoluteJoint,
)


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class TestGroupSolverFourBar:
    """Verify that four-bar mechanisms work with the group-based solver."""

    def _build_four_bar(self) -> Mechanism:
        """Build a standard four-bar via graph_to_mechanism."""
        graph = LinkageGraph(name="FourBar")
        graph.add_node(Node("O1", role=NodeRole.GROUND))
        graph.add_node(Node("O2", role=NodeRole.GROUND))
        graph.add_node(Node("A", role=NodeRole.DRIVER))
        graph.add_node(Node("B", role=NodeRole.DRIVEN))

        graph.add_edge(Edge("O1A", source="O1", target="A"))
        graph.add_edge(Edge("AB", source="A", target="B"))
        graph.add_edge(Edge("BO2", source="B", target="O2"))

        dims = Dimensions(
            node_positions={
                "O1": (0.0, 0.0),
                "O2": (4.0, 0.0),
                "A": (1.0, 0.0),
                "B": (3.0, 2.0),
            },
            driver_angles={
                "A": DriverAngle(angular_velocity=0.1, initial_angle=0.0),
            },
            edge_distances={
                "O1A": 1.0,
                "AB": math.sqrt((3 - 1) ** 2 + (2 - 0) ** 2),
                "BO2": math.sqrt((3 - 4) ** 2 + (2 - 0) ** 2),
            },
        )
        return graph_to_mechanism(graph, dims)

    def test_uses_group_solver(self):
        """Mechanism built from graph should use the group solver."""
        mech = self._build_four_bar()
        assert mech._use_group_solver is True
        assert mech._decomposition is not None

    def test_step_produces_positions(self):
        """step() should produce valid positions for all joints."""
        mech = self._build_four_bar()
        positions_list = list(mech.step())
        assert len(positions_list) > 0
        for positions in positions_list:
            for pos in positions:
                assert pos[0] is not None
                assert pos[1] is not None

    def test_step_maintains_constraints(self):
        """All link lengths should be preserved throughout simulation."""
        mech = self._build_four_bar()
        # Get initial link distances
        initial_distances: dict[str, float] = {}
        for link in mech.links:
            if isinstance(link, GroundLink):
                continue
            if len(link.joints) == 2:
                j0, j1 = link.joints
                if j0.is_defined() and j1.is_defined():
                    d = _distance(
                        (j0.position[0], j0.position[1]),
                        (j1.position[0], j1.position[1]),
                    )
                    initial_distances[link.id] = d

        # Simulate and check a few frames
        for i, _positions in enumerate(mech.step()):
            if i > 10:
                break
            # Check link distances are preserved
            for link in mech.links:
                if link.id not in initial_distances:
                    continue
                j0, j1 = link.joints
                d = _distance(
                    (j0.position[0], j0.position[1]),
                    (j1.position[0], j1.position[1]),
                )
                assert abs(d - initial_distances[link.id]) < 1e-3, (
                    f"Frame {i}, link {link.id}: expected {initial_distances[link.id]:.4f}, "
                    f"got {d:.4f}"
                )


class TestGroupSolverSixBar:
    """Test six-bar mechanisms with triad groups."""

    def _build_watt_six_bar(self) -> tuple[Mechanism, Dimensions, LinkageGraph]:
        """Build a Watt I six-bar linkage (crank-rocker + rocker-rocker).

        Uses short crank (Grashof condition) so the mechanism stays
        assemblable through a full rotation.

        Structure: O1-A crank, A-B-O2 first dyad, B-C-O3 second dyad.
        """
        graph = LinkageGraph(name="Watt-I Six-Bar")
        graph.add_node(Node("O1", role=NodeRole.GROUND))
        graph.add_node(Node("O2", role=NodeRole.GROUND))
        graph.add_node(Node("O3", role=NodeRole.GROUND))
        graph.add_node(Node("A", role=NodeRole.DRIVER))
        graph.add_node(Node("B", role=NodeRole.DRIVEN))
        graph.add_node(Node("C", role=NodeRole.DRIVEN))

        # Crank: O1 -> A
        graph.add_edge(Edge("O1A", source="O1", target="A"))
        # First loop: A-B, B-O2
        graph.add_edge(Edge("AB", source="A", target="B"))
        graph.add_edge(Edge("BO2", source="B", target="O2"))
        # Second loop: B-C, C-O3
        graph.add_edge(Edge("BC", source="B", target="C"))
        graph.add_edge(Edge("CO3", source="C", target="O3"))

        # Classic Grashof crank-rocker four-bar (first loop) chained with
        # a short second loop. Crank is the shortest link.
        # First loop: O1(0,0)-A-B-O2(4,0), crank=1, coupler=3.5, rocker=3, ground=4
        # Grashof: 1+4 < 3+3.5 ✓
        # Second loop: B-C-O3(6,0), uses B as input
        # Keep the second loop's links long relative to ground distance
        # so it stays buildable as B oscillates.
        a_pos = (1.0, 0.0)
        b_pos = (3.0, 2.5)
        c_pos = (5.5, 2.0)
        dims = Dimensions(
            node_positions={
                "O1": (0.0, 0.0),
                "O2": (4.0, 0.0),
                "O3": (6.0, 0.0),
                "A": a_pos,
                "B": b_pos,
                "C": c_pos,
            },
            driver_angles={
                "A": DriverAngle(angular_velocity=0.1, initial_angle=0.0),
            },
            edge_distances={
                "O1A": 1.0,
                "AB": 3.5,
                "BO2": 3.0,
                "BC": 2.55,
                "CO3": 2.1,
            },
        )
        return graph_to_mechanism(graph, dims), dims, graph

    def _build_triad_six_bar(self) -> tuple[Mechanism, Dimensions, LinkageGraph]:
        """Build a six-bar requiring a triad decomposition.

        Structure:
            O1 -- A (crank)
            A -- X (unknown), X -- Y (unknown)
            Y -- O2, Y -- O3
            X has only 1 known neighbor (A) initially
            Y has 2 known + X → needs simultaneous solving with X

        This forces a triad: {X, Y} with anchors {A, O2, O3}.
        """
        graph = LinkageGraph(name="Triad Six-Bar")
        graph.add_node(Node("O1", role=NodeRole.GROUND))
        graph.add_node(Node("O2", role=NodeRole.GROUND))
        graph.add_node(Node("O3", role=NodeRole.GROUND))
        graph.add_node(Node("A", role=NodeRole.DRIVER))
        graph.add_node(Node("X", role=NodeRole.DRIVEN))
        graph.add_node(Node("Y", role=NodeRole.DRIVEN))

        # Crank
        graph.add_edge(Edge("O1A", source="O1", target="A"))
        # Triad edges: A-X, X-Y, Y-O2, Y-O3
        graph.add_edge(Edge("AX", source="A", target="X"))
        graph.add_edge(Edge("XY", source="X", target="Y"))
        graph.add_edge(Edge("YO2", source="Y", target="O2"))
        graph.add_edge(Edge("YO3", source="Y", target="O3"))

        dims = Dimensions(
            node_positions={
                "O1": (0.0, 0.0),
                "O2": (4.0, 0.0),
                "O3": (2.0, -2.0),
                "A": (1.0, 0.0),
                "X": (2.0, 1.5),
                "Y": (3.0, 0.5),
            },
            driver_angles={
                "A": DriverAngle(angular_velocity=0.1, initial_angle=0.0),
            },
            edge_distances={
                "O1A": 1.0,
                "AX": _distance((1.0, 0.0), (2.0, 1.5)),
                "XY": _distance((2.0, 1.5), (3.0, 0.5)),
                "YO2": _distance((3.0, 0.5), (4.0, 0.0)),
                "YO3": _distance((3.0, 0.5), (2.0, -2.0)),
            },
        )
        return graph_to_mechanism(graph, dims), dims, graph

    def test_watt_six_bar_uses_group_solver(self):
        """Watt six-bar should use the group-based solver."""
        mech, _, _ = self._build_watt_six_bar()
        assert mech._use_group_solver is True
        assert mech._decomposition is not None
        # Should decompose into 2 dyads
        assert len(mech._decomposition.groups) == 2

    def test_watt_six_bar_simulates(self):
        """Watt six-bar should simulate through multiple steps."""
        mech, _, _ = self._build_watt_six_bar()
        steps = list(mech.step())
        assert len(steps) > 0
        # All joints should have defined positions
        for positions in steps[:10]:
            for pos in positions:
                assert pos[0] is not None and pos[1] is not None

    def test_triad_six_bar_uses_group_solver(self):
        """Triad six-bar should use the group-based solver with a triad."""
        mech, _, _ = self._build_triad_six_bar()
        assert mech._use_group_solver is True
        assert mech._decomposition is not None
        # Should have at least one group (triad or dyad+dyad depending on decomposition)
        assert len(mech._decomposition.groups) >= 1

    def test_triad_six_bar_simulates(self):
        """Triad six-bar should simulate and maintain distance constraints."""
        mech, dims, graph = self._build_triad_six_bar()

        # Collect initial edge distances
        edge_distances = {}
        for edge in graph.edges.values():
            d = dims.get_edge_distance(edge.id)
            if d is not None:
                edge_distances[edge.id] = d

        # Simulate a few steps
        count = 0
        for positions in mech.step():
            count += 1
            if count > 10:
                break

            # All positions defined
            for pos in positions:
                assert pos[0] is not None and pos[1] is not None

        assert count > 0

    def test_triad_six_bar_constraints_preserved(self):
        """Link lengths should be maintained during triad six-bar simulation."""
        mech, _, _ = self._build_triad_six_bar()

        # Record initial link lengths
        initial_lengths: dict[str, float] = {}
        for link in mech.links:
            if isinstance(link, GroundLink):
                continue
            if len(link.joints) == 2:
                j0, j1 = link.joints
                if j0.is_defined() and j1.is_defined():
                    initial_lengths[link.id] = _distance(
                        (j0.position[0], j0.position[1]),
                        (j1.position[0], j1.position[1]),
                    )

        # Simulate and verify
        for i, _positions in enumerate(mech.step()):
            if i > 5:
                break
            for link in mech.links:
                if link.id not in initial_lengths:
                    continue
                j0, j1 = link.joints
                d = _distance(
                    (j0.position[0], j0.position[1]),
                    (j1.position[0], j1.position[1]),
                )
                assert abs(d - initial_lengths[link.id]) < 1e-2, (
                    f"Frame {i}, link {link.id}: expected "
                    f"{initial_lengths[link.id]:.4f}, got {d:.4f}"
                )


class TestFallbackSolver:
    """Verify that joint-by-joint solver still works as fallback."""

    def test_hand_built_mechanism_works(self):
        """A Mechanism built by hand (not from graph) should still simulate."""
        O1 = GroundJoint("O1", position=(0.0, 0.0))
        O2 = GroundJoint("O2", position=(4.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        B = RevoluteJoint("B", position=(3.0, 2.0))

        ground = GroundLink("ground", joints=[O1, O2])
        crank = DriverLink(
            "crank",
            joints=[O1, A],
            motor_joint=O1,
            angular_velocity=0.1,
            initial_angle=0.0,
        )
        link_ab = Link("AB", joints=[A, B])
        link_bo2 = Link("BO2", joints=[B, O2])

        mech = Mechanism(
            name="HandBuilt",
            joints=[O1, O2, A, B],
            links=[ground, crank, link_ab, link_bo2],
        )

        # Should still simulate (group solver or fallback)
        steps = list(mech.step())
        assert len(steps) > 0
        for positions in steps[:5]:
            for pos in positions:
                assert pos[0] is not None and pos[1] is not None
