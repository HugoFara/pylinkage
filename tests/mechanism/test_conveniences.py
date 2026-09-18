"""Tests for set_completely / simulation() / indeterminacy() on the
modern Linkage and Mechanism containers, plus the cross-API name
aliases on Mechanism.
"""

from pylinkage.mechanism import fourbar


def _modern_fourbar():
    from pylinkage.actuators import Crank
    from pylinkage.components import Ground
    from pylinkage.dyads import RRRDyad
    from pylinkage.simulation import Linkage

    A = Ground(0.0, 0.0, name="A")
    D = Ground(4.0, 0.0, name="D")
    crank = Crank(anchor=A, radius=1.0, angular_velocity=0.1, name="crank")
    pin = RRRDyad(
        anchor1=crank.output,
        anchor2=D,
        distance1=3.0,
        distance2=3.0,
        name="C",
    )
    return Linkage([A, D, crank, pin], name="modern-fourbar")


# ---------------------------------------------------------------------------
# Mechanism: cross-API aliases
# ---------------------------------------------------------------------------


class TestNumConstraintsRemoved:
    """``get_num_constraints`` / ``set_num_constraints`` were deprecated
    aliases and have been removed. Accessing them raises ``AttributeError``.
    """

    def test_mechanism_get_num_constraints_gone(self) -> None:
        import pytest

        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        with pytest.raises(AttributeError):
            m.get_num_constraints()

    def test_mechanism_set_num_constraints_gone(self) -> None:
        import pytest

        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        with pytest.raises(AttributeError):
            m.set_num_constraints([])

    def test_simulation_linkage_get_num_constraints_gone(self) -> None:
        import pytest

        linkage = _modern_fourbar()
        with pytest.raises(AttributeError):
            linkage.get_num_constraints()


class TestCoordAliases:
    def test_mechanism_get_coords_matches_get_joint_positions(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        assert m.get_coords() == m.get_joint_positions()


# ---------------------------------------------------------------------------
# set_completely
# ---------------------------------------------------------------------------


class TestSetCompletely:
    def test_mechanism_set_completely(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        constraints = m.get_constraints()
        positions = m.get_joint_positions()
        m.set_completely(constraints, positions)
        assert m.get_constraints() == constraints
        assert m.get_joint_positions() == positions

    def test_simulation_linkage_set_completely(self) -> None:
        linkage = _modern_fourbar()
        constraints = linkage.get_constraints()
        positions = linkage.get_coords()
        linkage.set_completely(constraints, [(p[0] or 0.0, p[1] or 0.0) for p in positions])
        assert linkage.get_constraints() == constraints


# ---------------------------------------------------------------------------
# simulation() context manager
# ---------------------------------------------------------------------------


class TestSimulationContext:
    def test_mechanism_iterates_with_step_index(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        with m.simulation(iterations=5) as sim:
            results = list(sim)
        assert len(results) == 5
        assert results[0][0] == 0  # first step index is 0
        assert results[-1][0] == 4

    def test_mechanism_restores_initial_positions(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        before = m.get_joint_positions()
        with m.simulation(iterations=10):
            pass  # run nothing inside; the with-block still triggers __exit__
        # No iteration happened so positions shouldn't have moved either.
        assert m.get_joint_positions() == before

    def test_mechanism_simulation_restores_after_partial_run(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        before = m.get_joint_positions()
        with m.simulation(iterations=5) as sim:
            for _ in sim:
                pass
        assert m.get_joint_positions() == before

    def test_simulation_linkage_iterates(self) -> None:
        linkage = _modern_fourbar()
        with linkage.simulation(iterations=4) as sim:
            results = list(sim)
        assert len(results) == 4
        assert results[0][0] == 0


# ---------------------------------------------------------------------------
# indeterminacy
# ---------------------------------------------------------------------------


class TestIndeterminacy:
    def test_mechanism_fourbar_dof_one(self) -> None:
        """A standard Grashof four-bar has 1 DOF."""
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        assert m.indeterminacy() == 1

    def test_simulation_linkage_fourbar_dof_one(self) -> None:
        linkage = _modern_fourbar()
        assert linkage.indeterminacy() == 1

    def test_slider_crank_dof_one(self) -> None:
        """The slider is a revolute pair on the rod plus a prismatic pair with the frame."""
        from pylinkage.mechanism import slider_crank

        assert slider_crank(crank=1.0, rod=3.0).indeterminacy() == 1

    def test_ground_point_and_coupler_point_are_no_pairs(self) -> None:
        """A joint on a single link joins nothing: a marker on the frame or a coupler point."""
        from pylinkage.mechanism import GroundJoint, RevoluteJoint

        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        assert m.ground is not None
        m.ground.joints.append(GroundJoint("mark", position=(2.0, -1.0)))
        coupler = m.get_link("coupler")
        assert coupler is not None
        coupler.joints.append(RevoluteJoint("P", position=(2.0, 3.0)))
        m.joints.extend([m.ground.joints[-1], coupler.joints[-1]])
        m.rebuild()
        assert m.indeterminacy() == 1
