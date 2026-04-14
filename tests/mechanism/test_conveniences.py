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


class TestDeprecatedNumConstraintsAliases:
    """``get_num_constraints``/``set_num_constraints`` are deprecated
    aliases kept for one release for migration; they must still return
    the correct value but emit ``DeprecationWarning``.
    """

    def test_mechanism_get_num_constraints_emits_deprecation(self) -> None:
        import pytest

        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        with pytest.deprecated_call():
            result = m.get_num_constraints()
        assert result == m.get_constraints()

    def test_mechanism_set_num_constraints_emits_deprecation(self) -> None:
        import pytest

        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        original = m.get_constraints()
        with pytest.deprecated_call():
            m.set_num_constraints(original)
        assert m.get_constraints() == original

    def test_simulation_linkage_get_num_constraints_emits_deprecation(self) -> None:
        import pytest

        linkage = _modern_fourbar()
        with pytest.deprecated_call():
            result = linkage.get_num_constraints()
        assert result == linkage.get_constraints()


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
