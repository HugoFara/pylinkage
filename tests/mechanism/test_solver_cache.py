"""Cache-invalidation tests for set_constraints / set_num_constraints.

After changing constraints the cached numba ``SolverData`` must be
discarded so the next ``step_fast()`` rebuilds against the new
constraints. Otherwise stale link lengths would silently keep being
simulated.
"""

import numpy as np

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


class TestMechanismSetConstraintsInvalidatesCache:
    def test_set_constraints_drops_solver_data(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        m.compile()
        assert m._solver_data is not None
        m.set_constraints(m.get_constraints())
        assert m._solver_data is None

    def test_set_num_constraints_alias_drops_solver_data(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        m.compile()
        assert m._solver_data is not None
        m.set_num_constraints(m.get_num_constraints())
        assert m._solver_data is None

    def test_step_fast_recompiles_after_invalidation(self) -> None:
        """After set_constraints, the next step_fast() rebuilds SolverData."""
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        m.step_fast(iterations=5)
        assert m._solver_data is not None
        first_data = m._solver_data
        m.set_constraints(m.get_constraints())
        assert m._solver_data is None
        m.step_fast(iterations=5)
        # Freshly compiled — a new SolverData instance, not the old one.
        assert m._solver_data is not None
        assert m._solver_data is not first_data


class TestSimulationLinkageSetNumConstraintsInvalidatesCache:
    def test_drops_solver_data(self) -> None:
        linkage = _modern_fourbar()
        linkage.compile()
        assert linkage._solver_data is not None
        linkage.set_num_constraints(linkage.get_num_constraints())
        assert linkage._solver_data is None

    def test_step_fast_picks_up_new_radius(self) -> None:
        linkage = _modern_fourbar()
        traj_a = linkage.step_fast(iterations=10)

        # Replace the crank radius via the constraint vector.
        constraints = list(linkage.get_num_constraints())
        # constraints layout: crank radius first, then dyad d1/d2.
        constraints[0] = 1.5
        linkage.set_num_constraints(constraints)
        traj_b = linkage.step_fast(iterations=10)
        assert not np.allclose(traj_a, traj_b)
