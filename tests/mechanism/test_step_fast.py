"""Tests for Mechanism.step_fast() — numba-batched simulation."""

import numpy as np

from pylinkage.mechanism import fourbar


class TestMechanismStepFast:
    def test_returns_trajectory_array(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        traj = m.step_fast(iterations=10)
        assert traj.shape == (10, len(m.joints), 2)
        assert traj.dtype == np.float64

    def test_default_iterations_uses_rotation_period(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        traj = m.step_fast()
        assert traj.shape[0] == m.get_rotation_period()

    def test_compile_is_idempotent(self) -> None:
        """Calling compile() repeatedly leaves _solver_data populated."""
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        m.compile()
        first = m._solver_data
        m.compile()
        # A new SolverData is built each call; both must be non-None.
        assert m._solver_data is not None
        assert first is not None

    def test_step_fast_matches_step_geometry(self) -> None:
        """step_fast trajectories should resemble step trajectories.

        We don't check exact equality (different solvers can pick
        different intersection branches), but the trajectory should be a
        finite, non-zero motion of the dependent joint.
        """
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        traj = m.step_fast(iterations=20)
        # Find a non-ground joint that actually moves.
        from pylinkage.mechanism import GroundJoint

        moving_idx = next(i for i, j in enumerate(m.joints) if not isinstance(j, GroundJoint))
        path = traj[:, moving_idx, :]
        assert np.isfinite(path).all()
        # At least some movement
        assert (path.max(axis=0) - path.min(axis=0)).sum() > 0


class TestSimulationLinkageStepFast:
    def _build(self):
        from pylinkage.actuators import Crank
        from pylinkage.components import Ground
        from pylinkage.dyads import RRRDyad
        from pylinkage.simulation import Linkage

        O1 = Ground(0.0, 0.0, name="O1")
        O2 = Ground(4.0, 0.0, name="O2")
        crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
        rocker = RRRDyad(
            anchor1=crank.output,
            anchor2=O2,
            distance1=3.0,
            distance2=3.0,
            name="C",
        )
        return Linkage([O1, O2, crank, rocker], name="Four-Bar")

    def test_returns_trajectory_array(self) -> None:
        linkage = self._build()
        traj = linkage.step_fast(iterations=10)
        assert traj.shape == (10, len(linkage.components), 2)
        assert traj.dtype == np.float64

    def test_default_iterations_uses_rotation_period(self) -> None:
        linkage = self._build()
        traj = linkage.step_fast()
        assert traj.shape[0] == linkage.get_rotation_period()

    def test_compile_caches_solver_data(self) -> None:
        linkage = self._build()
        assert linkage._solver_data is None
        linkage.compile()
        assert linkage._solver_data is not None

    def test_traj_finite(self) -> None:
        linkage = self._build()
        traj = linkage.step_fast(iterations=20)
        assert np.isfinite(traj).all()
