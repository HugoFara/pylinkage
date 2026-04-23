"""Tests for pylinkage.optimization.multi_objective_optimization."""

from __future__ import annotations

import pytest

pytest.importorskip("pymoo")

import numpy as np  # noqa: E402

from pylinkage.mechanism import fourbar  # noqa: E402
from pylinkage.optimization.multi_objective import (  # noqa: E402
    LinkageProblem,
    multi_objective_optimization,
)


def _make_linkage():
    return fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)


def _single_objective(linkage, constraints, joint_pos) -> float:
    return float(sum(c * c for c in constraints))


def _pair_objective(linkage, constraints, joint_pos) -> float:
    return float(sum((c - 1.0) ** 2 for c in constraints))


class TestSingleObjectiveShape:
    """Regression: pymoo returns F with shape (n_pop,) for single-objective runs.

    The optimizer previously indexed ``result.F[:, 0]`` unconditionally, which
    crashed. It now reshapes to 2-D against ``n_obj``.
    """

    def test_single_objective_runs(self) -> None:
        linkage = _make_linkage()
        ensemble = multi_objective_optimization(
            objectives=[_single_objective],
            linkage=linkage,
            objective_names=["score"],
            n_generations=2,
            pop_size=8,
            seed=0,
            verbose=False,
        )
        assert "score" in ensemble.scores
        assert ensemble.scores["score"].ndim == 1
        assert ensemble.scores["score"].shape[0] == len(ensemble.dimensions)


def _linkage_factory():
    return _make_linkage()


class TestParallel:
    """``n_workers > 1`` routes through ``ProcessPoolExecutor``."""

    def test_n_workers_two_runs(self) -> None:
        linkage = _make_linkage()
        ensemble = multi_objective_optimization(
            objectives=[_single_objective, _pair_objective],
            linkage=linkage,
            objective_names=["a", "b"],
            n_generations=2,
            pop_size=8,
            seed=0,
            verbose=False,
            n_workers=2,
        )
        assert np.isfinite(ensemble.scores["a"]).all()
        assert np.isfinite(ensemble.scores["b"]).all()

    def test_linkage_factory_bypasses_pickling(self) -> None:
        """``linkage_factory`` lets workers rebuild the linkage locally."""
        linkage = _make_linkage()
        ensemble = multi_objective_optimization(
            objectives=[_single_objective, _pair_objective],
            linkage=linkage,
            objective_names=["a", "b"],
            n_generations=2,
            pop_size=8,
            seed=0,
            verbose=False,
            n_workers=2,
            linkage_factory=_linkage_factory,
        )
        assert len(ensemble.dimensions) > 0


class TestPoolReuse:
    """The shared process pool persists across batches and is cleaned
    up on ``close()``. Previously each generation forked a fresh pool
    of N workers — a measurable per-generation tax."""

    def _make_problem(self) -> LinkageProblem:
        linkage = _make_linkage()
        joint_pos = tuple(linkage.get_coords())
        constraints = tuple(
            c for c in linkage.get_constraints()
            if c is not None and isinstance(c, (int, float))
        )
        bounds = (
            [c * 0.5 for c in constraints],
            [c * 1.5 for c in constraints],
        )
        return LinkageProblem(
            linkage=linkage,
            objectives=[_single_objective, _pair_objective],
            bounds=bounds,
            joint_pos=joint_pos,
            n_workers=2,
        )

    def test_pool_lazy_create(self) -> None:
        """No pool is created until a parallel batch runs."""
        problem = self._make_problem()
        try:
            assert problem._pool is None
        finally:
            problem.close()

    def test_pool_persists_across_batches(self) -> None:
        """Same pool object survives multiple batch evaluations."""
        problem = self._make_problem()
        try:
            X = np.array([[1.0, 3.0, 3.0, 4.0], [1.2, 3.1, 2.9, 4.0]])
            problem._evaluate_batch(X)
            pool_after_first = problem._pool
            assert pool_after_first is not None
            problem._evaluate_batch(X)
            assert problem._pool is pool_after_first
        finally:
            problem.close()

    def test_close_shuts_down_pool(self) -> None:
        problem = self._make_problem()
        X = np.array([[1.0, 3.0, 3.0, 4.0]])
        problem._evaluate_batch(X)
        assert problem._pool is not None
        problem.close()
        assert problem._pool is None

    def test_close_idempotent(self) -> None:
        """Calling close() twice — or before any pool exists — is safe."""
        problem = self._make_problem()
        problem.close()  # before any pool
        problem.close()  # idempotent

    def test_optimization_driver_closes_pool(self) -> None:
        """``multi_objective_optimization`` shuts down the pool in
        ``finally`` so workers don't outlive the call."""
        linkage = _make_linkage()
        ensemble = multi_objective_optimization(
            objectives=[_single_objective, _pair_objective],
            linkage=linkage,
            objective_names=["a", "b"],
            n_generations=2,
            pop_size=8,
            seed=0,
            verbose=False,
            n_workers=2,
        )
        assert len(ensemble.dimensions) > 0


class TestTwoObjectives:
    def test_two_objectives_runs(self) -> None:
        linkage = _make_linkage()
        ensemble = multi_objective_optimization(
            objectives=[_single_objective, _pair_objective],
            linkage=linkage,
            objective_names=["a", "b"],
            n_generations=2,
            pop_size=8,
            seed=0,
            verbose=False,
        )
        assert set(ensemble.scores.keys()) == {"a", "b"}
        assert ensemble.scores["a"].shape == ensemble.scores["b"].shape
        assert np.isfinite(ensemble.scores["a"]).all()
