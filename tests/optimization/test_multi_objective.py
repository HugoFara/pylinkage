"""Tests for pylinkage.optimization.multi_objective_optimization."""

from __future__ import annotations

import pytest

pytest.importorskip("pymoo")

import numpy as np  # noqa: E402

from pylinkage.mechanism import fourbar  # noqa: E402
from pylinkage.optimization.multi_objective import (  # noqa: E402
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
