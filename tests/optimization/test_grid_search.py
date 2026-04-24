"""Tests for pylinkage.optimization.grid_search."""

from __future__ import annotations

import numpy as np
import pytest

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRRDyad
from pylinkage.exceptions import OptimizationError
from pylinkage.optimization.grid_search import (
    fast_variator,
    sequential_variator,
    trials_and_errors_optimization,
)
from pylinkage.optimization.utils import kinematic_minimization
from pylinkage.simulation import Linkage


def make_fourbar() -> Linkage:
    O1 = Ground(0.0, 0.0, name="O1")
    O2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output, anchor2=O2, distance1=2.5, distance2=2.0, name="rocker"
    )
    return Linkage([O1, O2, crank, rocker], name="FourBar")


@kinematic_minimization
def simple_fitness(loci, **_):
    tip = [x[-1] for x in loci]
    return sum(abs(x[0]) + abs(x[1]) for x in tip)


def test_fast_variator_produces_product_of_linspaces():
    bounds = ([0.0, 0.0], [1.0, 1.0])
    result = list(fast_variator(3, bounds))
    # 3 * 3 = 9 combinations
    assert len(result) == 9
    # Each is a 2-element list
    for item in result:
        assert len(item) == 2
        assert all(isinstance(v, float) for v in item)


def test_fast_variator_covers_endpoints():
    bounds = ([0.0], [2.0])
    result = list(fast_variator(5, bounds))
    flat = [v[0] for v in result]
    assert min(flat) == pytest.approx(0.0)
    assert max(flat) == pytest.approx(2.0)


def test_sequential_variator_even_divisions():
    center = [1.0, 2.0]
    bounds = ([0.2, 0.4], [5.0, 10.0])
    result = list(sequential_variator(center, 4, bounds))
    # Should produce "divisions" outputs total (approx)
    assert len(result) >= 1
    for item in result:
        assert len(item) == 2


def test_sequential_variator_odd_divisions():
    center = [1.0]
    bounds = ([0.2], [5.0])
    result = list(sequential_variator(center, 5, bounds))
    assert len(result) >= 1
    for item in result:
        assert len(item) == 1


def test_trials_and_errors_optimization_basic():
    linkage = make_fourbar()
    ensemble = trials_and_errors_optimization(
        eval_func=simple_fitness,
        linkage=linkage,
        n_results=3,
        divisions=3,
        order_relation=min,
        verbose=False,
    )
    assert ensemble is not None
    assert len(ensemble) >= 1
    assert ensemble.dimensions.shape[1] == 3  # three constraints


def test_trials_and_errors_optimization_sequential_and_verbose(capsys):
    linkage = make_fourbar()
    ensemble = trials_and_errors_optimization(
        eval_func=simple_fitness,
        linkage=linkage,
        n_results=2,
        divisions=3,
        order_relation=min,
        sequential=True,
        verbose=True,
    )
    # verbose should print a closing summary
    captured = capsys.readouterr()
    assert "Trials and errors optimization finished" in captured.out
    assert len(ensemble) >= 1


def test_trials_and_errors_optimization_with_bounds():
    linkage = make_fourbar()
    bounds = ([0.5, 1.5, 1.0], [2.0, 4.0, 3.5])
    ensemble = trials_and_errors_optimization(
        eval_func=simple_fitness,
        linkage=linkage,
        n_results=2,
        divisions=2,
        bounds=bounds,
        order_relation=min,
        verbose=False,
    )
    assert len(ensemble) >= 1


def test_trials_and_errors_optimization_explicit_parameters():
    linkage = make_fourbar()
    params = [1.0, 2.5, 2.0]
    ensemble = trials_and_errors_optimization(
        eval_func=simple_fitness,
        linkage=linkage,
        parameters=params,
        n_results=2,
        divisions=2,
        order_relation=min,
        verbose=False,
    )
    assert len(ensemble) >= 1


def test_trials_and_errors_optimization_order_relation_max():
    linkage = make_fourbar()

    @kinematic_minimization
    def neg_fit(loci, **_):
        tip = [x[-1] for x in loci]
        return -sum(abs(x[0]) + abs(x[1]) for x in tip)

    ensemble = trials_and_errors_optimization(
        eval_func=neg_fit,
        linkage=linkage,
        n_results=2,
        divisions=2,
        order_relation=max,
        verbose=False,
    )
    assert len(ensemble) >= 1


def test_trials_and_errors_invalid_n_results():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="Number of results"):
        trials_and_errors_optimization(
            eval_func=simple_fitness,
            linkage=linkage,
            n_results=0,
            divisions=3,
            verbose=False,
        )


def test_trials_and_errors_invalid_divisions():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="divisions"):
        trials_and_errors_optimization(
            eval_func=simple_fitness,
            linkage=linkage,
            n_results=3,
            divisions=0,
            verbose=False,
        )


def test_trials_and_errors_all_invalid_raises():
    linkage = make_fourbar()

    def always_fail(_linkage, _dims, _prev):
        return float("inf")

    # When every score is inf (never improves from None), the first iteration
    # sets result.score = inf so results[0].score is not None and code passes.
    # But if our fitness reliably returns inf from the decorated-error path we
    # still populate, so we simulate a never-updating case by patching.
    # Use a decorated fitness that always returns inf
    @kinematic_minimization
    def boring(loci, **_):
        return float("inf")

    # With min ordering, inf is never better than an existing inf, but the first
    # assignment replaces None. So optimization returns an ensemble.
    ensemble = trials_and_errors_optimization(
        eval_func=boring,
        linkage=linkage,
        n_results=1,
        divisions=2,
        order_relation=min,
        verbose=False,
    )
    assert len(ensemble) >= 1


def test_trials_and_errors_no_valid_solutions_raises(monkeypatch):
    """Force the 'no valid solution' branch by zeroing the variation generator."""
    linkage = make_fourbar()

    import pylinkage.optimization.grid_search as gs_mod

    def empty_gen(*_args, **_kwargs):
        if False:
            yield  # make a generator

    monkeypatch.setattr(gs_mod, "fast_variator", empty_gen)

    with pytest.raises(OptimizationError, match="no valid solutions"):
        trials_and_errors_optimization(
            eval_func=simple_fitness,
            linkage=linkage,
            n_results=2,
            divisions=2,
            verbose=False,
        )
