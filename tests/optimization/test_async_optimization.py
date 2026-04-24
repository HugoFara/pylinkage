"""Tests for pylinkage.optimization.async_optimization."""

from __future__ import annotations

import asyncio

import pytest

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRRDyad
from pylinkage.optimization.async_optimization import (
    OptimizationProgress,
    differential_evolution_optimization_async,
    minimize_linkage_async,
    particle_swarm_optimization_async,
    trials_and_errors_optimization_async,
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


# ---- OptimizationProgress ----


def test_optimization_progress_fraction():
    p = OptimizationProgress(current_iteration=3, total_iterations=10)
    assert p.progress_fraction == pytest.approx(0.3)


def test_optimization_progress_fraction_zero_total():
    p = OptimizationProgress(current_iteration=0, total_iterations=0)
    assert p.progress_fraction == 0.0


def test_optimization_progress_defaults():
    p = OptimizationProgress(current_iteration=1, total_iterations=5)
    assert p.best_score is None
    assert p.is_complete is False


# ---- particle_swarm_optimization_async ----


def test_particle_swarm_async_basic():
    linkage = make_fourbar()
    events: list[OptimizationProgress] = []

    def on_progress(p: OptimizationProgress) -> None:
        events.append(p)

    result = asyncio.run(
        particle_swarm_optimization_async(
            eval_func=simple_fitness,
            linkage=linkage,
            n_particles=4,
            iterations=2,
            neighbors=3,
            order_relation=min,
            on_progress=on_progress,
        )
    )
    assert len(result) == 1
    # Expect at least a start + end event
    assert len(events) >= 2
    assert events[0].current_iteration == 0
    assert events[0].is_complete is False
    assert events[-1].is_complete is True
    assert events[-1].best_score is not None


def test_particle_swarm_async_no_callback():
    linkage = make_fourbar()
    result = asyncio.run(
        particle_swarm_optimization_async(
            eval_func=simple_fitness,
            linkage=linkage,
            n_particles=4,
            iterations=2,
            neighbors=3,
            order_relation=min,
        )
    )
    assert len(result) == 1


def test_particle_swarm_async_iters_backcompat():
    linkage = make_fourbar()
    result = asyncio.run(
        particle_swarm_optimization_async(
            eval_func=simple_fitness,
            linkage=linkage,
            n_particles=4,
            neighbors=3,
            order_relation=min,
            iters=2,  # back-compat alias
        )
    )
    assert len(result) == 1


def test_particle_swarm_async_with_executor():
    from concurrent.futures import ThreadPoolExecutor

    linkage = make_fourbar()
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = asyncio.run(
            particle_swarm_optimization_async(
                eval_func=simple_fitness,
                linkage=linkage,
                n_particles=4,
                iterations=2,
                neighbors=3,
                order_relation=min,
                executor=executor,
            )
        )
    assert len(result) == 1


# ---- trials_and_errors_optimization_async ----


def test_trials_and_errors_async_basic():
    linkage = make_fourbar()
    events: list[OptimizationProgress] = []

    def on_progress(p: OptimizationProgress) -> None:
        events.append(p)

    result = asyncio.run(
        trials_and_errors_optimization_async(
            eval_func=simple_fitness,
            linkage=linkage,
            n_results=2,
            divisions=2,
            order_relation=min,
            on_progress=on_progress,
        )
    )
    assert len(result) >= 1
    assert events[0].is_complete is False
    assert events[-1].is_complete is True
    # total = divisions ** num_params (2**3 = 8)
    assert events[-1].total_iterations == 8


def test_trials_and_errors_async_no_callback():
    linkage = make_fourbar()
    result = asyncio.run(
        trials_and_errors_optimization_async(
            eval_func=simple_fitness,
            linkage=linkage,
            n_results=2,
            divisions=2,
            order_relation=min,
        )
    )
    assert len(result) >= 1


def test_trials_and_errors_async_with_executor_and_parameters():
    from concurrent.futures import ThreadPoolExecutor

    linkage = make_fourbar()
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = asyncio.run(
            trials_and_errors_optimization_async(
                eval_func=simple_fitness,
                linkage=linkage,
                parameters=[1.0, 2.5, 2.0],
                n_results=2,
                divisions=2,
                order_relation=min,
                executor=executor,
            )
        )
    assert len(result) >= 1


# ---- differential_evolution_optimization_async ----


def test_differential_evolution_async_basic():
    linkage = make_fourbar()
    events: list[OptimizationProgress] = []

    def on_progress(p: OptimizationProgress) -> None:
        events.append(p)

    result = asyncio.run(
        differential_evolution_optimization_async(
            eval_func=simple_fitness,
            linkage=linkage,
            maxiter=3,
            popsize=4,
            order_relation=min,
            seed=0,
            on_progress=on_progress,
        )
    )
    assert len(result) == 1
    assert events[0].is_complete is False
    assert events[-1].is_complete is True
    assert events[-1].total_iterations == 3


def test_differential_evolution_async_no_callback():
    linkage = make_fourbar()
    result = asyncio.run(
        differential_evolution_optimization_async(
            eval_func=simple_fitness,
            linkage=linkage,
            maxiter=3,
            popsize=4,
            order_relation=min,
            seed=0,
        )
    )
    assert len(result) == 1


def test_differential_evolution_async_with_executor():
    from concurrent.futures import ThreadPoolExecutor

    linkage = make_fourbar()
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = asyncio.run(
            differential_evolution_optimization_async(
                eval_func=simple_fitness,
                linkage=linkage,
                maxiter=3,
                popsize=4,
                order_relation=min,
                seed=0,
                executor=executor,
            )
        )
    assert len(result) == 1


# ---- minimize_linkage_async ----


def test_minimize_linkage_async_basic():
    linkage = make_fourbar()
    events: list[OptimizationProgress] = []

    def on_progress(p: OptimizationProgress) -> None:
        events.append(p)

    result = asyncio.run(
        minimize_linkage_async(
            eval_func=simple_fitness,
            linkage=linkage,
            method="Nelder-Mead",
            maxiter=10,
            order_relation=min,
            on_progress=on_progress,
        )
    )
    assert len(result) == 1
    assert events[0].is_complete is False
    assert events[-1].is_complete is True
    assert events[-1].total_iterations == 10


def test_minimize_linkage_async_default_maxiter():
    """When maxiter=None, total_iterations should default to 1000."""
    linkage = make_fourbar()
    events: list[OptimizationProgress] = []

    def on_progress(p: OptimizationProgress) -> None:
        events.append(p)

    result = asyncio.run(
        minimize_linkage_async(
            eval_func=simple_fitness,
            linkage=linkage,
            method="Nelder-Mead",
            order_relation=min,
            on_progress=on_progress,
            tol=1.0,  # Loose tol so it exits quickly
        )
    )
    assert len(result) == 1
    assert events[0].total_iterations == 1000


def test_minimize_linkage_async_no_callback():
    linkage = make_fourbar()
    result = asyncio.run(
        minimize_linkage_async(
            eval_func=simple_fitness,
            linkage=linkage,
            method="Nelder-Mead",
            maxiter=5,
            order_relation=min,
        )
    )
    assert len(result) == 1


def test_minimize_linkage_async_with_executor():
    from concurrent.futures import ThreadPoolExecutor

    linkage = make_fourbar()
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = asyncio.run(
            minimize_linkage_async(
                eval_func=simple_fitness,
                linkage=linkage,
                method="Nelder-Mead",
                maxiter=5,
                order_relation=min,
                executor=executor,
            )
        )
    assert len(result) == 1
