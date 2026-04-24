"""Tests for pylinkage.optimization.scipy_optimize."""

from __future__ import annotations

import pytest

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRRDyad
from pylinkage.exceptions import OptimizationError
from pylinkage.optimization.scipy_optimize import (
    chain_optimizers,
    differential_evolution_optimization,
    dual_annealing_optimization,
    minimize_linkage,
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


# ---- differential_evolution_optimization ----


def test_differential_evolution_basic():
    linkage = make_fourbar()
    ensemble = differential_evolution_optimization(
        eval_func=simple_fitness,
        linkage=linkage,
        maxiter=3,
        popsize=4,
        order_relation=min,
        seed=0,
        verbose=False,
    )
    assert len(ensemble) == 1
    assert ensemble.dimensions.shape == (1, 3)


def test_differential_evolution_max_order():
    linkage = make_fourbar()

    @kinematic_minimization
    def neg_fit(loci, **_):
        tip = [x[-1] for x in loci]
        return -sum(abs(x[0]) + abs(x[1]) for x in tip)

    ensemble = differential_evolution_optimization(
        eval_func=neg_fit,
        linkage=linkage,
        maxiter=3,
        popsize=4,
        order_relation=max,
        seed=0,
        verbose=False,
    )
    assert len(ensemble) == 1


def test_differential_evolution_with_explicit_bounds():
    linkage = make_fourbar()
    ensemble = differential_evolution_optimization(
        eval_func=simple_fitness,
        linkage=linkage,
        bounds=([0.5, 1.5, 1.0], [2.0, 4.0, 3.5]),
        maxiter=3,
        popsize=4,
        order_relation=min,
        seed=0,
        verbose=False,
    )
    assert len(ensemble) == 1


def test_differential_evolution_invalid_maxiter():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="iterations"):
        differential_evolution_optimization(
            eval_func=simple_fitness,
            linkage=linkage,
            maxiter=0,
            verbose=False,
        )


def test_differential_evolution_invalid_popsize():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="Population size"):
        differential_evolution_optimization(
            eval_func=simple_fitness,
            linkage=linkage,
            popsize=0,
            verbose=False,
        )


def test_differential_evolution_invalid_bounds_len():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="Bounds must be a tuple"):
        differential_evolution_optimization(
            eval_func=simple_fitness,
            linkage=linkage,
            bounds=([0.0, 0.0, 0.0],),  # type: ignore[arg-type]
            maxiter=3,
            popsize=4,
            verbose=False,
        )


def test_differential_evolution_bounds_dimension_mismatch():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="Bounds dimensions"):
        differential_evolution_optimization(
            eval_func=simple_fitness,
            linkage=linkage,
            bounds=([0.0, 0.0], [1.0, 1.0]),
            maxiter=3,
            popsize=4,
            verbose=False,
        )


# ---- dual_annealing_optimization ----


def test_dual_annealing_basic():
    linkage = make_fourbar()
    ensemble = dual_annealing_optimization(
        eval_func=simple_fitness,
        linkage=linkage,
        maxiter=3,
        order_relation=min,
        seed=0,
        verbose=False,
    )
    assert len(ensemble) == 1
    assert ensemble.dimensions.shape == (1, 3)


def test_dual_annealing_verbose_prints(capsys):
    linkage = make_fourbar()
    dual_annealing_optimization(
        eval_func=simple_fitness,
        linkage=linkage,
        maxiter=3,
        order_relation=min,
        seed=0,
        verbose=True,
    )
    # verbose callback prints progress (iteration count only when context fires)
    # A print("") is always emitted after completion, so stdout is touched.
    _ = capsys.readouterr()


def test_dual_annealing_max_order_with_bounds():
    linkage = make_fourbar()

    @kinematic_minimization
    def neg_fit(loci, **_):
        tip = [x[-1] for x in loci]
        return -sum(abs(x[0]) + abs(x[1]) for x in tip)

    ensemble = dual_annealing_optimization(
        eval_func=neg_fit,
        linkage=linkage,
        bounds=([0.5, 1.5, 1.0], [2.0, 4.0, 3.5]),
        maxiter=3,
        order_relation=max,
        seed=0,
        verbose=False,
    )
    assert len(ensemble) == 1


def test_dual_annealing_invalid_maxiter():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="iterations"):
        dual_annealing_optimization(
            eval_func=simple_fitness,
            linkage=linkage,
            maxiter=0,
            verbose=False,
        )


def test_dual_annealing_invalid_bounds_len():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="Bounds must be a tuple"):
        dual_annealing_optimization(
            eval_func=simple_fitness,
            linkage=linkage,
            bounds=([0.0, 0.0, 0.0],),  # type: ignore[arg-type]
            maxiter=3,
            verbose=False,
        )


def test_dual_annealing_bounds_dimension_mismatch():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="Bounds dimensions"):
        dual_annealing_optimization(
            eval_func=simple_fitness,
            linkage=linkage,
            bounds=([0.0, 0.0], [1.0, 1.0]),
            maxiter=3,
            verbose=False,
        )


# ---- minimize_linkage ----


def test_minimize_linkage_nelder_mead_default():
    linkage = make_fourbar()
    ensemble = minimize_linkage(
        eval_func=simple_fitness,
        linkage=linkage,
        method="Nelder-Mead",
        maxiter=10,
        order_relation=min,
        verbose=False,
    )
    assert len(ensemble) == 1


def test_minimize_linkage_bounded_method_lbfgsb():
    linkage = make_fourbar()
    ensemble = minimize_linkage(
        eval_func=simple_fitness,
        linkage=linkage,
        method="L-BFGS-B",
        maxiter=5,
        order_relation=min,
        verbose=False,
    )
    assert len(ensemble) == 1


def test_minimize_linkage_bounded_method_with_explicit_bounds():
    linkage = make_fourbar()
    ensemble = minimize_linkage(
        eval_func=simple_fitness,
        linkage=linkage,
        method="SLSQP",
        bounds=([0.5, 1.5, 1.0], [2.0, 4.0, 3.5]),
        maxiter=5,
        order_relation=min,
        verbose=False,
    )
    assert len(ensemble) == 1


def test_minimize_linkage_custom_x0():
    linkage = make_fourbar()
    ensemble = minimize_linkage(
        eval_func=simple_fitness,
        linkage=linkage,
        x0=[1.0, 2.5, 2.0],
        method="Nelder-Mead",
        maxiter=5,
        order_relation=min,
        verbose=False,
    )
    assert len(ensemble) == 1


def test_minimize_linkage_max_order():
    linkage = make_fourbar()

    @kinematic_minimization
    def neg_fit(loci, **_):
        tip = [x[-1] for x in loci]
        return -sum(abs(x[0]) + abs(x[1]) for x in tip)

    ensemble = minimize_linkage(
        eval_func=neg_fit,
        linkage=linkage,
        method="Nelder-Mead",
        maxiter=5,
        order_relation=max,
        verbose=False,
    )
    assert len(ensemble) == 1


def test_minimize_linkage_verbose(capsys):
    linkage = make_fourbar()
    minimize_linkage(
        eval_func=simple_fitness,
        linkage=linkage,
        method="Nelder-Mead",
        maxiter=3,
        order_relation=min,
        verbose=True,
    )
    # `disp=True` triggers scipy output
    _ = capsys.readouterr()


def test_minimize_linkage_invalid_x0_length():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="Initial guess length"):
        minimize_linkage(
            eval_func=simple_fitness,
            linkage=linkage,
            x0=[1.0, 2.5],  # wrong length (2 vs 3)
            method="Nelder-Mead",
            verbose=False,
        )


# ---- chain_optimizers ----


def test_chain_optimizers_de_then_minimize():
    linkage = make_fourbar()
    ensemble = chain_optimizers(
        eval_func=simple_fitness,
        linkage=linkage,
        stages=[
            (
                differential_evolution_optimization,
                {"maxiter": 3, "popsize": 4, "seed": 0},
            ),
            (
                minimize_linkage,
                {"method": "Nelder-Mead", "maxiter": 5},
            ),
        ],
        order_relation=min,
        verbose=False,
    )
    assert len(ensemble) == 1


def test_chain_optimizers_verbose(capsys):
    linkage = make_fourbar()
    ensemble = chain_optimizers(
        eval_func=simple_fitness,
        linkage=linkage,
        stages=[
            (
                differential_evolution_optimization,
                {"maxiter": 3, "popsize": 4, "seed": 0},
            ),
        ],
        order_relation=min,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "Stage" in captured.out
    assert len(ensemble) == 1


def test_chain_optimizers_passes_center_to_population_based():
    """Ensure second stage receives `center` when first param isn't x0."""
    from pylinkage.optimization.particle_swarm import particle_swarm_optimization

    linkage = make_fourbar()
    ensemble = chain_optimizers(
        eval_func=simple_fitness,
        linkage=linkage,
        stages=[
            (
                differential_evolution_optimization,
                {"maxiter": 3, "popsize": 4, "seed": 0},
            ),
            (
                particle_swarm_optimization,
                {"n_particles": 4, "iterations": 2, "neighbors": 3},
            ),
        ],
        order_relation=min,
        verbose=False,
    )
    assert len(ensemble) == 1


def test_chain_optimizers_empty_stages_raises():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="At least one"):
        chain_optimizers(
            eval_func=simple_fitness,
            linkage=linkage,
            stages=[],
            verbose=False,
        )
