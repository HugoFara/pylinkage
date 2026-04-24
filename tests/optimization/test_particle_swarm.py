"""Tests for pylinkage.optimization.particle_swarm."""

from __future__ import annotations

import numpy as np
import pytest

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRRDyad
from pylinkage.exceptions import OptimizationError
from pylinkage.optimization.particle_swarm import (
    _local_best_pso,
    particle_swarm_optimization,
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


def test_local_best_pso_sphere_without_center(capsys):
    def objective(x: np.ndarray) -> np.ndarray:
        return np.sum(x * x, axis=1)

    lb = np.array([-5.0, -5.0])
    ub = np.array([5.0, 5.0])
    cost, pos = _local_best_pso(
        objective=objective,
        dimensions=2,
        bounds=(lb, ub),
        n_particles=6,
        iterations=4,
        neighbors=3,
        verbose=False,
    )
    assert pos.shape == (2,)
    assert isinstance(cost, float)


def test_local_best_pso_with_center_and_verbose(capsys):
    def objective(x: np.ndarray) -> np.ndarray:
        return np.sum((x - 1.0) ** 2, axis=1)

    lb = np.array([-5.0, -5.0])
    ub = np.array([5.0, 5.0])
    cost, pos = _local_best_pso(
        objective=objective,
        dimensions=2,
        bounds=(lb, ub),
        n_particles=5,
        iterations=3,
        neighbors=3,
        center=np.array([1.0, 1.0]),
        verbose=True,
    )
    captured = capsys.readouterr()
    # verbose should print to stdout
    assert "PSO iter" in captured.out
    assert pos.shape == (2,)


def test_particle_swarm_optimization_basic():
    linkage = make_fourbar()
    ensemble = particle_swarm_optimization(
        eval_func=simple_fitness,
        linkage=linkage,
        n_particles=4,
        iterations=2,
        neighbors=3,
        order_relation=min,
        verbose=False,
    )
    assert len(ensemble) == 1
    assert ensemble.dimensions.shape == (1, 3)


def test_particle_swarm_optimization_max():
    linkage = make_fourbar()

    @kinematic_minimization
    def neg_fit(loci, **_):
        tip = [x[-1] for x in loci]
        return -sum(abs(x[0]) + abs(x[1]) for x in tip)

    ensemble = particle_swarm_optimization(
        eval_func=neg_fit,
        linkage=linkage,
        n_particles=4,
        iterations=2,
        neighbors=3,
        order_relation=max,
        verbose=False,
    )
    assert len(ensemble) == 1


def test_particle_swarm_optimization_verbose(capsys):
    linkage = make_fourbar()
    particle_swarm_optimization(
        eval_func=simple_fitness,
        linkage=linkage,
        n_particles=4,
        iterations=2,
        neighbors=3,
        order_relation=min,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "PSO iter" in captured.out


def test_particle_swarm_optimization_with_center_sequence():
    linkage = make_fourbar()
    ensemble = particle_swarm_optimization(
        eval_func=simple_fitness,
        linkage=linkage,
        center=[1.0, 2.5, 2.0],
        n_particles=4,
        iterations=2,
        neighbors=3,
        order_relation=min,
        verbose=False,
    )
    assert len(ensemble) == 1


def test_particle_swarm_optimization_with_center_scalar():
    linkage = make_fourbar()
    # A scalar center is valid per signature (int/float branch)
    ensemble = particle_swarm_optimization(
        eval_func=simple_fitness,
        linkage=linkage,
        center=1.0,
        n_particles=4,
        iterations=2,
        neighbors=3,
        order_relation=min,
        verbose=False,
    )
    assert len(ensemble) == 1


def test_particle_swarm_optimization_explicit_bounds():
    linkage = make_fourbar()
    bounds = ([0.5, 1.5, 1.0], [2.0, 4.0, 3.5])
    ensemble = particle_swarm_optimization(
        eval_func=simple_fitness,
        linkage=linkage,
        n_particles=4,
        iterations=2,
        neighbors=3,
        bounds=bounds,
        order_relation=min,
        verbose=False,
    )
    assert len(ensemble) == 1


def test_particle_swarm_optimization_iters_backcompat():
    linkage = make_fourbar()
    ensemble = particle_swarm_optimization(
        eval_func=simple_fitness,
        linkage=linkage,
        n_particles=4,
        neighbors=3,
        order_relation=min,
        verbose=False,
        iters=2,
    )
    assert len(ensemble) == 1


def test_particle_swarm_optimization_invalid_dimensions():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="Dimensions"):
        particle_swarm_optimization(
            eval_func=simple_fitness,
            linkage=linkage,
            dimensions=0,
            n_particles=4,
            iterations=2,
            neighbors=3,
            verbose=False,
        )


def test_particle_swarm_optimization_invalid_n_particles():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="particles"):
        particle_swarm_optimization(
            eval_func=simple_fitness,
            linkage=linkage,
            n_particles=0,
            iterations=2,
            neighbors=3,
            verbose=False,
        )


def test_particle_swarm_optimization_invalid_iterations():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="iterations"):
        particle_swarm_optimization(
            eval_func=simple_fitness,
            linkage=linkage,
            n_particles=4,
            iterations=0,
            neighbors=3,
            verbose=False,
        )


def test_particle_swarm_optimization_invalid_bounds_length():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="Bounds must be a tuple"):
        particle_swarm_optimization(
            eval_func=simple_fitness,
            linkage=linkage,
            n_particles=4,
            iterations=2,
            neighbors=3,
            bounds=([0.0, 0.0, 0.0],),  # type: ignore[arg-type]
            verbose=False,
        )


def test_particle_swarm_optimization_invalid_bounds_dimension_mismatch():
    linkage = make_fourbar()
    with pytest.raises(OptimizationError, match="Bounds dimensions"):
        particle_swarm_optimization(
            eval_func=simple_fitness,
            linkage=linkage,
            n_particles=4,
            iterations=2,
            neighbors=3,
            bounds=([0.0, 0.0], [1.0, 1.0]),  # wrong length
            verbose=False,
        )
