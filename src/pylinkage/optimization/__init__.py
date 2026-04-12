"""Optimization package."""

__all__ = [
    "chain_optimizers",
    "co_optimize",
    "CoOptimizationConfig",
    "CoOptimizationResult",
    "CoOptSolution",
    "collections",
    "Ensemble",
    "differential_evolution_optimization",
    "differential_evolution_optimization_async",
    "dual_annealing_optimization",
    "generate_bounds",
    "kinematic_maximization",
    "kinematic_minimization",
    "MixedChromosome",
    "minimize_linkage",
    "minimize_linkage_async",
    "multi_objective_optimization",
    "OptimizationProgress",
    "ParetoFront",
    "ParetoSolution",
    "particle_swarm_optimization",
    "particle_swarm_optimization_async",
    "TopologyNeighbor",
    "topology_neighbors",
    "trials_and_errors_optimization",
    "trials_and_errors_optimization_async",
    "warm_start_co_optimization",
]

import importlib as _importlib

from ..population import Ensemble as Ensemble

# Eagerly import lightweight submodules
from . import collections as collections
from .collections.pareto import (
    ParetoFront as ParetoFront,
)
from .collections.pareto import (
    ParetoSolution as ParetoSolution,
)
from .utils import (
    generate_bounds as generate_bounds,
)
from .utils import (
    kinematic_maximization as kinematic_maximization,
)
from .utils import (
    kinematic_minimization as kinematic_minimization,
)

# Lazy-loaded attributes (require scipy / pyswarms)
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "chain_optimizers": (".scipy_optimize", "chain_optimizers"),
    "differential_evolution_optimization": (
        ".scipy_optimize",
        "differential_evolution_optimization",
    ),
    "dual_annealing_optimization": (
        ".scipy_optimize",
        "dual_annealing_optimization",
    ),
    "minimize_linkage": (".scipy_optimize", "minimize_linkage"),
    "particle_swarm_optimization": (
        ".particle_swarm",
        "particle_swarm_optimization",
    ),
    "trials_and_errors_optimization": (
        ".grid_search",
        "trials_and_errors_optimization",
    ),
    "MixedChromosome": (".co_optimization_types", "MixedChromosome"),
    "CoOptimizationConfig": (".co_optimization_types", "CoOptimizationConfig"),
    "CoOptimizationResult": (".co_optimization_types", "CoOptimizationResult"),
    "CoOptSolution": (".co_optimization_types", "CoOptSolution"),
    "co_optimize": (".mixed_variable", "co_optimize"),
    "TopologyNeighbor": (".topology_neighborhood", "TopologyNeighbor"),
    "topology_neighbors": (".topology_neighborhood", "topology_neighbors"),
    "warm_start_co_optimization": (".warm_start", "warm_start_co_optimization"),
    "multi_objective_optimization": (
        ".multi_objective",
        "multi_objective_optimization",
    ),
    "OptimizationProgress": (".async_optimization", "OptimizationProgress"),
    "differential_evolution_optimization_async": (
        ".async_optimization",
        "differential_evolution_optimization_async",
    ),
    "minimize_linkage_async": (".async_optimization", "minimize_linkage_async"),
    "particle_swarm_optimization_async": (
        ".async_optimization",
        "particle_swarm_optimization_async",
    ),
    "trials_and_errors_optimization_async": (
        ".async_optimization",
        "trials_and_errors_optimization_async",
    ),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        mod = _importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
