"""Type definitions for topology + dimension co-optimization.

Provides the chromosome encoding, configuration, and result types
used by the mixed-variable evolutionary optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..linkage import Linkage
    from ..topology.catalog import CatalogEntry
    from .collections.pareto import ParetoFront


@dataclass
class MixedChromosome:
    """Chromosome encoding for co-optimization.

    Encodes both discrete topology choices and continuous dimensional
    parameters in a single structure that genetic operators can
    manipulate.

    Attributes:
        topology_idx: Integer index into the catalog's topology list.
        dimensions: Continuous variables — link lengths in the order
            defined by the topology's edge list.
    """

    topology_idx: int
    dimensions: NDArray[np.floating[Any]]

    def copy(self) -> MixedChromosome:
        """Return a deep copy."""
        return MixedChromosome(
            topology_idx=self.topology_idx,
            dimensions=self.dimensions.copy(),
        )


@dataclass(frozen=True)
class CoOptimizationConfig:
    """Configuration for a co-optimization run.

    Attributes:
        max_links: Maximum number of links to consider from catalog.
        algorithm: NSGA variant ("nsga2" or "nsga3").
        n_generations: Number of evolutionary generations.
        pop_size: Population size.
        dimension_bounds_factor: Link lengths are bounded to
            [original / factor, original * factor].
        topology_mutation_rate: Probability of mutating the topology.
        dimension_mutation_sigma: Std dev for Gaussian dimension mutation
            (as fraction of the variable range).
        crossover_prob: Crossover probability.
        seed: Random seed for reproducibility.
        verbose: Print progress during optimization.
    """

    max_links: int = 8
    algorithm: Literal["nsga2", "nsga3"] = "nsga2"
    n_generations: int = 200
    pop_size: int = 100
    dimension_bounds_factor: float = 5.0
    topology_mutation_rate: float = 0.1
    dimension_mutation_sigma: float = 0.2
    crossover_prob: float = 0.9
    seed: int | None = None
    verbose: bool = True


@dataclass
class CoOptSolution:
    """A single solution from co-optimization.

    Attributes:
        chromosome: The mixed chromosome that produced this solution.
        scores: Objective values (one per objective, all minimized).
        linkage: The built Linkage (if construction succeeded).
        topology_entry: The catalog entry for this topology.
    """

    chromosome: MixedChromosome
    scores: tuple[float, ...]
    linkage: Linkage | None = None
    topology_entry: CatalogEntry | None = None


@dataclass
class CoOptimizationResult:
    """Result of topology + dimension co-optimization.

    Attributes:
        pareto_front: The Pareto front of non-dominated solutions.
        solutions: All solutions with topology metadata.
        config: The configuration used.
        n_evaluations: Total number of fitness evaluations.
        convergence_history: Best score per generation (if tracked).
    """

    pareto_front: ParetoFront
    solutions: list[CoOptSolution] = field(default_factory=list)
    config: CoOptimizationConfig = field(default_factory=CoOptimizationConfig)
    n_evaluations: int = 0
    convergence_history: list[float] = field(default_factory=list)
