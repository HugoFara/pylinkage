"""Custom pymoo operators for mixed topology + dimension optimization.

Provides crossover, mutation, and sampling operators that handle the
mixed discrete/continuous chromosome encoding used by the co-optimizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .co_optimization_types import MixedChromosome


class MixedCrossover:
    """Crossover operator for mixed chromosomes.

    For topology: randomly inherit from either parent.
    For dimensions: BLX-alpha blend crossover on continuous variables.
    When parents have different topologies, the child gets one parent's
    topology and the other parent's dimensions are re-sampled within
    bounds (since dimensions are topology-specific).
    """

    def __init__(
        self,
        crossover_prob: float = 0.9,
        alpha: float = 0.5,
    ) -> None:
        self.crossover_prob = crossover_prob
        self.alpha = alpha

    def __call__(
        self,
        parent_a: NDArray[np.floating[Any]],
        parent_b: NDArray[np.floating[Any]],
        xl: NDArray[np.floating[Any]],
        xu: NDArray[np.floating[Any]],
        n_topologies: int,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Produce two offspring from two parents.

        Args:
            parent_a: First parent chromosome (flat array).
            parent_b: Second parent chromosome (flat array).
            xl: Lower bounds for all variables.
            xu: Upper bounds for all variables.
            n_topologies: Number of topologies in catalog.
            rng: Random number generator.

        Returns:
            Tuple of two offspring arrays.
        """
        child_a = parent_a.copy()
        child_b = parent_b.copy()

        if rng.random() > self.crossover_prob:
            return child_a, child_b

        # Topology gene (index 0): random inheritance
        if rng.random() < 0.5:
            child_a[0], child_b[0] = parent_b[0], parent_a[0]

        # Dimension genes (index 1+): BLX-alpha blend
        dim_start = 1
        for i in range(dim_start, len(parent_a)):
            lo = min(parent_a[i], parent_b[i])
            hi = max(parent_a[i], parent_b[i])
            span = hi - lo
            blend_lo = max(xl[i], lo - self.alpha * span)
            blend_hi = min(xu[i], hi + self.alpha * span)

            child_a[i] = rng.uniform(blend_lo, blend_hi)
            child_b[i] = rng.uniform(blend_lo, blend_hi)

        return child_a, child_b


class MixedMutation:
    """Combined mutation for mixed chromosomes.

    - Topology gene: with ``topology_rate``, mutate to a neighbor
      in the topology neighborhood graph.
    - Dimension genes: Gaussian perturbation with ``sigma`` fraction
      of the variable range.
    """

    def __init__(
        self,
        topology_rate: float = 0.1,
        dimension_sigma: float = 0.2,
        dimension_rate: float = 0.2,
    ) -> None:
        self.topology_rate = topology_rate
        self.dimension_sigma = dimension_sigma
        self.dimension_rate = dimension_rate

    def __call__(
        self,
        x: NDArray[np.floating[Any]],
        xl: NDArray[np.floating[Any]],
        xu: NDArray[np.floating[Any]],
        n_topologies: int,
        neighbor_map: dict[int, list[int]],
        rng: np.random.Generator,
    ) -> NDArray[np.floating[Any]]:
        """Mutate a single chromosome in place.

        Args:
            x: Chromosome (flat array, modified in place).
            xl: Lower bounds.
            xu: Upper bounds.
            n_topologies: Number of topologies in catalog.
            neighbor_map: Maps topology index to list of neighbor indices.
            rng: Random number generator.

        Returns:
            The mutated chromosome.
        """
        # Topology mutation
        if rng.random() < self.topology_rate:
            topo_idx = int(x[0])
            neighbors = neighbor_map.get(topo_idx, [])
            if neighbors:
                x[0] = float(rng.choice(neighbors))
            else:
                # Fallback: random topology
                x[0] = float(rng.integers(0, n_topologies))

        # Dimension mutation: Gaussian perturbation
        dim_start = 1
        for i in range(dim_start, len(x)):
            if rng.random() < self.dimension_rate:
                span = xu[i] - xl[i]
                noise = rng.normal(0, self.dimension_sigma * span)
                x[i] = np.clip(x[i] + noise, xl[i], xu[i])

        return x


def warm_start_sampling(
    chromosomes: list[MixedChromosome],
    pop_size: int,
    n_dim: int,
    xl: NDArray[np.floating[Any]],
    xu: NDArray[np.floating[Any]],
    n_topologies: int,
    rng: np.random.Generator,
) -> NDArray[np.floating[Any]]:
    """Create initial population seeded with synthesis results.

    Fills up to ``pop_size`` by:
    1. Including all provided chromosomes.
    2. Filling remaining slots with random chromosomes.

    Args:
        chromosomes: Seed chromosomes from Phase 3 synthesis.
        pop_size: Target population size.
        n_dim: Total chromosome length (1 + n_dimensions).
        xl: Lower bounds for all variables.
        xu: Upper bounds for all variables.
        n_topologies: Number of topologies in catalog.
        rng: Random number generator.

    Returns:
        Population array of shape (pop_size, n_dim).
    """
    pop = np.zeros((pop_size, n_dim))

    # Fill with seed chromosomes
    n_seeds = min(len(chromosomes), pop_size)
    for i in range(n_seeds):
        ch = chromosomes[i]
        pop[i, 0] = float(ch.topology_idx)
        n_copy = min(len(ch.dimensions), n_dim - 1)
        pop[i, 1 : 1 + n_copy] = ch.dimensions[:n_copy]
        # Clip to bounds
        pop[i] = np.clip(pop[i], xl, xu)

    # Fill remaining with random
    for i in range(n_seeds, pop_size):
        pop[i, 0] = float(rng.integers(0, n_topologies))
        for j in range(1, n_dim):
            pop[i, j] = rng.uniform(xl[j], xu[j])

    return pop
