"""Warm-start co-optimization from Phase 3 synthesis results.

Converts multi-topology synthesis solutions into an initial population
for the evolutionary co-optimizer, then runs the optimizer with those
seeds. This avoids cold-start and focuses search on promising regions.

Example::

    from pylinkage.optimization.warm_start import warm_start_co_optimization
    from pylinkage.synthesis.ranking import compute_path_accuracy

    points = [(0, 0), (1, 2), (2, 3), (3, 2), (4, 0)]

    def accuracy(linkage):
        return compute_path_accuracy(linkage, points)

    result = warm_start_co_optimization(
        precision_points=points,
        objectives=[accuracy],
        objective_names=["Path Error"],
    )
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .co_optimization_types import (
    CoOptimizationConfig,
    CoOptimizationResult,
    MixedChromosome,
)

if TYPE_CHECKING:
    from ..synthesis._types import PrecisionPoint
    from ..synthesis.topology_types import TopologySolution
    from ..topology.catalog import TopologyCatalog

logger = logging.getLogger(__name__)


def synthesis_to_chromosomes(
    solutions: list[TopologySolution],
    catalog: TopologyCatalog,
) -> list[MixedChromosome]:
    """Convert Phase 3 TopologySolution list to MixedChromosome seeds.

    For each TopologySolution:
    1. Look up the topology index from the catalog.
    2. Extract link lengths from the NBarSolution.
    3. Pack into a MixedChromosome.

    Args:
        solutions: Ranked solutions from multi-topology synthesis.
        catalog: The topology catalog (needed for index mapping).

    Returns:
        List of MixedChromosome, one per valid solution.
    """
    chromosomes: list[MixedChromosome] = []

    for sol in solutions:
        try:
            topo_id = sol.topology_entry.id
            topo_idx = catalog.topology_index(topo_id)
        except (KeyError, AttributeError):
            continue

        # Extract link lengths in edge order from the topology graph
        try:
            graph = sol.topology_entry.to_graph()
            edges = list(graph.edges.values())
            link_lengths = sol.solution.link_lengths

            dims: list[float] = []
            for edge in edges:
                length = link_lengths.get(edge.id)
                if length is not None and length > 0:
                    dims.append(length)
                else:
                    # Infer from joint positions
                    pos = sol.solution.joint_positions
                    if edge.source in pos and edge.target in pos:
                        p1, p2 = pos[edge.source], pos[edge.target]
                        d = math.sqrt(
                            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                        )
                        dims.append(max(d, 0.1))
                    else:
                        dims.append(1.0)

            chromosomes.append(MixedChromosome(
                topology_idx=topo_idx,
                dimensions=np.array(dims, dtype=np.float64),
            ))
        except Exception as exc:
            logger.debug("Failed to convert solution %s: %s", sol.topology_entry.id, exc)
            continue

    return chromosomes


def warm_start_co_optimization(
    precision_points: list[PrecisionPoint],
    objectives: Sequence[Callable[..., float]],
    catalog: TopologyCatalog | None = None,
    config: CoOptimizationConfig | None = None,
    objective_names: Sequence[str] | None = None,
    max_synthesis_solutions: int = 20,
    n_orientation_samples: int = 12,
) -> CoOptimizationResult:
    """Full pipeline: Phase 3 synthesis -> seed population -> co-optimize.

    1. Run multi-topology synthesis to get candidate solutions.
    2. Convert candidates to MixedChromosome seeds.
    3. Run co-optimization with those seeds as initial population.

    Args:
        precision_points: Target (x, y) points.
        objectives: Callables ``(linkage) -> float``, all minimized.
        catalog: Topology catalog. If None, loads built-in.
        config: Co-optimization config. If None, uses defaults.
        objective_names: Names for each objective.
        max_synthesis_solutions: Max solutions from Phase 3 synthesis.
        n_orientation_samples: Orientation search density for synthesis.

    Returns:
        CoOptimizationResult with Pareto front seeded from synthesis.
    """
    if catalog is None:
        from ..topology.catalog import load_catalog
        catalog = load_catalog()

    if config is None:
        config = CoOptimizationConfig()

    # Phase 3: multi-topology synthesis for initial candidates
    logger.info("Running Phase 3 synthesis for warm-start candidates...")
    from ..synthesis.multi_topology import synthesize

    synthesis_solutions = synthesize(
        precision_points,
        max_links=config.max_links,
        max_total_solutions=max_synthesis_solutions,
        n_orientation_samples=n_orientation_samples,
    )

    logger.info(
        "Phase 3 produced %d candidates, converting to chromosomes...",
        len(synthesis_solutions),
    )

    # Convert to chromosomes
    initial_population = synthesis_to_chromosomes(synthesis_solutions, catalog)
    logger.info("Converted %d candidates to seed chromosomes", len(initial_population))

    # Run co-optimization
    from .mixed_variable import co_optimize

    return co_optimize(
        objectives=objectives,
        precision_points=precision_points,
        catalog=catalog,
        config=config,
        initial_population=initial_population,
        objective_names=objective_names,
    )
