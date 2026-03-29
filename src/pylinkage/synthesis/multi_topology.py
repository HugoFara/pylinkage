"""Multi-topology synthesis: search across all compatible topologies.

Given precision points, tries every topology from the catalog
(four-bar, six-bar, eight-bar) and returns solutions ranked by
quality metrics. This is the top-level entry point for topology-aware
synthesis.

Example::

    from pylinkage.synthesis.multi_topology import synthesize

    points = [(0, 0), (1, 2), (2, 3), (3, 2), (4, 0)]
    solutions = synthesize(points, max_links=6)

    for sol in solutions[:3]:
        print(f"{sol.topology_entry.name}: score={sol.metrics.overall_score:.3f}")
        sol.linkage.show()
"""

from __future__ import annotations

import logging

from ._types import PrecisionPoint, SynthesisType
from .topology_types import QualityMetrics, TopologySolution

logger = logging.getLogger(__name__)


def synthesize(
    precision_points: list[PrecisionPoint],
    max_links: int = 8,
    synthesis_type: SynthesisType = SynthesisType.PATH,
    max_solutions_per_topology: int = 5,
    max_total_solutions: int = 20,
    n_orientation_samples: int = 24,
) -> list[TopologySolution]:
    """Synthesize linkages across all compatible topologies.

    Tries every topology from the built-in catalog (up to ``max_links``)
    and returns solutions ranked by quality metrics (path accuracy,
    transmission angle, link ratios, compactness, simplicity).

    For four-bars, delegates to the existing ``path_generation()``.
    For six-bars, delegates to ``six_bar_path_generation()``.
    For eight-bars and beyond, delegates to ``generalized_synthesis()``.

    Args:
        precision_points: Target (x, y) points.
        max_links: Maximum number of links to consider (4, 6, or 8).
        synthesis_type: ``PATH``, ``FUNCTION``, or ``MOTION``.
            Currently only PATH is supported for multi-topology.
        max_solutions_per_topology: Max solutions per topology type.
        max_total_solutions: Total solutions to return.
        n_orientation_samples: Orientation search density.

    Returns:
        List of TopologySolution ranked by ``overall_score`` (best first).

    Raises:
        NotImplementedError: If synthesis_type is not PATH.

    Example:
        >>> solutions = synthesize([(0,1), (1,2), (2,1.5), (3,0), (4,1)])
        >>> print(f"Best: {solutions[0].topology_entry.name}")
    """
    if synthesis_type != SynthesisType.PATH:
        raise NotImplementedError(
            f"Multi-topology synthesis currently only supports PATH, got {synthesis_type}."
        )

    from ..topology.catalog import load_catalog
    from .generalized import generalized_synthesis
    from .ranking import compute_metrics

    catalog = load_catalog()
    compatible = catalog.compatible_topologies(max_links=max_links)

    all_candidates: list[TopologySolution] = []

    for entry in compatible:
        logger.debug("Trying topology: %s (%s)", entry.id, entry.family)

        try:
            nbar_solutions = generalized_synthesis(
                entry,
                precision_points,
                partition_strategy="greedy" if entry.num_links > 6 else "exhaustive",
                max_solutions=max_solutions_per_topology,
                n_orientation_samples=n_orientation_samples,
            )
        except Exception as exc:
            logger.debug("Synthesis failed for %s: %s", entry.id, exc)
            continue

        for nbar in nbar_solutions:
            try:
                from .conversion import nbar_solution_to_linkage
                linkage = nbar_solution_to_linkage(nbar)
            except Exception:
                continue

            try:
                metrics = compute_metrics(
                    linkage, precision_points,
                    num_links=entry.num_links,
                )
            except Exception:
                metrics = QualityMetrics(num_links=entry.num_links)

            candidate = TopologySolution(
                solution=nbar,
                linkage=linkage,
                topology_entry=entry,
                metrics=metrics,
            )
            all_candidates.append(candidate)

    # Rank by overall score
    ranked = rank_solutions(all_candidates)
    return ranked[:max_total_solutions]


def rank_solutions(
    candidates: list[TopologySolution],
    weights: dict[str, float] | None = None,
) -> list[TopologySolution]:
    """Rank solutions by weighted quality metrics.

    Default weights prioritize:
      - Path accuracy (0.40)
      - Transmission angle (0.25)
      - Link ratio (0.10)
      - Grashof bonus (0.10)
      - Simplicity (0.10)
      - Compactness (0.05)

    Args:
        candidates: Unranked solutions.
        weights: Optional custom weight dict.

    Returns:
        Solutions sorted by ``overall_score`` (ascending = best first).
    """
    if weights is not None:
        from .ranking import score_solution
        for candidate in candidates:
            candidate.metrics.overall_score = score_solution(
                candidate.metrics, weights
            )

    return sorted(candidates, key=lambda c: c.metrics.overall_score)
