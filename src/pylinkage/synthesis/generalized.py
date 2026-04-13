"""Generalized N-bar synthesis via Assur decomposition.

For any topology from the catalog, decomposes into Assur groups and
synthesizes each group sequentially. Each dyad group is a local
Burmester sub-problem; triad groups use Newton-Raphson optimization.

This generalizes the approach in :mod:`~pylinkage.synthesis.six_bar`
to arbitrary topologies (eight-bars and beyond).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ._types import Point2D, PrecisionPoint
from .burmester import compute_circle_point_curve, select_compatible_dyads
from .path_generation import (
    _dyads_to_fourbar,
    _filter_solutions,
    _generate_orientation_candidates,
    _points_to_poses,
)
from .topology_types import GroupSynthesisResult, NBarSolution, PointPartition

if TYPE_CHECKING:
    from ..assur.decomposition import DecompositionResult
    from ..assur.groups import AssurGroup
    from ..topology.catalog import CatalogEntry


def generalized_synthesis(
    topology: CatalogEntry,
    precision_points: list[PrecisionPoint],
    partition_strategy: str = "greedy",
    max_solutions: int = 10,
    n_orientation_samples: int = 24,
) -> list[NBarSolution]:
    """Synthesize a linkage of arbitrary topology for path generation.

    Retrieves the Assur decomposition of the given topology, partitions
    precision points across groups, and synthesizes each group
    sequentially using Burmester (for dyads) or optimization (for triads).

    Args:
        topology: CatalogEntry from the topology catalog.
        precision_points: Target (x, y) points.
        partition_strategy: ``"greedy"`` or ``"exhaustive"``.
        max_solutions: Maximum solutions to return.
        n_orientation_samples: Orientation search density.

    Returns:
        List of NBarSolution objects (unranked).
    """
    # Special-case: delegate four-bar and six-bar to specialized modules
    if topology.family == "four-bar":
        return _synthesize_fourbar_as_nbar(
            precision_points, max_solutions, n_orientation_samples
        )

    if topology.family == "six-bar":
        return _synthesize_sixbar_as_nbar(
            precision_points, topology.id, max_solutions, n_orientation_samples
        )

    # General case: decompose and synthesize chain
    decomposition = _get_decomposition(topology)
    if decomposition is None:
        return []

    n_groups = len(decomposition.groups)
    if n_groups == 0:
        return []

    # Generate point partitions
    if partition_strategy == "exhaustive":
        partitions = _partition_points_exhaustive(
            len(precision_points), decomposition
        )
    else:
        partitions = _partition_points_greedy(
            len(precision_points), decomposition
        )

    if not partitions:
        return []

    all_solutions: list[NBarSolution] = []

    for partition in partitions:
        if len(all_solutions) >= max_solutions:
            break

        # Try a few orientation candidates
        first_group_points = [precision_points[i] for i in partition[0]]
        orientations_gen = _generate_orientation_candidates(
            first_group_points, n_samples=min(n_orientation_samples, 12)
        )

        for orientations_tried, orientations in enumerate(orientations_gen):
            if len(all_solutions) >= max_solutions:
                break
            if orientations_tried > 6:
                break

            solutions = _synthesize_chain(
                decomposition, topology, precision_points, partition, orientations
            )
            all_solutions.extend(solutions)

    return all_solutions[:max_solutions]


def _get_decomposition(topology: CatalogEntry) -> DecompositionResult | None:
    """Get Assur decomposition for a catalog entry."""
    try:
        from ..assur.decomposition import decompose_assur_groups
        from ..assur.hypergraph_conversion import from_hypergraph

        graph = topology.to_graph()
        assur_graph = from_hypergraph(graph)
        return decompose_assur_groups(assur_graph)
    except Exception:
        return None


def _partition_points_greedy(
    n_points: int,
    decomposition: DecompositionResult,
) -> list[PointPartition]:
    """Greedy point partitioning heuristic.

    Allocates points proportionally based on the number of internal
    nodes per group, ensuring each group gets at least 3 points.

    Args:
        n_points: Total number of precision points.
        decomposition: Assur decomposition result.

    Returns:
        List containing one partition (greedy produces a single result).
    """
    n_groups = len(decomposition.groups)
    if n_groups == 0 or n_points < 3 * n_groups:
        return []

    # Proportional allocation based on group size
    total_internal = sum(len(g.internal_nodes) for g in decomposition.groups)
    if total_internal == 0:
        total_internal = n_groups

    allocation: list[int] = []
    remaining = n_points
    for i, group in enumerate(decomposition.groups):
        if i == n_groups - 1:
            allocation.append(remaining)
        else:
            share = max(3, round(n_points * len(group.internal_nodes) / total_internal))
            share = min(share, min(5, remaining - 3 * (n_groups - i - 1)))
            allocation.append(share)
            remaining -= share

    # Validate
    if any(a < 3 for a in allocation) or any(a > 5 for a in allocation):
        # Fall back to equal split
        base = n_points // n_groups
        allocation = [base] * n_groups
        for i in range(n_points - base * n_groups):
            allocation[i] += 1

    if any(a < 3 for a in allocation):
        return []

    # Build index partition
    idx = 0
    groups: list[tuple[int, ...]] = []
    for count in allocation:
        groups.append(tuple(range(idx, idx + count)))
        idx += count

    return [tuple(groups)]


def _partition_points_exhaustive(
    n_points: int,
    decomposition: DecompositionResult,
    min_per_group: int = 3,
    max_per_group: int = 5,
) -> list[PointPartition]:
    """Exhaustive enumeration of valid point partitions.

    Enumerates all ways to assign n_points across groups where each
    group gets between min_per_group and max_per_group points.

    For more than 2 groups, limits output to avoid combinatorial explosion.
    """
    from itertools import combinations

    n_groups = len(decomposition.groups)
    if n_groups == 0 or n_points < min_per_group * n_groups:
        return []

    if n_groups == 1:
        if min_per_group <= n_points <= max_per_group:
            return [(tuple(range(n_points)),)]
        return []

    if n_groups == 2:
        # Delegate to six_bar-style partition
        from .six_bar import _generate_partitions
        return _generate_partitions(n_points, n_groups=2,
                                    min_per_group=min_per_group,
                                    max_per_group=max_per_group)

    # For 3+ groups, use recursive approach with pruning
    all_indices = list(range(n_points))
    results: list[PointPartition] = []
    max_partitions = 50  # Cap to prevent explosion

    def _recurse(
        remaining: list[int],
        current: list[tuple[int, ...]],
        group_idx: int,
    ) -> None:
        if len(results) >= max_partitions:
            return

        if group_idx == n_groups:
            if not remaining:
                results.append(tuple(current))
            return

        groups_left = n_groups - group_idx
        min_needed = min_per_group * (groups_left - 1)
        max_available = len(remaining)

        lo = max(min_per_group, max_available - max_per_group * (groups_left - 1))
        hi = min(max_per_group, max_available - min_needed)

        for k in range(lo, hi + 1):
            if len(results) >= max_partitions:
                return
            for chosen in combinations(remaining, k):
                if len(results) >= max_partitions:
                    return
                rest = [x for x in remaining if x not in chosen]
                _recurse(rest, current + [chosen], group_idx + 1)

    _recurse(all_indices, [], 0)
    return results


def _synthesize_chain(
    decomposition: DecompositionResult,
    topology: CatalogEntry,
    precision_points: list[PrecisionPoint],
    partition: PointPartition,
    orientations: list[float],
    max_candidates_per_group: int = 3,
) -> list[NBarSolution]:
    """Synthesize all groups in decomposition order using a given partition.

    Iterates over Assur groups. For each group:
    1. Get anchor positions (from previously solved groups or ground)
    2. Get assigned precision points
    3. Synthesize via Burmester (dyads) or optimization (triads)
    4. Propagate results to next group

    Args:
        decomposition: Assur decomposition with groups in order.
        topology: Catalog entry for metadata.
        precision_points: All precision points.
        partition: Point assignment to groups.
        orientations: Coupler orientations (for first group).
        max_candidates_per_group: Max solutions to carry per group.

    Returns:
        List of complete NBarSolution objects.
    """

    # Track partial solutions: list of (joint_positions, group_results, driving_fourbar)
    partial: list[tuple[dict[str, Point2D], list[GroupSynthesisResult], object]] = [
        ({}, [], None)
    ]

    for group_idx, group in enumerate(decomposition.groups):
        if group_idx >= len(partition):
            break

        assigned_indices = partition[group_idx]
        assigned_points = [precision_points[i] for i in assigned_indices]

        next_partial: list[tuple[dict[str, Point2D], list[GroupSynthesisResult], object]] = []
        for positions, group_results, prev_fourbar in partial:
            if len(next_partial) >= max_candidates_per_group * 3:
                break

            # Synthesize this group
            if group.group_class == 1:
                # Dyad: use Burmester
                results = _synthesize_dyad_group(
                    group_idx, group, assigned_points,
                    positions, orientations, prev_fourbar,
                    max_pairs=max_candidates_per_group,
                )
            else:
                # Triad or higher: use optimization
                results = _synthesize_triad_group(
                    group_idx, group, assigned_points,
                    positions, max_solutions=max_candidates_per_group,
                )

            for gsr, new_positions, fourbar in results:
                merged_pos = {**positions, **new_positions}
                next_partial.append((
                    merged_pos,
                    group_results + [gsr],
                    fourbar,
                ))

        partial = next_partial[:max_candidates_per_group * 2]

    # Convert partial solutions to NBarSolution
    solutions: list[NBarSolution] = []
    for positions, group_results, _ in partial:
        # Compute link lengths from positions
        link_lengths = _compute_link_lengths(positions, topology)

        nbar = NBarSolution(
            topology_id=topology.id,
            joint_positions=positions,
            link_lengths=link_lengths,
            group_results=group_results,
            coupler_point=precision_points[0] if precision_points else None,
        )
        solutions.append(nbar)

    return solutions


def _synthesize_dyad_group(
    group_index: int,
    group: AssurGroup,
    assigned_points: list[PrecisionPoint],
    known_positions: dict[str, Point2D],
    orientations: list[float],
    prev_fourbar: object,
    max_pairs: int = 3,
) -> list[tuple[GroupSynthesisResult, dict[str, Point2D], object]]:
    """Synthesize a single dyad group using Burmester theory.

    Returns list of (GroupSynthesisResult, new_positions, fourbar) tuples.
    """
    if len(assigned_points) < 3:
        return []

    # Construct poses
    poses = _points_to_poses(assigned_points, orientations[:len(assigned_points)])
    poses = poses[:5]

    try:
        curves = compute_circle_point_curve(poses)
    except (ValueError, Exception):
        return []
    if not curves:
        return []

    dyad_pairs = select_compatible_dyads(curves, max_pairs=max_pairs * 3)
    results: list[tuple[GroupSynthesisResult, dict[str, Point2D], object]] = []

    for dyad_left, dyad_right in dyad_pairs:
        fb = _dyads_to_fourbar(dyad_left, dyad_right, coupler_point_world=assigned_points[0])
        if fb is None:
            continue

        filtered, _ = _filter_solutions([fb], require_grashof=False)
        if not filtered:
            continue

        fb = filtered[0]
        internal_id = group.internal_nodes[0] if group.internal_nodes else f"node_{group_index}"

        # Map dyad positions to internal node
        new_positions: dict[str, Point2D] = {
            internal_id: fb.coupler_pivot_c,
        }
        # Also record anchor positions
        for i, anchor_id in enumerate(group.anchor_nodes):
            if anchor_id not in known_positions:
                if i == 0:
                    new_positions[anchor_id] = fb.ground_pivot_a
                else:
                    new_positions[anchor_id] = fb.ground_pivot_d

        gsr = GroupSynthesisResult(
            group_index=group_index,
            group_signature=group.joint_signature,
            dyads=(dyad_left, dyad_right),
            joint_positions=new_positions,
            precision_indices=(),
        )
        results.append((gsr, new_positions, fb))

        if len(results) >= max_pairs:
            break

    return results


def _synthesize_triad_group(
    group_index: int,
    group: AssurGroup,
    assigned_points: list[PrecisionPoint],
    known_positions: dict[str, Point2D],
    max_solutions: int = 3,
) -> list[tuple[GroupSynthesisResult, dict[str, Point2D], object]]:
    """Synthesize a triad group via optimization.

    Uses scipy least_squares to find internal node positions that
    minimize path error at the assigned precision points.
    """
    import numpy as np
    from scipy.optimize import least_squares as scipy_ls

    if len(assigned_points) < 2:
        return []

    # Initial guess: centroid of assigned points
    cx = sum(p[0] for p in assigned_points) / len(assigned_points)
    cy = sum(p[1] for p in assigned_points) / len(assigned_points)

    n_internal = len(group.internal_nodes)
    # Initial positions spread around centroid
    x0 = np.array([
        cx + 0.5 * (i - n_internal / 2)
        for i in range(n_internal)
        for _ in (0, 1)  # x, y pairs
    ])
    # Adjust y values
    for i in range(n_internal):
        x0[2 * i + 1] = cy + 0.5 * (i - n_internal / 2)

    def residuals(x: np.ndarray) -> np.ndarray:
        # Place internal nodes at positions from x
        errs = []
        for pt in assigned_points:
            # Distance from closest internal node to target
            min_dist_sq = float("inf")
            for i in range(n_internal):
                dx = x[2 * i] - pt[0]
                dy = x[2 * i + 1] - pt[1]
                min_dist_sq = min(min_dist_sq, dx * dx + dy * dy)
            errs.append(math.sqrt(min_dist_sq))
        return np.array(errs)

    try:
        result = scipy_ls(residuals, x0, max_nfev=100)
    except Exception:
        return []

    new_positions: dict[str, Point2D] = {}
    for i, node_id in enumerate(group.internal_nodes):
        new_positions[node_id] = (float(result.x[2 * i]), float(result.x[2 * i + 1]))

    gsr = GroupSynthesisResult(
        group_index=group_index,
        group_signature=group.joint_signature,
        joint_positions=new_positions,
        residual=float(result.cost),
    )

    return [(gsr, new_positions, None)]


def _compute_link_lengths(
    positions: dict[str, Point2D],
    topology: CatalogEntry,
) -> dict[str, float]:
    """Compute link lengths from joint positions and topology edges."""
    graph = topology.to_graph()
    lengths: dict[str, float] = {}

    for edge in graph.edges.values():
        if edge.source in positions and edge.target in positions:
            p1 = positions[edge.source]
            p2 = positions[edge.target]
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            lengths[edge.id] = dist

    return lengths


def _synthesize_fourbar_as_nbar(
    precision_points: list[PrecisionPoint],
    max_solutions: int = 10,
    n_orientation_samples: int = 36,
) -> list[NBarSolution]:
    """Wrap existing four-bar path_generation as NBarSolution."""
    from .path_generation import path_generation

    # Use a higher internal search limit — path_generation's max_solutions
    # controls the search breadth, not just the output count.
    result = path_generation(
        precision_points,
        max_solutions=max(max_solutions, 10),
        n_orientation_samples=n_orientation_samples,
    )

    nbar_solutions: list[NBarSolution] = []
    for _linkage, raw in zip(result.solutions, result.raw_solutions, strict=True):
        nbar = NBarSolution(
            topology_id="four-bar",
            joint_positions={
                "A": raw.ground_pivot_a,
                "D": raw.ground_pivot_d,
                "B": raw.crank_pivot_b,
                "C": raw.coupler_pivot_c,
            },
            link_lengths={
                "crank_AB": raw.crank_length,
                "coupler_BC": raw.coupler_length,
                "rocker_DC": raw.rocker_length,
                "ground_AD": raw.ground_length,
            },
            coupler_point=raw.coupler_point,
        )
        nbar_solutions.append(nbar)

    return nbar_solutions


def _synthesize_sixbar_as_nbar(
    precision_points: list[PrecisionPoint],
    topology_id: str,
    max_solutions: int = 10,
    n_orientation_samples: int = 24,
) -> list[NBarSolution]:
    """Wrap six-bar synthesis as NBarSolution list."""
    from .six_bar import six_bar_path_generation

    topology_name = "watt" if topology_id == "watt" else "stephenson"
    result = six_bar_path_generation(
        precision_points,
        topology=topology_name,
        max_solutions=max_solutions,
        n_orientation_samples=n_orientation_samples,
    )

    # Extract NBarSolutions from the linkages
    # The six_bar module already creates proper linkages
    nbar_solutions: list[NBarSolution] = []
    for linkage in result.solutions:
        positions: dict[str, Point2D] = {}
        lengths: dict[str, float] = {}

        from .._compat import get_parts

        for joint in get_parts(linkage):
            name = getattr(joint, "name", None)
            if name and joint.x is not None and joint.y is not None:
                positions[name] = (joint.x, joint.y)

        nbar = NBarSolution(
            topology_id=topology_id,
            joint_positions=positions,
            link_lengths=lengths,
            coupler_point=precision_points[0] if precision_points else None,
        )
        nbar_solutions.append(nbar)

    return nbar_solutions
