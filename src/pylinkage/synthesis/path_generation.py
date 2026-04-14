"""Path generation synthesis for four-bar linkages.

Path generation synthesizes a linkage where a point on the coupler
(the coupler point) traces a curve passing through specified precision
points. This is used for:

- Walking mechanisms (leg tip trajectories)
- Pick-and-place motions
- General path-following applications

The coupler curve of a four-bar is a tricircular sextic (degree 6 algebraic
curve), which can produce a wide variety of shapes including loops,
cusps, and approximate straight lines.

Path generation is more complex than function generation because the
coupler orientation at each precision point is not specified (free
variables). We must search over possible orientations.

For exact synthesis:
- 4 precision points with prescribed timing: Exact solution possible
- 4-5 points without timing: Search over orientation space
- 6+ points: Over-constrained, typically no exact solution

References:
    - Wampler et al., "Complete Solution of the Nine-Point Path Synthesis"
    - McCarthy, Chapter 7: "Path Synthesis"
    - Sandor & Erdman, Chapter 5: "Path, Motion, and Function Generation"
"""

from __future__ import annotations

import math
from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg as scipy_linalg

from ._types import FourBarSolution, Point2D, Pose, PrecisionPoint, SynthesisType
from .burmester import (
    complex_to_point,
    compute_circle_point_curve,
    select_compatible_dyads,
)
from .core import Dyad, SynthesisProblem, SynthesisResult
from .utils import GrashofType, grashof_check, validate_fourbar

if TYPE_CHECKING:
    from ..simulation import Linkage


def _estimate_orientations_from_path(
    points: list[PrecisionPoint],
) -> list[float]:
    """Estimate coupler orientations from path direction.

    Uses the direction between consecutive points as an initial
    estimate for coupler orientation at each point.

    Args:
        points: List of precision points.

    Returns:
        List of orientation angles in radians.
    """
    n = len(points)
    orientations: list[float] = []

    for i in range(n):
        if i < n - 1:
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
        else:
            # Last point: use direction from previous
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]

        angle = math.atan2(dy, dx)
        orientations.append(angle)

    return orientations


def _generate_orientation_candidates(
    points: list[PrecisionPoint],
    n_samples: int = 36,
    perturbation_range: float = math.pi / 3,
) -> Generator[list[float], None, None]:
    """Generate candidate orientation sequences for precision points.

    For path generation without prescribed timing, the coupler
    orientation at each point is a free variable. This generator
    systematically samples the orientation space.

    The search fixes the first orientation (a reference frame choice)
    and varies the remaining orientations independently, since each
    precision point can have a completely different coupler angle.

    Args:
        points: List of precision points.
        n_samples: Number of samples per point.
        perturbation_range: Range of perturbation around base estimate.

    Yields:
        Lists of orientation angles (one per point).
    """
    n = len(points)
    base_orientations = _estimate_orientations_from_path(points)

    # First yield the base estimate
    yield base_orientations

    # --- Independent orientation search ---
    # Fix first orientation (reference) and search the rest over
    # the full circle.  For 3 points this is a 2-D grid; for more
    # points the grid grows, so we use a coarser per-axis resolution.
    free = n - 1  # number of free orientations (first is fixed)
    if free > 0:
        per_axis = max(6, int(round(n_samples ** (1.0 / free))))
        angle_grid = np.linspace(-math.pi, math.pi, per_axis, endpoint=False)

        if free == 1:
            for a in angle_grid:
                yield [base_orientations[0], a]
        elif free == 2:
            for a in angle_grid:
                for b in angle_grid:
                    yield [base_orientations[0], a, b]
        else:
            # For 4+ points, combine a coarse independent grid with
            # random samples to keep the count manageable.
            from itertools import product as _product

            for count, combo in enumerate(_product(angle_grid, repeat=free)):
                yield [base_orientations[0], *combo]
                if count + 1 >= n_samples * n_samples:
                    break

    # Uniform perturbations (original strategy, still useful)
    deltas = np.linspace(
        -perturbation_range, perturbation_range, max(6, n_samples // 3), endpoint=True
    )
    for delta in deltas:
        if abs(delta) < 1e-10:
            continue
        yield [o + delta for o in base_orientations]

    # Progressive rotation
    for total_rotation in [math.pi / 4, math.pi / 2, math.pi, -math.pi / 4, -math.pi / 2]:
        yield [base_orientations[i] + total_rotation * i / max(n - 1, 1) for i in range(n)]


def _points_to_poses(
    points: list[PrecisionPoint],
    orientations: list[float],
) -> list[Pose]:
    """Convert precision points and orientations to Pose objects.

    Args:
        points: List of (x, y) points.
        orientations: List of orientation angles.

    Returns:
        List of Pose objects.
    """
    return [Pose(x=p[0], y=p[1], angle=o) for p, o in zip(points, orientations, strict=True)]


def _dyads_to_fourbar(
    dyad_left: Dyad,
    dyad_right: Dyad,
    coupler_point_world: Point2D | None = None,
) -> FourBarSolution | None:
    """Convert two dyads to a four-bar solution.

    The left dyad connects ground pivot A to coupler point B.
    The right dyad connects ground pivot D to coupler point C.

    Args:
        dyad_left: Dyad for the crank side (A-B).
        dyad_right: Dyad for the rocker side (D-C).
        coupler_point_world: World-frame position of the traced coupler
            point at the first precision position. This is the point that
            should pass through the precision points.

    Returns:
        FourBarSolution if valid, None if degenerate.
    """
    # Ground pivots (center points)
    A = complex_to_point(dyad_left.center_point)
    D = complex_to_point(dyad_right.center_point)

    # Coupler attachment points (circle points)
    B = complex_to_point(dyad_left.circle_point)
    C = complex_to_point(dyad_right.circle_point)

    # Compute link lengths
    crank = abs(dyad_left.circle_point - dyad_left.center_point)
    rocker = abs(dyad_right.circle_point - dyad_right.center_point)
    coupler = abs(dyad_left.circle_point - dyad_right.circle_point)
    ground = abs(dyad_left.center_point - dyad_right.center_point)

    # Check for degenerate cases
    min_length = 1e-6
    if any(link < min_length for link in [crank, coupler, rocker, ground]):
        return None

    return FourBarSolution(
        ground_pivot_a=A,
        ground_pivot_d=D,
        crank_pivot_b=B,
        coupler_pivot_c=C,
        crank_length=crank,
        coupler_length=coupler,
        rocker_length=rocker,
        ground_length=ground,
        coupler_point=coupler_point_world,
    )


def _filter_solutions(
    solutions: list[FourBarSolution],
    require_grashof: bool = True,
    require_crank_rocker: bool = False,
    min_link_ratio: float = 0.1,
    max_link_ratio: float = 10.0,
) -> tuple[list[FourBarSolution], list[str]]:
    """Filter solutions based on kinematic criteria.

    Args:
        solutions: List of candidate solutions.
        require_grashof: Reject non-Grashof linkages.
        require_crank_rocker: Only accept crank-rocker type.
        min_link_ratio: Minimum ratio between any two links.
        max_link_ratio: Maximum ratio between any two links.

    Returns:
        Tuple of (valid_solutions, warning_messages).
    """
    valid: list[FourBarSolution] = []
    warnings: list[str] = []

    for sol in solutions:
        # Basic validation
        is_valid, errors = validate_fourbar(sol)
        if not is_valid:
            continue

        # Link ratio check
        lengths = [
            sol.crank_length,
            sol.coupler_length,
            sol.rocker_length,
            sol.ground_length,
        ]
        min_len = min(lengths)
        max_len = max(lengths)

        if max_len / min_len > max_link_ratio:
            continue
        if min_len / max_len < min_link_ratio:
            continue

        # Grashof check
        grashof_type = grashof_check(
            sol.crank_length,
            sol.coupler_length,
            sol.rocker_length,
            sol.ground_length,
        )

        if require_crank_rocker and grashof_type not in (
            GrashofType.GRASHOF_CRANK_ROCKER,
            GrashofType.CHANGE_POINT,
        ):
            continue

        if require_grashof and grashof_type == GrashofType.NON_GRASHOF:
            continue

        valid.append(sol)

    if len(solutions) > 0 and len(valid) == 0:
        warnings.append(
            f"All {len(solutions)} candidate solutions were filtered out. "
            "Try relaxing constraints (require_grashof=False)."
        )

    return valid, warnings


def _verify_coupler_path(
    solution: FourBarSolution,
    precision_points: list[PrecisionPoint],
    iterations: int = 300,
    tolerance: float | None = None,
) -> bool:
    """Check that a solution's coupler point traces through precision points.

    Builds a temporary linkage, simulates one full rotation, and verifies
    that each precision point has a nearby point on the coupler trajectory.

    Args:
        solution: Candidate four-bar solution.
        precision_points: Target points the coupler must pass through.
        iterations: Number of simulation steps.
        tolerance: Maximum acceptable distance.  Defaults to 5 % of the
            bounding-box diagonal of the precision points.

    Returns:
        True if the coupler passes near all precision points.
    """
    from .conversion import solution_to_linkage

    if not precision_points or solution.coupler_point is None:
        return True

    # Compute tolerance from precision point spread
    if tolerance is None:
        xs = [p[0] for p in precision_points]
        ys = [p[1] for p in precision_points]
        diag = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
        tolerance = max(0.05 * diag, 0.01)

    try:
        lk = solution_to_linkage(solution, iterations=iterations)
    except Exception:
        return False

    # Simulate and collect coupler point trajectory
    try:
        trajectory: list[tuple[float, float]] = []
        for positions in lk.step(iterations=iterations):
            px, py = positions[-1]  # P is the last joint
            if px is not None and py is not None:
                trajectory.append((px, py))
    except Exception:
        return False

    if not trajectory:
        return False

    traj_arr = np.asarray(trajectory)
    for px, py in precision_points:
        dists = np.hypot(traj_arr[:, 0] - px, traj_arr[:, 1] - py)
        if dists.min() > tolerance:
            return False

    return True


def _remove_duplicate_solutions(
    solutions: list[FourBarSolution],
    tolerance: float = 0.01,
) -> list[FourBarSolution]:
    """Remove duplicate solutions within tolerance.

    Args:
        solutions: List of solutions.
        tolerance: Relative tolerance for considering solutions equal.

    Returns:
        List with duplicates removed.
    """
    if not solutions:
        return []

    unique: list[FourBarSolution] = [solutions[0]]

    for sol in solutions[1:]:
        is_duplicate = False

        for usol in unique:
            # Compare link lengths
            diff = abs(sol.crank_length - usol.crank_length)
            diff += abs(sol.coupler_length - usol.coupler_length)
            diff += abs(sol.rocker_length - usol.rocker_length)
            diff += abs(sol.ground_length - usol.ground_length)

            scale = sol.crank_length + sol.coupler_length + sol.rocker_length + sol.ground_length

            if diff / scale < tolerance:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(sol)

    return unique


def path_generation(
    precision_points: list[PrecisionPoint],
    coupler_point_offset: tuple[float, float] = (0.0, 0.0),
    ground_pivot_a: Point2D | None = None,
    ground_pivot_d: Point2D | None = None,
    n_orientation_samples: int = 36,
    max_solutions: int | None = 10,
    require_grashof: bool = True,
    require_crank_rocker: bool = False,
) -> SynthesisResult:
    """Synthesize a four-bar for path generation.

    Find four-bar linkages where a coupler point passes through
    the specified precision points.

    For path generation without prescribed timing, the coupler
    orientation at each point is a free variable. This function
    searches over orientation candidates using Burmester theory.

    Args:
        precision_points: List of (x, y) points the coupler should pass through.
            Best results with 3-5 points.
        coupler_point_offset: Offset of traced point from coupler reference.
        ground_pivot_a: Optional fixed position for left ground pivot.
        ground_pivot_d: Optional fixed position for right ground pivot.
        n_orientation_samples: Number of orientation samples to try.
        max_solutions: Maximum number of solutions to return (None for all).
        require_grashof: If True, reject non-Grashof solutions.
        require_crank_rocker: If True, only accept crank-rocker type.

    Returns:
        SynthesisResult containing valid linkages.

    Example:
        >>> points = [(0, 1), (1, 2), (2, 1.5), (3, 0)]
        >>> result = path_generation(points)
        >>> print(f"Found {len(result.solutions)} solutions")
        >>> for linkage in result.solutions:
        ...     linkage.show()
    """
    problem = SynthesisProblem(
        synthesis_type=SynthesisType.PATH,
        precision_points=list(precision_points),
        ground_pivot_a=ground_pivot_a,
        ground_pivot_d=ground_pivot_d,
    )

    warnings: list[str] = []
    raw_solutions: list[FourBarSolution] = []

    n_points = len(precision_points)

    if n_points < 3:
        warnings.append(f"Path generation needs at least 3 points, got {n_points}")
        return SynthesisResult(
            solutions=[],
            raw_solutions=[],
            problem=problem,
            warnings=warnings,
        )

    if n_points > 5:
        warnings.append(
            f"Path generation with {n_points} points is over-constrained. "
            "Exact solutions may not exist; using best-fit approach."
        )

    # Ground constraint for dyad selection
    ground_constraint = None
    ground_tolerance = 0.1
    if ground_pivot_a is not None and ground_pivot_d is not None:
        ground_constraint = (ground_pivot_a, ground_pivot_d)

    # Search over orientation candidates
    orientation_gen = _generate_orientation_candidates(
        precision_points, n_samples=n_orientation_samples
    )

    # The coupler point is the first precision point — it's the
    # body reference that should trace through all precision points.
    coupler_pt: Point2D = precision_points[0]

    # Verified solutions are collected inline so that the search keeps
    # going until we have enough *confirmed* results (not just raw
    # candidates). Each orientation+dyad pair is verified immediately;
    # bad assembly modes are discarded without consuming the budget.
    n_assembly_rejected = 0
    _seen_lengths: set[tuple[float, ...]] = set()

    for orientations in orientation_gen:
        poses = _points_to_poses(precision_points, orientations)

        try:
            poses_for_burmester = poses[: min(5, len(poses))]
            curves = compute_circle_point_curve(poses_for_burmester)
            if not curves:
                continue

            dyad_pairs = select_compatible_dyads(
                curves,
                ground_constraint=ground_constraint,
                ground_tolerance=ground_tolerance,
                max_pairs=20,
            )

            for dyad_left, dyad_right in dyad_pairs:
                solution = _dyads_to_fourbar(
                    dyad_left,
                    dyad_right,
                    coupler_point_world=coupler_pt,
                )
                if solution is None:
                    continue

                # Quick dedup by rounded link lengths
                key = (
                    round(solution.crank_length, 2),
                    round(solution.coupler_length, 2),
                    round(solution.rocker_length, 2),
                    round(solution.ground_length, 2),
                )
                if key in _seen_lengths:
                    continue
                _seen_lengths.add(key)

                # Link-ratio filter: reject degenerate mechanisms
                # where one link is much shorter/longer than the rest
                _lengths = [
                    solution.crank_length,
                    solution.coupler_length,
                    solution.rocker_length,
                    solution.ground_length,
                ]
                if max(_lengths) / min(_lengths) > 10.0:
                    continue

                # Lightweight Grashof filter (no simulation)
                grashof_type = grashof_check(
                    solution.crank_length,
                    solution.coupler_length,
                    solution.rocker_length,
                    solution.ground_length,
                )
                if require_grashof and grashof_type == GrashofType.NON_GRASHOF:
                    continue
                if require_crank_rocker and grashof_type not in (
                    GrashofType.GRASHOF_CRANK_ROCKER,
                    GrashofType.CHANGE_POINT,
                ):
                    continue

                # Verify coupler passes through precision points
                if not _verify_coupler_path(solution, precision_points):
                    n_assembly_rejected += 1
                    continue

                raw_solutions.append(solution)

                if max_solutions and len(raw_solutions) >= max_solutions:
                    break

        except (ValueError, np.linalg.LinAlgError, scipy_linalg.LinAlgError):
            continue

        if max_solutions and len(raw_solutions) >= max_solutions:
            break

    if n_assembly_rejected > 0:
        warnings.append(
            f"{n_assembly_rejected} candidate(s) rejected: coupler point did "
            "not pass through all precision points (assembly-mode mismatch)."
        )

    # Limit to max_solutions
    if max_solutions and len(raw_solutions) > max_solutions:
        raw_solutions = raw_solutions[:max_solutions]

    # Convert to Linkage objects
    from .conversion import solutions_to_linkages

    linkages = solutions_to_linkages(raw_solutions, SynthesisType.PATH)

    if not linkages:
        warnings.append(
            "No valid solutions found. Try:\n"
            "  - Using fewer precision points (3-4 recommended)\n"
            "  - Adjusting point locations\n"
            "  - Setting require_grashof=False"
        )

    return SynthesisResult(
        solutions=linkages,
        raw_solutions=raw_solutions,
        problem=problem,
        warnings=warnings,
    )


def verify_path_generation(
    linkage: Linkage,
    precision_points: list[PrecisionPoint],
    tolerance: float | None = None,
) -> tuple[bool, list[float]]:
    """Verify that a synthesized linkage's coupler point passes near target points.

    Simulates the linkage for one full crank rotation and finds, for each
    precision point, the minimum distance from the coupler point trajectory
    to that target point.

    The linkage must have a joint named "P" (the coupler point tracker
    added by ``solution_to_linkage``). If no such joint exists, the last
    joint in the solve order is used as the coupler point.

    Args:
        linkage: The synthesized Linkage to verify.
        precision_points: Target (x, y) points the coupler should pass through.
        tolerance: Maximum acceptable distance for each point. If None,
            defaults to 5% of the bounding box diagonal of the precision points.

    Returns:
        Tuple of (all_satisfied, distances) where distances[i] is the
        minimum distance from the coupler trajectory to precision_points[i].
    """
    from ..exceptions import UnbuildableError

    if not precision_points:
        return True, []

    from .._compat import get_parts

    parts = get_parts(linkage)

    # Find the coupler point joint
    coupler_joint = None
    for joint in parts:
        if getattr(joint, "name", None) == "P":
            coupler_joint = joint
            break

    if coupler_joint is None:
        # Fallback: use last joint in order (typically joint C)
        coupler_joint = parts[-1]

    coupler_idx = list(parts).index(coupler_joint)

    # Compute auto-tolerance from precision point spread
    if tolerance is None:
        xs = [p[0] for p in precision_points]
        ys = [p[1] for p in precision_points]
        diag = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
        tolerance = max(0.05 * diag, 0.01)

    # Simulate one full rotation
    try:
        trajectory: list[tuple[float, float]] = []
        for positions in linkage.step():
            px, py = positions[coupler_idx]
            if px is not None and py is not None:
                trajectory.append((px, py))
    except UnbuildableError:
        # Linkage locks up — return worst-case
        return False, [float("inf")] * len(precision_points)

    if not trajectory:
        return False, [float("inf")] * len(precision_points)

    # For each precision point, find the minimum distance to trajectory
    distances: list[float] = []
    for px, py in precision_points:
        min_dist = min(math.sqrt((tx - px) ** 2 + (ty - py) ** 2) for tx, ty in trajectory)
        distances.append(min_dist)

    all_ok = all(d <= tolerance for d in distances)
    return all_ok, distances


def path_generation_with_timing(
    precision_points: list[PrecisionPoint],
    crank_angles: list[float],
    ground_pivot_a: Point2D = (0.0, 0.0),
    ground_length: float | None = None,
    require_grashof: bool = True,
) -> SynthesisResult:
    """Synthesize a four-bar with prescribed timing.

    Path generation with prescribed timing specifies both the
    coupler points AND the crank angles at which they should be reached.
    This is more constrained than general path generation.

    Args:
        precision_points: List of (x, y) coupler points.
        crank_angles: Crank angles (radians) for each point.
        ground_pivot_a: Position of crank ground pivot.
        ground_length: Fixed ground link length (optional).
        require_grashof: If True, reject non-Grashof solutions.

    Returns:
        SynthesisResult containing valid linkages.

    Note:
        This is a more specialized synthesis that combines elements
        of path and function generation. With 4 points and 4 angles,
        exact synthesis is possible.
    """
    # For prescribed timing, coupler orientations are determined
    # by the crank angles, reducing the degrees of freedom

    problem = SynthesisProblem(
        synthesis_type=SynthesisType.PATH,
        precision_points=list(precision_points),
        ground_pivot_a=ground_pivot_a,
        ground_length=ground_length,
    )

    warnings: list[str] = []

    if len(precision_points) != len(crank_angles):
        warnings.append(
            f"Number of points ({len(precision_points)}) must equal "
            f"number of angles ({len(crank_angles)})"
        )
        return SynthesisResult(
            solutions=[],
            raw_solutions=[],
            problem=problem,
            warnings=warnings,
        )

    # Convert crank angles to coupler orientations
    # The orientation change equals the crank angle change for a simple coupler
    base_angle = crank_angles[0]
    orientations = [ca - base_angle for ca in crank_angles]

    poses = _points_to_poses(precision_points, orientations)

    raw_solutions: list[FourBarSolution] = []

    try:
        curves = compute_circle_point_curve(poses[: min(5, len(poses))])

        if curves:
            # Limit pairs to avoid excessive computation
            dyad_pairs = select_compatible_dyads(curves, max_pairs=30)

            coupler_pt: Point2D = precision_points[0]
            for dyad_left, dyad_right in dyad_pairs:
                solution = _dyads_to_fourbar(dyad_left, dyad_right, coupler_point_world=coupler_pt)
                if solution is not None:
                    raw_solutions.append(solution)

    except (ValueError, np.linalg.LinAlgError, scipy_linalg.LinAlgError) as e:
        warnings.append(f"Synthesis failed: {e}")

    # Filter
    raw_solutions, filter_warnings = _filter_solutions(
        raw_solutions, require_grashof=require_grashof
    )
    warnings.extend(filter_warnings)

    from .conversion import solutions_to_linkages

    linkages = solutions_to_linkages(raw_solutions, SynthesisType.PATH)

    return SynthesisResult(
        solutions=linkages,
        raw_solutions=raw_solutions,
        problem=problem,
        warnings=warnings,
    )
