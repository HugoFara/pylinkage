"""Six-bar path generation synthesis.

Decomposes six-bar topologies (Watt, Stephenson) into Assur groups
and synthesizes each group sequentially using Burmester theory.

A Watt six-bar decomposes into two RRR dyads stacked in series.
The algorithm:

1. Synthesize the driving four-bar (first dyad pair) via Burmester
   on a subset of precision points.
2. Simulate the driving four-bar to compute intermediate coupler
   positions at the remaining precision points.
3. Synthesize the second dyad using those intermediate positions.
4. Combine and validate the complete six-bar assembly.

References:
    - Plecnik & McCarthy, "Kinematic synthesis of Stephenson III
      six-bar function generators" (2016)
    - McCarthy & Soh, "Geometric Design of Linkages" (2nd ed., 2011)
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import TYPE_CHECKING

from ._types import FourBarSolution, Point2D, Pose, PrecisionPoint, SynthesisType
from .burmester import (
    compute_circle_point_curve,
    select_compatible_dyads,
)
from .conversion import solution_to_linkage
from .core import SynthesisProblem, SynthesisResult
from .path_generation import (
    _dyads_to_fourbar,
    _filter_solutions,
    _generate_orientation_candidates,
    _points_to_poses,
    _remove_duplicate_solutions,
)
from .topology_types import GroupSynthesisResult, NBarSolution

if TYPE_CHECKING:
    from ..linkage import Linkage
    from ..simulation import Linkage as SimLinkage


def six_bar_path_generation(
    precision_points: list[PrecisionPoint],
    topology: str = "watt",
    max_solutions: int = 10,
    require_grashof_driver: bool = True,
    n_orientation_samples: int = 24,
) -> SynthesisResult:
    """Synthesize a six-bar linkage for path generation.

    Uses decomposition-based synthesis: the six-bar is split into
    Assur groups, and each group is synthesized sequentially using
    Burmester theory.

    For Watt type (2 stacked RRR dyads):
      - First dyad pair: driving four-bar synthesized on a subset
        of precision points.
      - Second dyad pair: refines path using remaining points, with
        the first dyad's coupler providing anchor positions.

    Args:
        precision_points: Target (x, y) points. 4-7 recommended
            for six-bars (more points than four-bar can handle).
        topology: ``"watt"`` or ``"stephenson"``.
        max_solutions: Maximum solutions to return.
        require_grashof_driver: Require Grashof criterion on the
            driving four-bar loop.
        n_orientation_samples: Orientation search density per stage.

    Returns:
        SynthesisResult containing valid six-bar Linkage objects.

    Example:
        >>> points = [(0, 0), (1, 2), (2, 3), (3, 2), (4, 0)]
        >>> result = six_bar_path_generation(points, topology="watt")
        >>> print(f"Found {len(result.solutions)} solutions")
    """
    if topology not in ("watt", "stephenson"):
        raise ValueError(f"Unknown six-bar topology: {topology!r}. Use 'watt' or 'stephenson'.")

    n_points = len(precision_points)
    warnings: list[str] = []

    if n_points < 4:
        warnings.append(
            f"Six-bar synthesis works best with 4-7 precision points, got {n_points}. "
            "Consider using four-bar path_generation() instead."
        )
    if n_points > 7:
        warnings.append(
            f"Got {n_points} precision points. Six-bar synthesis is over-constrained "
            "beyond ~7 points; results may have reduced accuracy."
        )

    if topology == "stephenson":
        warnings.append(
            "Stephenson six-bar synthesis uses optimization-based approach "
            "and may be slower than Watt synthesis."
        )
        return _stephenson_synthesis(
            precision_points,
            max_solutions=max_solutions,
            n_orientation_samples=n_orientation_samples,
            warnings=warnings,
        )

    # Watt six-bar: decomposition-based (2 dyad stages)
    return _watt_synthesis(
        precision_points,
        max_solutions=max_solutions,
        require_grashof_driver=require_grashof_driver,
        n_orientation_samples=n_orientation_samples,
        warnings=warnings,
    )


def _watt_synthesis(
    precision_points: list[PrecisionPoint],
    max_solutions: int = 10,
    require_grashof_driver: bool = True,
    n_orientation_samples: int = 24,
    warnings: list[str] | None = None,
) -> SynthesisResult:
    """Watt six-bar synthesis via two-stage Burmester decomposition."""
    if warnings is None:
        warnings = []

    n_points = len(precision_points)
    partitions = _generate_partitions(n_points, n_groups=2, min_per_group=3, max_per_group=5)

    if not partitions:
        warnings.append(
            f"Cannot partition {n_points} points into 2 groups of 3-5. "
            "Need at least 6 points for six-bar, or use four-bar for fewer."
        )
        return SynthesisResult(
            solutions=[],
            raw_solutions=[],
            problem=SynthesisProblem(
                synthesis_type=SynthesisType.PATH,
                precision_points=list(precision_points),
            ),
            warnings=warnings,
        )

    all_nbar_solutions: list[NBarSolution] = []
    all_raw_fourbars: list[FourBarSolution] = []

    for partition in partitions:
        stage1_indices, stage2_indices = partition
        stage1_points = [precision_points[i] for i in stage1_indices]
        stage2_points = [precision_points[i] for i in stage2_indices]

        # Try multiple orientation candidates for stage 1
        orientation_gen = _generate_orientation_candidates(
            stage1_points, n_samples=n_orientation_samples
        )

        orientations_tried = 0
        for orientations in orientation_gen:
            if len(all_nbar_solutions) >= max_solutions * 3:
                break
            orientations_tried += 1
            if orientations_tried > n_orientation_samples + 6:
                break

            # Stage 1: synthesize driving four-bar
            fourbars = _synthesize_driving_fourbar(
                stage1_points, orientations, require_grashof_driver
            )

            for fourbar in fourbars:
                if len(all_nbar_solutions) >= max_solutions * 3:
                    break

                # Simulate driving four-bar to find intermediate positions
                intermediate = _find_coupler_positions_at_targets(fourbar, stage2_points)
                if intermediate is None:
                    continue

                # Stage 2: synthesize second dyad
                second_stage = _synthesize_second_dyad(intermediate, stage2_points)

                for second_fourbar in second_stage:
                    nbar = _combine_to_six_bar(
                        fourbar,
                        second_fourbar,
                        precision_points=precision_points,
                        stage1_indices=stage1_indices,
                        stage2_indices=stage2_indices,
                    )
                    if nbar is not None:
                        all_nbar_solutions.append(nbar)
                        all_raw_fourbars.append(fourbar)

    # Convert to Linkage objects and validate
    solutions: list[Linkage] = []
    valid_raw: list[FourBarSolution] = []

    for nbar, raw_fb in zip(all_nbar_solutions, all_raw_fourbars, strict=True):
        linkage = _nbar_to_six_bar_linkage(nbar)
        if linkage is not None and _validate_six_bar(linkage, precision_points):
            solutions.append(linkage)  # type: ignore[arg-type]
            valid_raw.append(raw_fb)
            if len(solutions) >= max_solutions:
                break

    if not solutions and all_nbar_solutions:
        warnings.append(
            f"Generated {len(all_nbar_solutions)} candidate six-bars but "
            "none passed assembly validation. Try relaxing constraints."
        )

    return SynthesisResult(
        solutions=solutions,
        raw_solutions=valid_raw,
        problem=SynthesisProblem(
            synthesis_type=SynthesisType.PATH,
            precision_points=list(precision_points),
        ),
        warnings=warnings,
    )


def _stephenson_synthesis(
    precision_points: list[PrecisionPoint],
    max_solutions: int = 10,
    n_orientation_samples: int = 24,
    warnings: list[str] | None = None,
) -> SynthesisResult:
    """Stephenson six-bar synthesis via optimization.

    The Stephenson topology decomposes into a triad (not two dyads),
    so Burmester theory doesn't apply directly. We use an optimization
    approach: synthesize a driving four-bar on a subset of points, then
    optimize the triad link lengths to minimize path error at remaining
    points.
    """

    if warnings is None:
        warnings = []

    n_points = len(precision_points)
    partitions = _generate_partitions(n_points, n_groups=2, min_per_group=3, max_per_group=5)

    if not partitions:
        return SynthesisResult(
            solutions=[],
            raw_solutions=[],
            problem=SynthesisProblem(
                synthesis_type=SynthesisType.PATH,
                precision_points=list(precision_points),
            ),
            warnings=warnings,
        )

    all_solutions: list[Linkage] = []
    all_raw: list[FourBarSolution] = []

    for partition in partitions:
        stage1_indices, stage2_indices = partition
        stage1_points = [precision_points[i] for i in stage1_indices]
        stage2_points = [precision_points[i] for i in stage2_indices]

        for orientations in _generate_orientation_candidates(
            stage1_points, n_samples=min(n_orientation_samples, 12)
        ):
            if len(all_solutions) >= max_solutions:
                break

            fourbars = _synthesize_driving_fourbar(
                stage1_points, orientations, require_grashof=True
            )

            for fourbar in fourbars[:3]:
                if len(all_solutions) >= max_solutions:
                    break

                # Get intermediate positions from driving four-bar
                intermediate = _find_coupler_positions_at_targets(fourbar, stage2_points)
                if intermediate is None:
                    continue

                # Optimize triad link lengths
                linkage = _optimize_stephenson_triad(
                    fourbar, intermediate, stage2_points, precision_points
                )
                if linkage is not None:
                    all_solutions.append(linkage)
                    all_raw.append(fourbar)

            # Only try first few orientations for Stephenson
            break

    return SynthesisResult(
        solutions=all_solutions[:max_solutions],
        raw_solutions=all_raw[:max_solutions],
        problem=SynthesisProblem(
            synthesis_type=SynthesisType.PATH,
            precision_points=list(precision_points),
        ),
        warnings=warnings,
    )


def _generate_partitions(
    n_points: int,
    n_groups: int = 2,
    min_per_group: int = 3,
    max_per_group: int = 5,
) -> list[tuple[tuple[int, ...], ...]]:
    """Generate all valid partitions of point indices into groups.

    Each group gets between min_per_group and max_per_group indices.

    Args:
        n_points: Total number of precision points.
        n_groups: Number of groups (currently only 2 supported).
        min_per_group: Minimum points per group.
        max_per_group: Maximum points per group.

    Returns:
        List of partitions, each a tuple of index tuples.
    """
    if n_groups != 2:
        raise NotImplementedError("Only 2-group partitioning is implemented.")

    if n_points < 2 * min_per_group:
        return []

    all_indices = tuple(range(n_points))
    partitions: list[tuple[tuple[int, ...], ...]] = []

    # Enumerate all subsets of size k for group 1
    for k in range(min_per_group, min(max_per_group, n_points - min_per_group) + 1):
        remainder_size = n_points - k
        if remainder_size < min_per_group or remainder_size > max_per_group:
            continue

        for group1 in combinations(all_indices, k):
            group2 = tuple(i for i in all_indices if i not in group1)
            partitions.append((group1, group2))

    return partitions


def _synthesize_driving_fourbar(
    points: list[PrecisionPoint],
    orientations: list[float],
    require_grashof: bool = True,
    max_pairs: int = 5,
) -> list[FourBarSolution]:
    """Synthesize driving four-bar on a subset of precision points.

    Applies Burmester theory to get dyad pairs, converts to FourBarSolution,
    and filters.

    Args:
        points: Precision points for this stage.
        orientations: Coupler orientations at each point.
        require_grashof: Filter for Grashof criterion.
        max_pairs: Maximum dyad pairs to consider.

    Returns:
        List of valid FourBarSolution objects.
    """
    poses = _points_to_poses(points, orientations)

    # Use at most 5 poses for Burmester
    poses_for_burmester = poses[:5]

    try:
        curves = compute_circle_point_curve(poses_for_burmester)
    except (ValueError, Exception):
        return []
    if not curves:
        return []

    dyad_pairs = select_compatible_dyads(
        curves,
        max_pairs=max_pairs * 3,
    )

    fourbars: list[FourBarSolution] = []
    for dyad_left, dyad_right in dyad_pairs:
        fb = _dyads_to_fourbar(dyad_left, dyad_right, coupler_point_world=points[0])
        if fb is not None:
            fourbars.append(fb)

    # Filter
    filtered, _ = _filter_solutions(
        fourbars,
        require_grashof=require_grashof,
        require_crank_rocker=False,
    )
    filtered = _remove_duplicate_solutions(filtered)
    return filtered[:max_pairs]


def _find_coupler_positions_at_targets(
    fourbar: FourBarSolution,
    target_points: list[PrecisionPoint],
    n_steps: int = 360,
) -> list[tuple[Point2D, Point2D, Point2D]] | None:
    """Simulate driving four-bar and find coupler frame at target points.

    For each target point, finds the simulation timestep where the
    coupler point (midpoint of B-C) is closest to the target, then
    returns the positions of B, C, and the midpoint at that timestep.

    Args:
        fourbar: The driving four-bar solution.
        target_points: Points to find intermediate positions for.
        n_steps: Simulation resolution.

    Returns:
        List of (B_pos, C_pos, coupler_midpoint) tuples, one per
        target point. None if simulation fails.
    """
    from ..exceptions import UnbuildableError

    try:
        linkage = solution_to_linkage(fourbar, iterations=n_steps)
    except Exception:
        return None

    # Identify B and C joint indices
    from .._compat import get_parts

    b_idx = c_idx = None
    for i, joint in enumerate(get_parts(linkage)):
        name = getattr(joint, "name", "")
        if name == "B":
            b_idx = i
        elif name == "C":
            c_idx = i

    if b_idx is None or c_idx is None:
        return None

    # Simulate and record B, C positions at each step
    b_trajectory: list[Point2D] = []
    c_trajectory: list[Point2D] = []

    try:
        for positions in linkage.step():
            bx, by = positions[b_idx]
            cx, cy = positions[c_idx]
            if bx is None or by is None or cx is None or cy is None:
                continue
            b_trajectory.append((bx, by))
            c_trajectory.append((cx, cy))
    except UnbuildableError:
        return None

    if len(b_trajectory) < 10:
        return None

    # For each target point, find closest approach of coupler midpoint
    results: list[tuple[Point2D, Point2D, Point2D]] = []
    for tx, ty in target_points:
        best_dist = float("inf")
        best_idx = 0

        for i, (b_pos, c_pos) in enumerate(zip(b_trajectory, c_trajectory, strict=True)):
            # Use the coupler point (P) position if available, else midpoint
            if fourbar.coupler_point is not None:
                # Compute where P would be based on B, C transform
                mx, my = _compute_coupler_point_from_bc(
                    b_pos,
                    c_pos,
                    fourbar.crank_pivot_b,
                    fourbar.coupler_pivot_c,
                    fourbar.coupler_point,
                )
            else:
                mx = (b_pos[0] + c_pos[0]) / 2
                my = (b_pos[1] + c_pos[1]) / 2

            dist = math.sqrt((mx - tx) ** 2 + (my - ty) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        results.append(
            (
                b_trajectory[best_idx],
                c_trajectory[best_idx],
                _midpoint(b_trajectory[best_idx], c_trajectory[best_idx]),
            )
        )

    return results


def _compute_coupler_point_from_bc(
    b_now: Point2D,
    c_now: Point2D,
    b_ref: Point2D,
    c_ref: Point2D,
    p_ref: Point2D,
) -> Point2D:
    """Compute where coupler point P is given current B, C positions.

    P is rigidly attached to the coupler link B-C. Given the reference
    (initial) positions of B, C, P and the current B, C positions,
    compute current P position via rigid body transform.
    """
    # Reference frame: B→C direction
    ref_dx = c_ref[0] - b_ref[0]
    ref_dy = c_ref[1] - b_ref[1]
    ref_angle = math.atan2(ref_dy, ref_dx)

    # Current frame: B→C direction
    cur_dx = c_now[0] - b_now[0]
    cur_dy = c_now[1] - b_now[1]
    cur_angle = math.atan2(cur_dy, cur_dx)

    # P relative to B in reference frame
    px_rel = p_ref[0] - b_ref[0]
    py_rel = p_ref[1] - b_ref[1]

    # Rotate by angle difference
    d_angle = cur_angle - ref_angle
    cos_a = math.cos(d_angle)
    sin_a = math.sin(d_angle)

    px_rot = px_rel * cos_a - py_rel * sin_a
    py_rot = px_rel * sin_a + py_rel * cos_a

    return (b_now[0] + px_rot, b_now[1] + py_rot)


def _synthesize_second_dyad(
    intermediate_positions: list[tuple[Point2D, Point2D, Point2D]],
    target_points: list[PrecisionPoint],
    max_pairs: int = 5,
) -> list[FourBarSolution]:
    """Synthesize the second dyad given intermediate anchor positions.

    The second dyad sees the driving four-bar's coupler joint C as
    one of its anchors and needs a new ground pivot E plus a new
    coupler point F. We construct poses from the intermediate C
    positions and target points, then apply Burmester.

    Args:
        intermediate_positions: (B, C, midpoint) at each target point.
        target_points: The remaining precision points.
        max_pairs: Maximum dyad pairs.

    Returns:
        List of valid FourBarSolution for the second stage.
    """
    if len(intermediate_positions) < 3:
        return []

    # Construct poses for second stage:
    # Position = target point, orientation = angle of B→C at that config
    poses: list[Pose] = []
    for (b_pos, c_pos, _mid), (tx, ty) in zip(intermediate_positions, target_points, strict=True):
        angle = math.atan2(c_pos[1] - b_pos[1], c_pos[0] - b_pos[0])
        poses.append(Pose(x=tx, y=ty, angle=angle))

    # Use at most 5 poses
    poses = poses[:5]

    try:
        curves = compute_circle_point_curve(poses)
    except (ValueError, Exception):
        return []
    if not curves:
        return []

    dyad_pairs = select_compatible_dyads(
        curves,
        max_pairs=max_pairs * 3,
    )

    fourbars: list[FourBarSolution] = []
    for dyad_left, dyad_right in dyad_pairs:
        fb = _dyads_to_fourbar(dyad_left, dyad_right, coupler_point_world=target_points[0])
        if fb is not None:
            fourbars.append(fb)

    filtered, _ = _filter_solutions(
        fourbars,
        require_grashof=False,  # Second stage doesn't need Grashof
        require_crank_rocker=False,
    )
    filtered = _remove_duplicate_solutions(filtered)
    return filtered[:max_pairs]


def _combine_to_six_bar(
    driving_fourbar: FourBarSolution,
    second_fourbar: FourBarSolution,
    precision_points: list[PrecisionPoint],
    stage1_indices: tuple[int, ...],
    stage2_indices: tuple[int, ...],
) -> NBarSolution | None:
    """Combine driving four-bar and second stage into a six-bar solution.

    Maps the two four-bar solutions onto a six-bar joint structure:
    A (ground), D (ground), E (ground from stage 2), B (crank),
    C (coupler pivot), F (output).

    Args:
        driving_fourbar: First-stage four-bar solution.
        second_fourbar: Second-stage four-bar solution.
        precision_points: All precision points.
        stage1_indices: Point indices assigned to stage 1.
        stage2_indices: Point indices assigned to stage 2.

    Returns:
        NBarSolution or None if combination is invalid.
    """
    # Joint positions
    joint_positions: dict[str, Point2D] = {
        "A": driving_fourbar.ground_pivot_a,
        "D": driving_fourbar.ground_pivot_d,
        "B": driving_fourbar.crank_pivot_b,
        "C": driving_fourbar.coupler_pivot_c,
        "E": second_fourbar.ground_pivot_d,  # New ground pivot
        "F": second_fourbar.coupler_pivot_c,  # Output joint
    }

    # Check for degenerate overlap
    positions = list(joint_positions.values())
    for i, p1 in enumerate(positions):
        for j, p2 in enumerate(positions):
            if i < j:
                dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if dist < 1e-6:
                    return None

    # Link lengths
    link_lengths: dict[str, float] = {
        "crank_AB": driving_fourbar.crank_length,
        "coupler_BC": driving_fourbar.coupler_length,
        "rocker_DC": driving_fourbar.rocker_length,
        "ground_AD": driving_fourbar.ground_length,
        "link_CF": second_fourbar.coupler_length,
        "rocker_EF": second_fourbar.rocker_length,
    }

    # Group results
    group_results = [
        GroupSynthesisResult(
            group_index=0,
            group_signature="RRR",
            joint_positions={"B": joint_positions["B"], "C": joint_positions["C"]},
            precision_indices=stage1_indices,
        ),
        GroupSynthesisResult(
            group_index=1,
            group_signature="RRR",
            joint_positions={"F": joint_positions["F"]},
            precision_indices=stage2_indices,
        ),
    ]

    # Coupler point: use the target's first precision point position
    coupler_point = precision_points[0] if precision_points else None

    return NBarSolution(
        topology_id="watt",
        joint_positions=joint_positions,
        link_lengths=link_lengths,
        group_results=group_results,
        coupler_node="F",
        coupler_point=coupler_point,
    )


def _nbar_to_six_bar_linkage(
    solution: NBarSolution,
    iterations: int = 360,
) -> SimLinkage | None:
    """Convert an NBarSolution to a six-bar SimLinkage.

    Creates components: A (Ground), D (Ground), E (Ground),
    B (Crank from A), C (RRRDyad from B, D),
    F (RRRDyad from C, E), P (FixedDyad coupler point on C-F).
    """
    from ..actuators import Crank
    from ..components import Ground
    from ..dyads import FixedDyad, RRRDyad
    from ..simulation import Linkage as SimLinkage

    pos = solution.joint_positions
    lengths = solution.link_lengths

    try:
        # Ground pivots
        ground_a = Ground(pos["A"][0], pos["A"][1], name="A")
        ground_d = Ground(pos["D"][0], pos["D"][1], name="D")
        ground_e = Ground(pos["E"][0], pos["E"][1], name="E")

        # Crank
        angular_velocity = 2 * math.pi / iterations
        initial_angle = math.atan2(
            pos["B"][1] - pos["A"][1],
            pos["B"][0] - pos["A"][0],
        )
        crank_b = Crank(
            anchor=ground_a,
            radius=lengths["crank_AB"],
            angular_velocity=angular_velocity,
            initial_angle=initial_angle,
            name="B",
        )

        # First dyad: C connected to B (coupler) and D (rocker)
        joint_c = RRRDyad(
            anchor1=crank_b.output,
            anchor2=ground_d,
            distance1=lengths["coupler_BC"],
            distance2=lengths["rocker_DC"],
            name="C",
        )

        # Second dyad: F connected to C (link CF) and E (rocker EF)
        joint_f = RRRDyad(
            anchor1=joint_c,
            anchor2=ground_e,
            distance1=lengths["link_CF"],
            distance2=lengths["rocker_EF"],
            name="F",
        )

        components: list[object] = [
            ground_a,
            ground_d,
            ground_e,
            crank_b,
            joint_c,
            joint_f,
        ]

        # Add coupler point tracker if available
        if solution.coupler_point is not None:
            p = solution.coupler_point
            from .conversion import _compute_coupler_point_params

            dist_cp, angle_cp = _compute_coupler_point_params(
                pos["C"],
                pos["F"],
                p,
            )
            joint_p = FixedDyad(
                anchor1=joint_c,
                anchor2=joint_f,
                distance=dist_cp,
                angle=angle_cp,
                name="P",
            )
            components.append(joint_p)

        return SimLinkage(components, name="six_bar_watt")  # type: ignore[arg-type]

    except Exception:
        return None


def _validate_six_bar(
    linkage: Linkage | SimLinkage,
    precision_points: list[PrecisionPoint],
    min_trajectory_fraction: float = 0.1,
) -> bool:
    """Validate that a six-bar can assemble and simulate.

    Checks that:
    1. The linkage doesn't lock up (can complete simulation).
    2. The trajectory has enough valid points (not degenerate).

    This is a structural validation — path accuracy is evaluated
    separately by :func:`~pylinkage.synthesis.ranking.compute_metrics`.

    Args:
        linkage: Six-bar Linkage to validate.
        precision_points: Target points (unused in structural check,
            kept for API compatibility).
        min_trajectory_fraction: Minimum fraction of simulation steps
            that must produce valid positions.

    Returns:
        True if the linkage is structurally valid.
    """
    from ..exceptions import UnbuildableError

    try:
        valid_steps = 0
        total_steps = 0
        for positions in linkage.step():
            total_steps += 1
            # Check that all joints have valid positions
            if all(px is not None and py is not None for px, py in positions):
                valid_steps += 1
    except UnbuildableError:
        return False

    if total_steps == 0:
        return False

    return valid_steps / total_steps >= min_trajectory_fraction


def _optimize_stephenson_triad(
    driving_fourbar: FourBarSolution,
    intermediate_positions: list[tuple[Point2D, Point2D, Point2D]],
    target_points: list[PrecisionPoint],
    all_points: list[PrecisionPoint],
) -> Linkage | None:
    """Optimize Stephenson triad link lengths via least squares.

    For the Stephenson topology, the second stage is a triad (not a
    dyad). We optimize the triad's internal link lengths to minimize
    the distance from the output point to the target precision points.
    """
    import numpy as np
    from scipy.optimize import least_squares as scipy_least_squares

    if len(intermediate_positions) < 2:
        return None

    # Use the first intermediate position to set initial guesses
    # for the new ground pivot E and output joint F
    _, c_pos, _ = intermediate_positions[0]
    tx, ty = target_points[0]

    # Initial guess: E near the target, F at the target
    e_init = ((c_pos[0] + tx) / 2, (c_pos[1] + ty) / 2 - 1.0)
    f_init = (tx, ty)

    # Parameters to optimize: E position (2) + F distances from C and E (2)
    dist_cf_init = math.sqrt((f_init[0] - c_pos[0]) ** 2 + (f_init[1] - c_pos[1]) ** 2)
    dist_ef_init = math.sqrt((f_init[0] - e_init[0]) ** 2 + (f_init[1] - e_init[1]) ** 2)

    x0 = np.array([e_init[0], e_init[1], dist_cf_init, dist_ef_init])

    def residuals(x: np.ndarray) -> np.ndarray:
        ex, ey = x[0], x[1]
        d_cf, d_ef = x[2], x[3]

        if d_cf < 0.1 or d_ef < 0.1:
            return np.full(len(target_points) * 2, 100.0)

        errs = []
        for (_, c_pos_i, _), (tx_i, ty_i) in zip(
            intermediate_positions, target_points, strict=True
        ):
            # Solve for F given C position, E position, distances
            from ..geometry.secants import circle_intersect

            n_int, x1, y1, x2, y2 = circle_intersect(
                c_pos_i[0],
                c_pos_i[1],
                d_cf,
                ex,
                ey,
                d_ef,
            )
            if n_int == 0:
                errs.extend([10.0, 10.0])
                continue

            # Pick closer intersection to target
            d1 = math.sqrt((x1 - tx_i) ** 2 + (y1 - ty_i) ** 2)
            d2 = math.sqrt((x2 - tx_i) ** 2 + (y2 - ty_i) ** 2)
            if d1 <= d2:
                errs.extend([x1 - tx_i, y1 - ty_i])
            else:
                errs.extend([x2 - tx_i, y2 - ty_i])

        return np.array(errs)

    try:
        result = scipy_least_squares(
            residuals,
            x0,
            bounds=([x0[0] - 5, x0[1] - 5, 0.1, 0.1], [x0[0] + 5, x0[1] + 5, 20.0, 20.0]),
            max_nfev=200,
        )
    except Exception:
        return None

    if result.cost > 1.0:
        return None

    ex, ey = result.x[0], result.x[1]
    d_cf, d_ef = result.x[2], result.x[3]

    # Build NBarSolution
    nbar = NBarSolution(
        topology_id="stephenson",
        joint_positions={
            "A": driving_fourbar.ground_pivot_a,
            "D": driving_fourbar.ground_pivot_d,
            "B": driving_fourbar.crank_pivot_b,
            "C": driving_fourbar.coupler_pivot_c,
            "E": (ex, ey),
            "F": f_init,
        },
        link_lengths={
            "crank_AB": driving_fourbar.crank_length,
            "coupler_BC": driving_fourbar.coupler_length,
            "rocker_DC": driving_fourbar.rocker_length,
            "ground_AD": driving_fourbar.ground_length,
            "link_CF": d_cf,
            "rocker_EF": d_ef,
        },
        coupler_node="F",
        coupler_point=all_points[0] if all_points else None,
    )

    return _nbar_to_six_bar_linkage(nbar)  # type: ignore[return-value]


def _midpoint(a: Point2D, b: Point2D) -> Point2D:
    """Compute midpoint of two points."""
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
