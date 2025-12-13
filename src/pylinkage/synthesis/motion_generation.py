"""Motion generation synthesis for four-bar linkages.

Motion generation (rigid body guidance) synthesizes a linkage where
a body attached to the coupler moves through specified poses
(position + orientation). This is the most constrained form of
synthesis because both position and orientation are prescribed.

Applications:
- Robot end-effector positioning
- Pick-and-place with orientation control
- Assembly operations
- Door/hatch mechanisms

Constraints:
- 3 poses: Continuous curve of solutions (circular cubic)
- 4 poses: Up to 6 discrete solutions (Ball's points)
- 5 poses: Typically 0-2 solutions (over-constrained)
- 6+ poses: Almost always no solution exists

References:
    - McCarthy, Chapter 6: "Spherical and Planar Four-Bar Motion"
    - Sandor & Erdman, Chapter 5: "Path, Motion, and Function Generation"
    - Bottema & Roth, "Theoretical Kinematics"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg as scipy_linalg

from ._types import FourBarSolution, Point2D, Pose, SynthesisType
from .burmester import (
    complex_to_point,
    compute_circle_point_curve,
    select_compatible_dyads,
)
from .core import Dyad, SynthesisProblem, SynthesisResult
from .utils import GrashofType, grashof_check, validate_fourbar

if TYPE_CHECKING:
    pass


def _dyads_to_motion_fourbar(
    dyad_left: Dyad,
    dyad_right: Dyad,
    reference_pose: Pose,
) -> FourBarSolution | None:
    """Convert dyads to four-bar for motion generation.

    In motion generation, the circle points are on the moving body
    (coupler) and the center points are fixed pivots (frame).

    Args:
        dyad_left: Dyad for left side (A-B connection).
        dyad_right: Dyad for right side (D-C connection).
        reference_pose: Reference pose (typically first pose).

    Returns:
        FourBarSolution if valid, None if degenerate.
    """
    # Ground pivots (center points are fixed)
    A = complex_to_point(dyad_left.center_point)
    D = complex_to_point(dyad_right.center_point)

    # Coupler attachment points (circle points move with body)
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
    )


def _filter_motion_solutions(
    solutions: list[FourBarSolution],
    require_grashof: bool = True,
    require_crank_rocker: bool = False,
    min_link_ratio: float = 0.1,
    max_link_ratio: float = 10.0,
) -> tuple[list[FourBarSolution], list[str]]:
    """Filter motion generation solutions.

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
            f"All {len(solutions)} candidate solutions were filtered out."
        )

    return valid, warnings


def _remove_duplicate_solutions(
    solutions: list[FourBarSolution],
    tolerance: float = 0.01,
) -> list[FourBarSolution]:
    """Remove duplicate solutions within tolerance."""
    if not solutions:
        return []

    unique: list[FourBarSolution] = [solutions[0]]

    for sol in solutions[1:]:
        is_duplicate = False

        for usol in unique:
            diff = abs(sol.crank_length - usol.crank_length)
            diff += abs(sol.coupler_length - usol.coupler_length)
            diff += abs(sol.rocker_length - usol.rocker_length)
            diff += abs(sol.ground_length - usol.ground_length)

            scale = (
                sol.crank_length
                + sol.coupler_length
                + sol.rocker_length
                + sol.ground_length
            )

            if diff / scale < tolerance:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(sol)

    return unique


def motion_generation(
    poses: list[Pose],
    ground_pivot_a: Point2D | None = None,
    ground_pivot_d: Point2D | None = None,
    max_solutions: int | None = 10,
    require_grashof: bool = True,
    require_crank_rocker: bool = False,
) -> SynthesisResult:
    """Synthesize a four-bar for motion generation (rigid body guidance).

    Find four-bar linkages where a body attached to the coupler
    passes through the specified poses (position + orientation).

    This is the most constrained synthesis problem because both
    position and orientation are specified at each precision position.

    Args:
        poses: List of Pose objects specifying body positions and orientations.
            Typically 3-5 poses (5 is maximum for exact synthesis).
        ground_pivot_a: Optional fixed position for left ground pivot.
        ground_pivot_d: Optional fixed position for right ground pivot.
        max_solutions: Maximum number of solutions to return (None for all).
        require_grashof: If True, reject non-Grashof solutions.
        require_crank_rocker: If True, only accept crank-rocker type.

    Returns:
        SynthesisResult containing valid linkages.

    Example:
        >>> from pylinkage.synthesis import Pose, motion_generation
        >>> poses = [
        ...     Pose(0, 0, 0),
        ...     Pose(1, 1, 0.5),
        ...     Pose(2, 0.5, 1.0),
        ... ]
        >>> result = motion_generation(poses)
        >>> print(f"Found {len(result)} solutions")
    """
    problem = SynthesisProblem(
        synthesis_type=SynthesisType.MOTION,
        poses=list(poses),
        ground_pivot_a=ground_pivot_a,
        ground_pivot_d=ground_pivot_d,
    )

    warnings: list[str] = []
    raw_solutions: list[FourBarSolution] = []

    n_poses = len(poses)

    if n_poses < 3:
        warnings.append(f"Motion generation needs at least 3 poses, got {n_poses}")
        return SynthesisResult(
            solutions=[],
            raw_solutions=[],
            problem=problem,
            warnings=warnings,
        )

    if n_poses > 5:
        warnings.append(
            f"Motion generation with {n_poses} poses is highly over-constrained. "
            "Exact solutions likely do not exist. Using first 5 poses."
        )
        poses = poses[:5]

    # Ground constraint for dyad selection
    ground_constraint = None
    ground_tolerance = 0.1
    if ground_pivot_a is not None and ground_pivot_d is not None:
        ground_constraint = (ground_pivot_a, ground_pivot_d)

    try:
        # Compute Burmester curves directly from poses
        # Motion generation uses poses directly (unlike path gen which searches orientations)
        curves = compute_circle_point_curve(poses)

        if not curves:
            warnings.append(
                "No Burmester solutions found for these poses. "
                "The poses may be geometrically incompatible."
            )
        else:
            # Select compatible dyad pairs
            # Use early termination: collect 3x max_solutions to allow for filtering
            max_pairs = (max_solutions * 3) if max_solutions else None
            dyad_pairs = select_compatible_dyads(
                curves,
                ground_constraint=ground_constraint,
                ground_tolerance=ground_tolerance,
                max_pairs=max_pairs,
            )

            if not dyad_pairs:
                warnings.append(
                    "Burmester curves found but no compatible dyad pairs. "
                    "Try different poses or remove ground constraints."
                )

            for dyad_left, dyad_right in dyad_pairs:
                solution = _dyads_to_motion_fourbar(
                    dyad_left, dyad_right, poses[0]
                )

                if solution is not None:
                    raw_solutions.append(solution)

    except (ValueError, np.linalg.LinAlgError, scipy_linalg.LinAlgError) as e:
        warnings.append(f"Synthesis computation failed: {e}")

    # Remove duplicates
    raw_solutions = _remove_duplicate_solutions(raw_solutions)

    # Filter solutions
    raw_solutions, filter_warnings = _filter_motion_solutions(
        raw_solutions,
        require_grashof=require_grashof,
        require_crank_rocker=require_crank_rocker,
    )
    warnings.extend(filter_warnings)

    # Limit to max_solutions
    if max_solutions and len(raw_solutions) > max_solutions:
        raw_solutions = raw_solutions[:max_solutions]

    # Convert to Linkage objects
    from .conversion import solutions_to_linkages

    linkages = solutions_to_linkages(raw_solutions, SynthesisType.MOTION)

    if not linkages and not warnings:
        warnings.append(
            "No valid solutions found. Motion generation is highly constrained. "
            "Try:\n"
            "  - Using 3-4 poses instead of more\n"
            "  - Checking that poses are kinematically feasible\n"
            "  - Setting require_grashof=False"
        )

    return SynthesisResult(
        solutions=linkages,
        raw_solutions=raw_solutions,
        problem=problem,
        warnings=warnings,
    )


def motion_generation_3_poses(
    poses: list[Pose],
    n_samples: int = 36,
    max_solutions: int | None = 50,
) -> SynthesisResult:
    """Specialized synthesis for exactly 3 poses.

    With 3 poses, there is a continuous curve of solutions.
    This function samples the curve and returns multiple linkages.

    Args:
        poses: Exactly 3 Pose objects.
        n_samples: Number of samples along the solution curve.
        max_solutions: Maximum raw solutions to collect (early termination).

    Returns:
        SynthesisResult with sampled solutions from the curve.
    """
    if len(poses) != 3:
        return motion_generation(poses)  # Fallback to general

    problem = SynthesisProblem(
        synthesis_type=SynthesisType.MOTION,
        poses=list(poses),
    )

    warnings: list[str] = []
    raw_solutions: list[FourBarSolution] = []

    try:
        curves = compute_circle_point_curve(poses, n_samples=n_samples)

        if curves:
            # Sample dyad pairs along the curve
            step = max(1, len(curves) // n_samples)
            done = False

            for i in range(0, len(curves), step):
                if done:
                    break
                for j in range(i + step, len(curves), step):
                    dyad_left = curves.get_dyad(i)
                    dyad_right = curves.get_dyad(j)

                    solution = _dyads_to_motion_fourbar(
                        dyad_left, dyad_right, poses[0]
                    )

                    if solution is not None:
                        # Quick validation
                        is_valid, _ = validate_fourbar(solution)
                        if is_valid:
                            raw_solutions.append(solution)
                            if max_solutions and len(raw_solutions) >= max_solutions:
                                done = True
                                break

    except (ValueError, np.linalg.LinAlgError, scipy_linalg.LinAlgError) as e:
        warnings.append(f"Synthesis failed: {e}")

    # Remove duplicates
    raw_solutions = _remove_duplicate_solutions(raw_solutions, tolerance=0.05)

    # Filter to reasonable linkages
    raw_solutions, filter_warnings = _filter_motion_solutions(
        raw_solutions, require_grashof=False
    )
    warnings.extend(filter_warnings)

    from .conversion import solutions_to_linkages

    linkages = solutions_to_linkages(raw_solutions, SynthesisType.MOTION)

    return SynthesisResult(
        solutions=linkages,
        raw_solutions=raw_solutions,
        problem=problem,
        warnings=warnings,
    )
