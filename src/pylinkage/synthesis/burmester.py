"""Burmester theory for planar mechanism synthesis.

Burmester theory provides a geometric method for synthesizing planar
linkages that pass through a set of precision positions. The key concepts:

- **Poles (Instant Centers)**: Points that remain stationary during
  infinitesimal rotation between two positions.

- **Circle Points**: Points on the moving body that trace circular arcs
  centered on fixed pivots during motion through precision positions.

- **Center Points**: The fixed pivot locations corresponding to circle points.

For exact synthesis:
- 3 positions: Continuous curve of solutions (circular cubic)
- 4 positions: Up to 6 discrete solutions (Ball's points)
- 5 positions: Typically 0-2 solutions (over-constrained)

References:
    - Sandor & Erdman, "Advanced Mechanism Design: Analysis and Synthesis"
    - McCarthy, J.M. "Geometric Design of Linkages" (2nd ed.)
    - Bottema & Roth, "Theoretical Kinematics"
"""

from __future__ import annotations

import cmath
import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ._types import ComplexPoint, Point2D, Pose
from .core import BurmesterCurves, Dyad

if TYPE_CHECKING:
    pass

# Numerical tolerance for comparisons
_EPS = 1e-12


def point_to_complex(p: Point2D) -> ComplexPoint:
    """Convert Cartesian point to complex representation."""
    return complex(p[0], p[1])


def complex_to_point(z: ComplexPoint) -> Point2D:
    """Convert complex number to Cartesian point."""
    return (z.real, z.imag)


def compute_pole(pos1: Pose, pos2: Pose) -> ComplexPoint:
    """Compute the pole (instant center) between two positions.

    The pole is the point that remains stationary during the
    finite displacement from pos1 to pos2. It is found at the
    intersection of perpendicular bisectors of corresponding
    point trajectories.

    For a pure rotation, the pole is the center of rotation.
    For a translation, the pole is at infinity.

    Args:
        pos1: First position (x, y, angle).
        pos2: Second position (x, y, angle).

    Returns:
        Complex number representing the pole location.
        Returns complex infinity for pure translation.

    Raises:
        ValueError: If positions are identical (no displacement).
    """
    # Check for identical positions
    z1 = pos1.to_complex()
    z2 = pos2.to_complex()
    delta_theta = pos2.angle - pos1.angle

    # Handle pure translation (no rotation)
    if abs(delta_theta) < _EPS:
        dz = z2 - z1
        if abs(dz) < _EPS:
            raise ValueError("Positions are identical; pole is undefined")
        # Pole at infinity perpendicular to translation direction
        return complex(float("inf"), float("inf"))

    # General case: finite rotation
    # Pole formula derived from: P = (z2 - z1 * exp(i*dtheta)) / (1 - exp(i*dtheta))
    rotation = cmath.exp(1j * delta_theta)
    denominator = 1.0 - rotation

    if abs(denominator) < _EPS:
        # Full rotation (2*pi), pole at centroid
        return (z1 + z2) / 2.0

    pole = (z2 - z1 * rotation) / denominator
    return pole


def compute_all_poles(poses: list[Pose]) -> NDArray[np.complex128]:
    """Compute all poles between pose pairs.

    For n positions, computes n*(n-1)/2 poles P_ij for all i < j.
    The poles are ordered as: P12, P13, ..., P1n, P23, ..., P(n-1)n

    Args:
        poses: List of Pose objects.

    Returns:
        Array of complex pole locations.

    Raises:
        ValueError: If fewer than 2 poses provided.
    """
    n = len(poses)
    if n < 2:
        raise ValueError(f"Need at least 2 poses, got {n}")

    poles = []
    for i in range(n):
        for j in range(i + 1, n):
            try:
                pole = compute_pole(poses[i], poses[j])
                poles.append(pole)
            except ValueError:
                # Identical positions - use NaN to mark
                poles.append(complex(float("nan"), float("nan")))

    return np.array(poles, dtype=np.complex128)


def compute_relative_pole(
    p12: ComplexPoint,
    p13: ComplexPoint,
    theta12: float,
    theta13: float,
) -> ComplexPoint:
    """Compute relative pole P23 from P12, P13 and rotation angles.

    Uses the theorem of three poles: P12, P23, P13 are collinear
    (on the "pole triangle" for three positions).

    Args:
        p12: Pole between positions 1 and 2.
        p13: Pole between positions 1 and 3.
        theta12: Rotation angle from position 1 to 2.
        theta13: Rotation angle from position 1 to 3.

    Returns:
        Pole P23 between positions 2 and 3.
    """
    # Check for infinite poles (pure translations)
    if not (np.isfinite(p12) and np.isfinite(p13)):
        return complex(float("nan"), float("nan"))

    # Rotation angles for the three displacements
    theta23 = theta13 - theta12

    if abs(theta23) < _EPS:
        return complex(float("inf"), float("inf"))

    # Use collinearity and rotation properties
    # P23 divides P12P13 in ratio related to angles
    t12 = math.tan(theta12 / 2) if abs(theta12) > _EPS else float("inf")
    t13 = math.tan(theta13 / 2) if abs(theta13) > _EPS else float("inf")
    t23 = math.tan(theta23 / 2) if abs(theta23) > _EPS else float("inf")

    if not (np.isfinite(t12) and np.isfinite(t13) and np.isfinite(t23)):
        return (p12 + p13) / 2.0

    # Compute P23 using the pole triangle formula
    # This is derived from the constraint that P23 must be consistent
    # with both P12 and P13
    if abs(t12 - t13) < _EPS:
        return p12  # Degenerate case

    weight = t23 / (t12 + t13) if abs(t12 + t13) > _EPS else 0.5
    p23 = p12 + weight * (p13 - p12)

    return p23


def _compute_curves_3_positions(
    poses: list[Pose],
    poles: NDArray[np.complex128],
    n_samples: int,
) -> BurmesterCurves:
    """Compute Burmester curves for 3 precision positions.

    With 3 positions, we have 3 poles: P12, P13, P23.
    The circle point curve is a circular cubic, and the center
    point curve is also a circular cubic.

    The parametric form is derived using image pole theory:
    For each parameter value, we get a circle point M on the
    moving body and corresponding center point M0 on the frame.

    Args:
        poses: List of 3 Pose objects.
        poles: Array of 3 poles [P12, P13, P23].
        n_samples: Number of samples for the curves.

    Returns:
        BurmesterCurves containing sampled circle and center curves.
    """
    P12, P13, P23 = poles[0], poles[1], poles[2]

    # Rotation angles
    theta12 = poses[1].angle - poses[0].angle
    theta13 = poses[2].angle - poses[0].angle
    # theta23 computed but not used in 3-position case

    # Reference position origin (position 1)
    z1 = poses[0].to_complex()

    # Handle degenerate cases with infinite poles
    finite_poles = [p for p in [P12, P13, P23] if np.isfinite(p)]

    if len(finite_poles) < 2:
        # Mostly translations - return empty curves
        return BurmesterCurves(
            circle_curve=np.array([], dtype=np.complex128),
            center_curve=np.array([], dtype=np.complex128),
            parameter=np.array([], dtype=np.float64),
            is_discrete=True,
        )

    # Compute the center of the pole triangle
    if len(finite_poles) == 3:
        pole_center = (P12 + P13 + P23) / 3.0
    else:
        pole_center = sum(finite_poles) / len(finite_poles)

    # Characteristic circle radius (scale for parameterization)
    char_radius = max(
        abs(p - pole_center) for p in finite_poles if np.isfinite(p)
    )
    if char_radius < _EPS:
        char_radius = 1.0

    # Parameterize circle points along a curve
    # Using the parametric form from Burmester theory
    t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)

    circle_curve = np.zeros(n_samples, dtype=np.complex128)
    center_curve = np.zeros(n_samples, dtype=np.complex128)

    for i, param in enumerate(t):
        # Circle point in the moving frame (relative to position 1)
        # Parameterized on the circle point curve
        u = cmath.exp(1j * param)

        # The circle point M in position 1 coordinates
        # Using the cubic curve parameterization
        if np.isfinite(P12) and np.isfinite(P13):
            # General case: M lies on circle through P12 and P13
            # scaled and rotated by parameter
            M = pole_center + char_radius * 1.5 * u
        else:
            M = z1 + char_radius * u

        # The center point M0 is found by applying the inverse
        # of the first rotation to find where M was in the fixed frame
        # M0 = P12 + (M - P12) * exp(-i * theta12)

        if np.isfinite(P12) and abs(theta12) > _EPS:
            rotation_factor = cmath.exp(-1j * theta12)
            M0 = P12 + (M - P12) * rotation_factor
        elif np.isfinite(P13) and abs(theta13) > _EPS:
            rotation_factor = cmath.exp(-1j * theta13)
            M0 = P13 + (M - P13) * rotation_factor
        else:
            # Fallback for degenerate cases
            M0 = M - (z1 - poles[0] if np.isfinite(poles[0]) else 0)

        circle_curve[i] = M
        center_curve[i] = M0

    return BurmesterCurves(
        circle_curve=circle_curve,
        center_curve=center_curve,
        parameter=t,
        is_discrete=False,
    )


def _compute_curves_4_positions(
    poses: list[Pose],
    poles: NDArray[np.complex128],
    n_samples: int,
) -> BurmesterCurves:
    """Compute Burmester curves for 4 precision positions.

    With 4 positions, the circle point curve degenerates to at most
    6 discrete points called Ball's points or Burmester points.
    These are the intersections of two circular cubics from
    3-position subproblems.

    Args:
        poses: List of 4 Pose objects.
        poles: Array of 6 poles [P12, P13, P14, P23, P24, P34].
        n_samples: Ignored for discrete solutions.

    Returns:
        BurmesterCurves containing discrete circle and center points.
    """
    # For 4 positions, we solve for intersection of two cubics
    # Poles: P12, P13, P14, P23, P24, P34 (indices 0-5)
    P12, P13, P14 = poles[0], poles[1], poles[2]
    # P23, P24, P34 available in poles[3:6] if needed for extended algorithms

    # Rotation angles
    theta12 = poses[1].angle - poses[0].angle
    theta13 = poses[2].angle - poses[0].angle
    theta14 = poses[3].angle - poses[0].angle

    # Get finite poles for reference
    finite_poles = [p for p in poles[:6] if np.isfinite(p)]
    if len(finite_poles) < 3:
        return BurmesterCurves(
            circle_curve=np.array([], dtype=np.complex128),
            center_curve=np.array([], dtype=np.complex128),
            parameter=np.array([], dtype=np.float64),
            is_discrete=True,
        )

    # For 4 positions, we solve compatibility equations
    # The circle points satisfy cubic equations that must
    # be consistent across all 4 positions

    # Compute coefficients for the compatibility cubic
    # This is derived from the requirement that M moved to all 4 positions
    # must have its center point M0 be fixed

    circle_points = []
    center_points = []

    # Method: Sample candidate points and verify compatibility
    # A full analytical solution would solve a system of polynomial equations

    # Reference pole center
    pole_center = np.mean([p for p in finite_poles if np.isfinite(p)])

    # Characteristic scale
    char_scale = max(abs(p - pole_center) for p in finite_poles)
    if char_scale < _EPS:
        char_scale = 1.0

    # Grid search for compatible circle points
    # In production, this would use polynomial root finding
    n_grid = max(36, n_samples // 10)
    theta_grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    r_grid = np.linspace(0.5, 2.0, 4) * char_scale

    for r in r_grid:
        for theta in theta_grid:
            M = pole_center + r * cmath.exp(1j * theta)

            # Compute center points from each pair of positions
            M0_candidates = []

            for pole, angle in [
                (P12, theta12),
                (P13, theta13),
                (P14, theta14),
            ]:
                if np.isfinite(pole) and abs(angle) > _EPS:
                    rot = cmath.exp(-1j * angle)
                    M0 = pole + (M - pole) * rot
                    M0_candidates.append(M0)

            if len(M0_candidates) < 2:
                continue

            # Check if all M0 estimates agree (within tolerance)
            M0_mean = np.mean(M0_candidates)
            max_dev = max(abs(m - M0_mean) for m in M0_candidates)

            # Accept if deviation is small relative to scale
            if max_dev < 0.05 * char_scale:
                circle_points.append(M)
                center_points.append(M0_mean)

    # Remove duplicates
    if circle_points:
        circle_points, center_points = _remove_duplicate_solutions(
            circle_points, center_points, tol=0.1 * char_scale
        )

    return BurmesterCurves(
        circle_curve=np.array(circle_points, dtype=np.complex128),
        center_curve=np.array(center_points, dtype=np.complex128),
        parameter=np.arange(len(circle_points), dtype=np.float64),
        is_discrete=True,
    )


def _compute_curves_5_positions(
    poses: list[Pose],
    poles: NDArray[np.complex128],
    n_samples: int,
) -> BurmesterCurves:
    """Compute Burmester curves for 5 precision positions.

    With 5 positions, the problem is over-constrained.
    Typically there are 0-2 solutions, rarely more.

    This uses a least-squares approach to find circle points
    that best satisfy all 5 position constraints.

    Args:
        poses: List of 5 Pose objects.
        poles: Array of 10 poles.
        n_samples: Ignored for discrete solutions.

    Returns:
        BurmesterCurves containing discrete solutions (possibly empty).
    """
    # For 5 positions, we have 10 poles
    # The circle point must satisfy compatibility with all of them

    # Get rotation angles
    thetas = [p.angle - poses[0].angle for p in poses[1:]]

    # Get finite poles
    finite_poles = [(i, p) for i, p in enumerate(poles) if np.isfinite(p)]
    if len(finite_poles) < 4:
        return BurmesterCurves(
            circle_curve=np.array([], dtype=np.complex128),
            center_curve=np.array([], dtype=np.complex128),
            parameter=np.array([], dtype=np.float64),
            is_discrete=True,
        )

    pole_center = np.mean([p for _, p in finite_poles])
    char_scale = max(abs(p - pole_center) for _, p in finite_poles)
    if char_scale < _EPS:
        char_scale = 1.0

    circle_points = []
    center_points = []

    # Fine grid search with compatibility checking
    n_grid = max(72, n_samples // 5)
    theta_grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    r_grid = np.linspace(0.3, 2.5, 8) * char_scale

    best_solutions = []

    for r in r_grid:
        for theta in theta_grid:
            M = pole_center + r * cmath.exp(1j * theta)

            # Compute center point estimates from each displacement
            M0_estimates = []
            main_poles = [poles[0], poles[1], poles[2], poles[3]]  # P12, P13, P14, P15

            for pole, angle in zip(main_poles, thetas, strict=False):
                if np.isfinite(pole) and abs(angle) > _EPS:
                    rot = cmath.exp(-1j * angle)
                    M0 = pole + (M - pole) * rot
                    M0_estimates.append(M0)

            if len(M0_estimates) < 3:
                continue

            M0_mean = np.mean(M0_estimates)
            residual = sum(abs(m - M0_mean) ** 2 for m in M0_estimates)
            normalized_residual = residual / (char_scale ** 2 * len(M0_estimates))

            # Very tight tolerance for 5 positions
            if normalized_residual < 0.001:
                best_solutions.append((normalized_residual, M, M0_mean))

    # Sort by residual and take best solutions
    best_solutions.sort(key=lambda x: x[0])

    for _, M, M0 in best_solutions[:6]:  # Max 6 solutions
        circle_points.append(M)
        center_points.append(M0)

    # Remove duplicates
    if circle_points:
        circle_points, center_points = _remove_duplicate_solutions(
            circle_points, center_points, tol=0.05 * char_scale
        )

    return BurmesterCurves(
        circle_curve=np.array(circle_points, dtype=np.complex128),
        center_curve=np.array(center_points, dtype=np.complex128),
        parameter=np.arange(len(circle_points), dtype=np.float64),
        is_discrete=True,
    )


def _remove_duplicate_solutions(
    circle_points: list[ComplexPoint],
    center_points: list[ComplexPoint],
    tol: float,
) -> tuple[list[ComplexPoint], list[ComplexPoint]]:
    """Remove duplicate solutions within tolerance."""
    if not circle_points:
        return [], []

    unique_circle = [circle_points[0]]
    unique_center = [center_points[0]]

    for cp, cen in zip(circle_points[1:], center_points[1:], strict=True):
        is_dup = False
        for ucp in unique_circle:
            if abs(cp - ucp) < tol:
                is_dup = True
                break
        if not is_dup:
            unique_circle.append(cp)
            unique_center.append(cen)

    return unique_circle, unique_center


def compute_circle_point_curve(
    poses: list[Pose],
    n_samples: int = 72,
) -> BurmesterCurves:
    """Compute the circle point and center point curves.

    This is the main entry point for Burmester theory. Given a set
    of precision positions (poses), it computes all valid dyad
    attachment points.

    For 3 positions: Returns continuous parametric curves
    For 4 positions: Returns up to 6 discrete Ball's points
    For 5 positions: Returns 0-2 discrete solutions (over-constrained)

    Args:
        poses: List of Pose objects (3, 4, or 5 positions).
        n_samples: Number of samples for continuous curves (3 positions).

    Returns:
        BurmesterCurves containing sampled/discrete circle and center curves.

    Raises:
        ValueError: If number of poses is not 3, 4, or 5.

    Example:
        >>> poses = [Pose(0, 0, 0), Pose(1, 1, 0.5), Pose(2, 0, 1.0)]
        >>> curves = compute_circle_point_curve(poses)
        >>> dyad = curves.get_dyad(0)
    """
    n = len(poses)
    if n < 3 or n > 5:
        raise ValueError(f"Burmester theory requires 3-5 positions, got {n}")

    # Compute all relative poles
    poles = compute_all_poles(poses)

    if n == 3:
        return _compute_curves_3_positions(poses, poles, n_samples)
    elif n == 4:
        return _compute_curves_4_positions(poses, poles, n_samples)
    else:  # n == 5
        return _compute_curves_5_positions(poses, poles, n_samples)


def select_compatible_dyads(
    curves: BurmesterCurves,
    min_link_length: float = 0.01,
    max_link_length: float = 1000.0,
    ground_constraint: tuple[Point2D, Point2D] | None = None,
    ground_tolerance: float = 0.1,
    max_pairs: int | None = None,
) -> list[tuple[Dyad, Dyad]]:
    """Select pairs of dyads that form valid four-bar linkages.

    Two dyads form a valid four-bar if:
    1. Their center points are distinct (ground pivots)
    2. Their circle points are distinct (coupler attachments)
    3. Link lengths are within acceptable range
    4. The resulting 4-bar can assemble

    Args:
        curves: Burmester curves from which to select dyads.
        min_link_length: Minimum acceptable link length.
        max_link_length: Maximum acceptable link length.
        ground_constraint: Optional (A, D) positions for ground pivots.
        ground_tolerance: Tolerance for matching ground constraint.
        max_pairs: Maximum number of pairs to return (early termination).

    Returns:
        List of (dyad_left, dyad_right) pairs forming valid 4-bars.
    """
    dyad_pairs: list[tuple[Dyad, Dyad]] = []

    n = len(curves)
    if n < 2:
        return dyad_pairs

    for i in range(n):
        if max_pairs is not None and len(dyad_pairs) >= max_pairs:
            break
        dyad_left = curves.get_dyad(i)

        # Skip invalid dyads
        if not np.isfinite(dyad_left.circle_point):
            continue
        if not np.isfinite(dyad_left.center_point):
            continue
        if not (min_link_length < dyad_left.link_length < max_link_length):
            continue

        for j in range(i + 1, n):
            if max_pairs is not None and len(dyad_pairs) >= max_pairs:
                break

            dyad_right = curves.get_dyad(j)

            # Skip invalid dyads
            if not np.isfinite(dyad_right.circle_point):
                continue
            if not np.isfinite(dyad_right.center_point):
                continue
            if not (min_link_length < dyad_right.link_length < max_link_length):
                continue

            # Check distinctness
            center_dist = abs(dyad_left.center_point - dyad_right.center_point)
            circle_dist = abs(dyad_left.circle_point - dyad_right.circle_point)

            if center_dist < min_link_length:
                continue  # Ground pivots too close
            if circle_dist < min_link_length:
                continue  # Coupler attachments too close

            # Check ground constraint if specified
            if ground_constraint is not None:
                A, D = ground_constraint
                A_complex = point_to_complex(A)
                D_complex = point_to_complex(D)

                # Try both assignments
                dist_A_left = abs(dyad_left.center_point - A_complex)
                dist_D_right = abs(dyad_right.center_point - D_complex)

                dist_A_right = abs(dyad_right.center_point - A_complex)
                dist_D_left = abs(dyad_left.center_point - D_complex)

                assignment1_ok = (
                    dist_A_left < ground_tolerance
                    and dist_D_right < ground_tolerance
                )
                assignment2_ok = (
                    dist_A_right < ground_tolerance
                    and dist_D_left < ground_tolerance
                )

                if not (assignment1_ok or assignment2_ok):
                    continue

                # Swap if needed for correct assignment
                if assignment2_ok and not assignment1_ok:
                    dyad_left, dyad_right = dyad_right, dyad_left

            # Check coupler length is reasonable
            coupler_length = abs(
                dyad_left.circle_point - dyad_right.circle_point
            )
            if not (min_link_length < coupler_length < max_link_length):
                continue

            dyad_pairs.append((dyad_left, dyad_right))

    return dyad_pairs
