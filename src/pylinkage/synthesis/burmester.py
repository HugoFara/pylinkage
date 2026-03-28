"""Burmester theory for planar mechanism synthesis.

Burmester theory provides a geometric method for synthesizing planar
linkages that pass through a set of precision positions. The key concepts:

- **Circle Points**: Points on the moving body whose world-frame positions
  at all precision poses lie on a single circle.

- **Center Points**: The centers of those circles — these become the
  fixed pivots (ground joints) of the linkage.

For exact synthesis:
- 3 positions: Every body-frame point is a valid circle point (∞ solutions)
- 4 positions: Circle points form a curve (circular cubic); we find points
  where all 4 world positions are concyclic
- 5 positions: Typically 0–6 discrete solutions (highly constrained)

Two compatible dyads (circle point + center point pairs) define a four-bar
linkage: the circle points are coupler attachment joints, the center points
are ground pivots.

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


def _body_point_world_positions(
    q: ComplexPoint,
    poses: list[Pose],
) -> list[ComplexPoint]:
    """Compute world positions of a body-frame point at each pose.

    A point with body-fixed coordinates ``q`` (relative to the body
    reference frame at pose 1) has world position at pose j:

        w_j = z_j + q * exp(i * θ_j)

    where z_j and θ_j are the pose origin and orientation.

    For pose 1 the body reference is at z_1 with orientation θ_1,
    so w_1 = z_1 + q * exp(i * θ_1).

    Args:
        q: Body-frame coordinates of the point (complex number).
        poses: List of Pose objects defining body positions.

    Returns:
        List of world-frame positions (complex numbers).
    """
    return [
        pose.to_complex() + q * cmath.exp(1j * pose.angle)
        for pose in poses
    ]


def _circumcenter(z1: ComplexPoint, z2: ComplexPoint, z3: ComplexPoint) -> ComplexPoint:
    """Compute the circumcenter of three points in the complex plane.

    The circumcenter is equidistant from all three points. Returns
    complex infinity if the points are collinear.

    Uses the formula:
        m0 = |z1|²(z2-z3) + |z2|²(z3-z1) + |z3|²(z1-z2)
             -----------------------------------------------
             2 * Im[(z1-z3)(conj(z2)-conj(z3))] * i    ... (cross product denom)

    Equivalently, using the determinant formula.
    """
    ax, ay = z1.real, z1.imag
    bx, by = z2.real, z2.imag
    cx, cy = z3.real, z3.imag

    D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    if abs(D) < _EPS:
        return complex(float("inf"), float("inf"))

    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy

    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / D
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / D

    return complex(ux, uy)


def _circumradius(center: ComplexPoint, point: ComplexPoint) -> float:
    """Distance from circumcenter to a point on the circle."""
    return abs(point - center)


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


def _compute_curves_3_positions(
    poses: list[Pose],
    poles: NDArray[np.complex128],
    n_samples: int,
) -> BurmesterCurves:
    """Compute Burmester curves for 3 precision positions.

    With 3 positions, every point on the moving body is a valid circle
    point — its 3 world positions always define a unique circumcircle.
    The center point is the circumcenter of those 3 positions.

    We sample body-frame points on concentric rings around the body
    reference (the coupler point / precision point location) and compute
    the corresponding center points.

    Args:
        poses: List of 3 Pose objects.
        poles: Array of 3 poles [P12, P13, P23].
        n_samples: Number of samples for the curves.

    Returns:
        BurmesterCurves containing sampled circle and center curves.
    """
    # Compute characteristic scale from the poses
    pose_positions = [p.to_complex() for p in poses]
    dists = [abs(pose_positions[i] - pose_positions[j])
             for i in range(3) for j in range(i + 1, 3)]
    char_scale = max(dists) if dists else 1.0
    if char_scale < _EPS:
        char_scale = 1.0

    # Sample body-frame points q on multiple rings
    # Use several radii to get a diverse set of candidates
    n_angles = max(12, n_samples // 4)
    radii = [0.3, 0.7, 1.2, 2.0]
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    circle_points: list[ComplexPoint] = []
    center_points: list[ComplexPoint] = []

    for r_factor in radii:
        r = r_factor * char_scale
        for angle in angles:
            q = r * cmath.exp(1j * angle)

            # World positions of this body-frame point
            w = _body_point_world_positions(q, poses)

            # Circumcenter = center point (ground pivot)
            m0 = _circumcenter(w[0], w[1], w[2])

            if not np.isfinite(m0):
                continue

            # The circle point in world frame at position 1
            circle_pt = w[0]

            # Sanity: link length should be reasonable
            link_len = abs(circle_pt - m0)
            if link_len < _EPS or link_len > 100.0 * char_scale:
                continue

            circle_points.append(circle_pt)
            center_points.append(m0)

    # Pad to n_samples if we have fewer (or trim if more)
    if len(circle_points) > n_samples:
        step = len(circle_points) / n_samples
        indices = [int(i * step) for i in range(n_samples)]
        circle_points = [circle_points[i] for i in indices]
        center_points = [center_points[i] for i in indices]

    return BurmesterCurves(
        circle_curve=np.array(circle_points, dtype=np.complex128),
        center_curve=np.array(center_points, dtype=np.complex128),
        parameter=np.arange(len(circle_points), dtype=np.float64),
        is_discrete=False,
    )


def _compute_curves_4_positions(
    poses: list[Pose],
    poles: NDArray[np.complex128],
    n_samples: int,
) -> BurmesterCurves:
    """Compute Burmester curves for 4 precision positions.

    With 4 positions, a body-frame point is a valid circle point only if
    all 4 of its world positions lie on a single circle (are concyclic).
    We sample candidate body-frame points and check the concyclicity
    condition: compute circumcircle of the first 3 world positions, then
    check whether the 4th lies on that circle.

    Args:
        poses: List of 4 Pose objects.
        poles: Array of 6 poles [P12, P13, P14, P23, P24, P34].
        n_samples: Number of samples for candidate search.

    Returns:
        BurmesterCurves containing discrete circle and center points.
    """
    # Characteristic scale
    pose_positions = [p.to_complex() for p in poses]
    dists = [abs(pose_positions[i] - pose_positions[j])
             for i in range(4) for j in range(i + 1, 4)]
    char_scale = max(dists) if dists else 1.0
    if char_scale < _EPS:
        char_scale = 1.0

    circle_points: list[ComplexPoint] = []
    center_points: list[ComplexPoint] = []

    # Dense grid search over body-frame points
    n_grid = max(48, n_samples)
    theta_grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    r_factors = np.linspace(0.2, 3.0, 12)

    for r_factor in r_factors:
        r = r_factor * char_scale
        for theta in theta_grid:
            q = r * cmath.exp(1j * theta)

            # World positions
            w = _body_point_world_positions(q, poses)

            # Circumcircle of first 3 positions
            m0 = _circumcenter(w[0], w[1], w[2])
            if not np.isfinite(m0):
                continue

            circumradius = _circumradius(m0, w[0])
            if circumradius < _EPS:
                continue

            # Check if 4th point lies on the same circle
            r4 = _circumradius(m0, w[3])
            relative_error = abs(r4 - circumradius) / circumradius

            if relative_error < 0.02:  # 2% tolerance
                circle_points.append(w[0])
                center_points.append(m0)

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


def _compute_curves_5_positions(
    poses: list[Pose],
    poles: NDArray[np.complex128],
    n_samples: int,
) -> BurmesterCurves:
    """Compute Burmester curves for 5 precision positions.

    With 5 positions, the problem is highly constrained. A body-frame
    point is valid only if all 5 world positions are concyclic. We use
    a fine grid search with tight tolerance.

    Args:
        poses: List of 5 Pose objects.
        poles: Array of 10 poles.
        n_samples: Ignored for discrete solutions.

    Returns:
        BurmesterCurves containing discrete solutions (possibly empty).
    """
    # Characteristic scale
    pose_positions = [p.to_complex() for p in poses]
    dists = [abs(pose_positions[i] - pose_positions[j])
             for i in range(5) for j in range(i + 1, 5)]
    char_scale = max(dists) if dists else 1.0
    if char_scale < _EPS:
        char_scale = 1.0

    # Collect candidates with their residuals for ranking
    candidates: list[tuple[float, ComplexPoint, ComplexPoint]] = []

    # Fine grid search
    n_grid = max(72, n_samples)
    theta_grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    r_factors = np.linspace(0.2, 3.0, 16)

    for r_factor in r_factors:
        r = r_factor * char_scale
        for theta in theta_grid:
            q = r * cmath.exp(1j * theta)

            # World positions
            w = _body_point_world_positions(q, poses)

            # Circumcircle of first 3 positions
            m0 = _circumcenter(w[0], w[1], w[2])
            if not np.isfinite(m0):
                continue

            circumradius = _circumradius(m0, w[0])
            if circumradius < _EPS:
                continue

            # Check positions 4 and 5
            errors = []
            for k in range(3, 5):
                rk = _circumradius(m0, w[k])
                errors.append(abs(rk - circumradius) / circumradius)

            max_error = max(errors)

            if max_error < 0.01:  # 1% tolerance
                candidates.append((max_error, w[0], m0))

    # Sort by residual and take best
    candidates.sort(key=lambda x: x[0])

    circle_points = [c[1] for c in candidates[:6]]
    center_points = [c[2] for c in candidates[:6]]

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
    if abs(t12 - t13) < _EPS:
        return p12  # Degenerate case

    weight = t23 / (t12 + t13) if abs(t12 + t13) > _EPS else 0.5
    p23 = p12 + weight * (p13 - p12)

    return p23
