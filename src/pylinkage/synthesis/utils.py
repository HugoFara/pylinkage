"""Utility functions for mechanism synthesis.

This module provides helper functions for:
- Coordinate conversions between Cartesian and complex representations
- Grashof criterion checking for four-bar mobility
- Solution validation and filtering
- Numerical stability utilities
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as scipy_linalg

from ._types import ComplexPoint, FourBarSolution, Point2D

if TYPE_CHECKING:
    pass


class GrashofType(Enum):
    """Classification of four-bar linkage by Grashof criterion."""

    GRASHOF_CRANK_ROCKER = auto()
    """Shortest link is crank: full rotation of crank, oscillation of rocker."""
    GRASHOF_DOUBLE_CRANK = auto()
    """Shortest link is frame: both crank and rocker can fully rotate."""
    GRASHOF_ROCKER_CRANK = auto()
    """Shortest link is rocker: crank oscillates, rocker fully rotates."""
    GRASHOF_DOUBLE_ROCKER = auto()
    """Shortest link is coupler: both crank and rocker oscillate."""
    NON_GRASHOF = auto()
    """Non-Grashof: s + l > p + q, no link can fully rotate."""
    CHANGE_POINT = auto()
    """Change point (special Grashof): s + l = p + q exactly."""


def point_to_complex(p: Point2D) -> ComplexPoint:
    """Convert Cartesian point to complex representation.

    Args:
        p: Point as (x, y) tuple.

    Returns:
        Complex number x + iy.
    """
    return complex(p[0], p[1])


def complex_to_point(z: ComplexPoint) -> Point2D:
    """Convert complex number to Cartesian point.

    Args:
        z: Complex number x + iy.

    Returns:
        Point as (x, y) tuple.
    """
    return (z.real, z.imag)


def distance(p1: Point2D, p2: Point2D) -> float:
    """Euclidean distance between two points.

    Args:
        p1: First point (x, y).
        p2: Second point (x, y).

    Returns:
        Distance between points.
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def grashof_check(
    crank: float,
    coupler: float,
    rocker: float,
    ground: float,
) -> GrashofType:
    """Determine Grashof classification of a four-bar linkage.

    The Grashof criterion states that for a four-bar linkage to have
    at least one link capable of full rotation, the sum of the shortest
    and longest links must be less than or equal to the sum of the
    remaining two links: s + l <= p + q

    Args:
        crank: Length of crank (input link).
        coupler: Length of coupler (connecting link).
        rocker: Length of rocker (output link).
        ground: Length of ground (frame link).

    Returns:
        GrashofType classification of the linkage.

    Example:
        >>> grashof_check(1.0, 3.0, 3.0, 4.0)
        GrashofType.GRASHOF_CRANK_ROCKER
    """
    links = [crank, coupler, rocker, ground]
    sorted_links = sorted(links)
    s, p, q, longest = sorted_links  # shortest to longest

    grashof_sum = s + longest
    other_sum = p + q

    # Tolerance for numerical comparison
    tol = 1e-10 * max(links)

    if abs(grashof_sum - other_sum) < tol:
        return GrashofType.CHANGE_POINT

    if grashof_sum > other_sum:
        return GrashofType.NON_GRASHOF

    # Grashof condition satisfied - classify by shortest link position
    shortest_link = sorted_links[0]

    if abs(shortest_link - ground) < tol:
        return GrashofType.GRASHOF_DOUBLE_CRANK
    elif abs(shortest_link - crank) < tol:
        return GrashofType.GRASHOF_CRANK_ROCKER
    elif abs(shortest_link - rocker) < tol:
        return GrashofType.GRASHOF_ROCKER_CRANK
    else:  # shortest is coupler
        return GrashofType.GRASHOF_DOUBLE_ROCKER


def is_grashof(
    crank: float,
    coupler: float,
    rocker: float,
    ground: float,
) -> bool:
    """Check if a four-bar linkage satisfies Grashof criterion.

    A Grashof linkage has at least one link that can fully rotate.

    Args:
        crank: Length of crank (input link).
        coupler: Length of coupler (connecting link).
        rocker: Length of rocker (output link).
        ground: Length of ground (frame link).

    Returns:
        True if Grashof criterion is satisfied.
    """
    grashof_type = grashof_check(crank, coupler, rocker, ground)
    return grashof_type != GrashofType.NON_GRASHOF


def is_crank_rocker(
    crank: float,
    coupler: float,
    rocker: float,
    ground: float,
) -> bool:
    """Check if linkage is a crank-rocker mechanism.

    A crank-rocker has the crank as the shortest link and satisfies
    Grashof. The crank can fully rotate while the rocker oscillates.

    Args:
        crank: Length of crank (input link).
        coupler: Length of coupler (connecting link).
        rocker: Length of rocker (output link).
        ground: Length of ground (frame link).

    Returns:
        True if linkage is crank-rocker type.
    """
    return grashof_check(crank, coupler, rocker, ground) in (
        GrashofType.GRASHOF_CRANK_ROCKER,
        GrashofType.CHANGE_POINT,
    )


def validate_fourbar(solution: FourBarSolution) -> tuple[bool, list[str]]:
    """Validate a four-bar solution for geometric consistency.

    Checks for:
    - Non-zero link lengths
    - Positive link lengths
    - Assembly feasibility (triangle inequality)

    Args:
        solution: FourBarSolution to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors: list[str] = []

    # Check for non-zero, positive link lengths
    lengths = [
        ("crank", solution.crank_length),
        ("coupler", solution.coupler_length),
        ("rocker", solution.rocker_length),
        ("ground", solution.ground_length),
    ]

    for name, length in lengths:
        if length <= 0:
            errors.append(f"{name} length must be positive, got {length}")
        elif length < 1e-10:
            errors.append(f"{name} length is too small: {length}")

    if errors:
        return False, errors

    # Check assembly feasibility at initial position
    crank = solution.crank_length
    coupler = solution.coupler_length
    rocker = solution.rocker_length
    ground = solution.ground_length

    # Coupler must be able to connect crank to rocker
    # At closest approach: |ground - crank| <= coupler + rocker
    # At farthest: ground + crank >= |coupler - rocker|
    min_reach = abs(ground - crank)
    max_reach = ground + crank

    if coupler + rocker < min_reach:
        errors.append(
            f"Linkage cannot assemble: coupler + rocker ({coupler + rocker:.4f}) "
            f"< |ground - crank| ({min_reach:.4f})"
        )

    if abs(coupler - rocker) > max_reach:
        errors.append(
            f"Linkage cannot assemble: |coupler - rocker| ({abs(coupler - rocker):.4f}) "
            f"> ground + crank ({max_reach:.4f})"
        )

    return len(errors) == 0, errors


def rotation_matrix_2d(angle: float) -> NDArray[np.float64]:
    """Create 2D rotation matrix.

    Args:
        angle: Rotation angle in radians.

    Returns:
        2x2 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotate_point(
    point: Point2D,
    angle: float,
    center: Point2D = (0.0, 0.0),
) -> Point2D:
    """Rotate a point about a center.

    Args:
        point: Point to rotate.
        angle: Rotation angle in radians (counterclockwise positive).
        center: Center of rotation.

    Returns:
        Rotated point.
    """
    # Translate to origin
    px, py = point[0] - center[0], point[1] - center[1]

    # Rotate
    c, s = math.cos(angle), math.sin(angle)
    rx = px * c - py * s
    ry = px * s + py * c

    # Translate back
    return (rx + center[0], ry + center[1])


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi] range.

    Args:
        angle: Angle in radians.

    Returns:
        Equivalent angle in [-pi, pi].
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def angle_between_points(
    origin: Point2D,
    target: Point2D,
) -> float:
    """Compute angle from origin to target point.

    Args:
        origin: Reference point.
        target: Target point.

    Returns:
        Angle in radians from positive x-axis.
    """
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    return math.atan2(dy, dx)


def check_condition_number(
    matrix: NDArray[np.floating],
    threshold: float = 1e10,
) -> bool:
    """Check if matrix is well-conditioned.

    Args:
        matrix: Matrix to check.
        threshold: Maximum acceptable condition number.

    Returns:
        True if condition number is below threshold.
    """
    try:
        cond = float(np.linalg.cond(matrix))
        return cond < threshold
    except (np.linalg.LinAlgError, scipy_linalg.LinAlgError):
        return False
