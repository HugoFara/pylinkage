#!/usr/bin/env python3
"""
The geometry module provides general geometry functions.

It is used extensively, so each function is optimized with numba.
"""

import math

from .._numba_compat import njit

from .._types import Circle, Coord
from .core import dist

# Return type constants for circle_intersect
INTERSECTION_NONE = 0
INTERSECTION_ONE = 1
INTERSECTION_TWO = 2
INTERSECTION_SAME = 3


@njit(cache=True)  # type: ignore[untyped-decorator]
def circle_intersect(
    x1: float,
    y1: float,
    r1: float,
    x2: float,
    y2: float,
    r2: float,
    tol: float = 0.0,
) -> tuple[int, float, float, float, float]:
    """
    Get the intersections of two circles.

    Transcription of a Matt Woodhead program, method provided by Paul Bourke,
    1997. http://paulbourke.net/geometry/circlesphere/.

    :param x1: X coordinate of first circle center.
    :param y1: Y coordinate of first circle center.
    :param r1: Radius of first circle.
    :param x2: X coordinate of second circle center.
    :param y2: Y coordinate of second circle center.
    :param r2: Radius of second circle.
    :param tol: Distance under which two points are considered equal.

    :returns: Tuple of (n_intersections, x1, y1, x2, y2) where:
        - n=0: No intersection (other values undefined)
        - n=1: One intersection at (x1, y1)
        - n=2: Two intersections at (x1, y1) and (x2, y2)
        - n=3: Same circle (x1, y1, x2 are center and radius)
    """
    dist_x = x2 - x1
    dist_y = y2 - y1
    distance = math.sqrt(dist_x * dist_x + dist_y * dist_y)

    # Use a minimum tolerance for floating-point comparisons
    # to handle near-tangent cases correctly
    eps = max(tol, 1e-12 * max(distance, r1, r2, 1.0))

    # Circles too far apart
    if distance > r1 + r2 + eps:
        return (INTERSECTION_NONE, 0.0, 0.0, 0.0, 0.0)

    # One circle inside the other (with tolerance for near-tangent cases)
    if distance < abs(r2 - r1) - eps:
        return (INTERSECTION_NONE, 0.0, 0.0, 0.0, 0.0)

    # Same circle
    if distance <= eps and abs(r1 - r2) <= eps:
        return (INTERSECTION_SAME, x1, y1, r1, 0.0)

    # Check for tangent case (internal or external)
    is_tangent = (
        abs(distance - (r1 + r2)) <= eps  # External tangent
        or abs(distance - abs(r1 - r2)) <= eps  # Internal tangent
    )

    # Distance from first circle's center to orthogonal projection
    mid_dist = (r1 * r1 - r2 * r2 + distance * distance) / (2.0 * distance)

    # Projected point
    proj_x = x1 + (mid_dist * dist_x) / distance
    proj_y = y1 + (mid_dist * dist_y) / distance

    if is_tangent:
        return (INTERSECTION_ONE, proj_x, proj_y, 0.0, 0.0)

    # Two intersections - compute height from projection to intersections
    height_squared = max(0.0, r1 * r1 - mid_dist * mid_dist)
    height = math.sqrt(height_squared) / distance

    inter1_x = proj_x + height * dist_y
    inter1_y = proj_y - height * dist_x
    inter2_x = proj_x - height * dist_y
    inter2_y = proj_y + height * dist_x

    return (INTERSECTION_TWO, inter1_x, inter1_y, inter2_x, inter2_y)


@njit(cache=True)  # type: ignore[untyped-decorator]
def circle_line_from_points_intersection(
    cx: float,
    cy: float,
    r: float,
    p1_x: float,
    p1_y: float,
    p2_x: float,
    p2_y: float,
) -> tuple[int, float, float, float, float]:
    """
    Intersection(s) of a circle and a line defined by two points.

    :param cx: X coordinate of circle center.
    :param cy: Y coordinate of circle center.
    :param r: Circle radius.
    :param p1_x: X coordinate of first point on line.
    :param p1_y: Y coordinate of first point on line.
    :param p2_x: X coordinate of second point on line.
    :param p2_y: Y coordinate of second point on line.

    :returns: Tuple of (n_intersections, x1, y1, x2, y2) where:
        - n=0: No intersection
        - n=1: One intersection (tangent) at (x1, y1)
        - n=2: Two intersections at (x1, y1) and (x2, y2)
    """
    # Move axis to circle center
    fp_x = p1_x - cx
    fp_y = p1_y - cy
    sp_x = p2_x - cx
    sp_y = p2_y - cy

    dx = sp_x - fp_x
    dy = sp_y - fp_y

    dr2 = dx * dx + dy * dy
    cross = fp_x * sp_y - sp_x * fp_y
    discriminant = r * r * dr2 - cross * cross

    if discriminant < 0:
        return (INTERSECTION_NONE, 0.0, 0.0, 0.0, 0.0)

    reduced_x = cross / dr2
    reduced_y = math.sqrt(discriminant) / dr2

    if discriminant == 0:
        # Tangent line
        return (INTERSECTION_ONE, reduced_x * dy + cx, -reduced_x * dx + cy, 0.0, 0.0)

    # Two intersections
    sign_dy = 1.0 if dy >= 0 else -1.0
    abs_dy = abs(dy)

    x1 = reduced_x * dy - sign_dy * dx * reduced_y + cx
    y1 = -reduced_x * dx - abs_dy * reduced_y + cy
    x2 = reduced_x * dy + sign_dy * dx * reduced_y + cx
    y2 = -reduced_x * dx + abs_dy * reduced_y + cy

    return (INTERSECTION_TWO, x1, y1, x2, y2)


def circle_line_intersection(
    cx: float,
    cy: float,
    r: float,
    a: float,
    b: float,
    c: float,
) -> tuple[int, float, float, float, float]:
    """
    Return the intersection between a line and a circle.

    From https://mathworld.wolfram.com/Circle-LineIntersection.html

    :param cx: X coordinate of circle center.
    :param cy: Y coordinate of circle center.
    :param r: Circle radius.
    :param a: Line equation coefficient a (ax + by + c = 0).
    :param b: Line equation coefficient b.
    :param c: Line equation coefficient c.

    :returns: Tuple of (n_intersections, x1, y1, x2, y2).
    """
    # Find two points on the line
    if b != 0:
        p1_x, p1_y = 0.0, -c / b
        p2_x, p2_y = 1.0, -(c + a) / b
    else:
        p1_x, p1_y = -c / a, 0.0
        p2_x, p2_y = -(c + b) / a, 1.0

    return circle_line_from_points_intersection(cx, cy, r, p1_x, p1_y, p2_x, p2_y)  # type: ignore[no-any-return]


def intersection(
    obj_1: Coord | Circle,
    obj_2: Coord | Circle,
    tol: float = 0.0,
) -> Coord | tuple[Coord, ...] | Circle | None:
    """Intersection of two arbitrary objects.

    The input objects should be points or circles.

    :param obj_1: First point or circle (as tuple).
    :param obj_2: Second point or circle (as tuple).
    :param tol: Absolute tolerance to use if provided.

    :returns: The intersection found, if any.
    """
    # Two points
    if len(obj_1) == 2 and len(obj_2) == 2:
        d = dist(obj_1[0], obj_1[1], obj_2[0], obj_2[1])
        if obj_1 == obj_2 or (tol and d <= tol):
            return obj_1
        return None

    # Two circles
    if len(obj_1) == 3 and len(obj_2) == 3:
        result = circle_intersect(obj_1[0], obj_1[1], obj_1[2], obj_2[0], obj_2[1], obj_2[2], tol)
        if result[0] == INTERSECTION_NONE:
            return ()
        if result[0] == INTERSECTION_ONE:
            return ((result[1], result[2]),)
        if result[0] == INTERSECTION_TWO:
            return ((result[1], result[2]), (result[3], result[4]))
        # Same circle - return the circle itself
        return (obj_1[0], obj_1[1], obj_1[2])

    # Point and circle
    if len(obj_1) == 2 and len(obj_2) == 3:
        d = dist(obj_1[0], obj_1[1], obj_2[0], obj_2[1])
        if d - obj_2[2] <= tol:
            return obj_1
        return None

    # Circle and point
    if len(obj_1) == 3 and len(obj_2) == 2:
        return intersection(obj_1=obj_2, obj_2=obj_1, tol=tol)

    return None


@njit(cache=True)  # type: ignore[untyped-decorator]
def line_line_intersection(
    p1_x: float,
    p1_y: float,
    p2_x: float,
    p2_y: float,
    p3_x: float,
    p3_y: float,
    p4_x: float,
    p4_y: float,
) -> tuple[int, float, float]:
    """
    Intersection of two lines defined by point pairs.

    Line 1 passes through (p1_x, p1_y) and (p2_x, p2_y).
    Line 2 passes through (p3_x, p3_y) and (p4_x, p4_y).

    Uses the parametric line intersection formula.

    :param p1_x: X coordinate of first point on line 1.
    :param p1_y: Y coordinate of first point on line 1.
    :param p2_x: X coordinate of second point on line 1.
    :param p2_y: Y coordinate of second point on line 1.
    :param p3_x: X coordinate of first point on line 2.
    :param p3_y: Y coordinate of first point on line 2.
    :param p4_x: X coordinate of second point on line 2.
    :param p4_y: Y coordinate of second point on line 2.

    :returns: Tuple of (n_intersections, x, y) where:
        - n=0: No intersection (parallel lines)
        - n=1: One intersection at (x, y)
        - n=3: Coincident lines (same line)
    """
    # Direction vectors
    d1_x = p2_x - p1_x
    d1_y = p2_y - p1_y
    d2_x = p4_x - p3_x
    d2_y = p4_y - p3_y

    # Cross product of direction vectors (determinant)
    cross = d1_x * d2_y - d1_y * d2_x

    # Vector from p1 to p3
    diff_x = p3_x - p1_x
    diff_y = p3_y - p1_y

    # Tolerance for parallel check
    eps = 1e-12

    if abs(cross) < eps:
        # Lines are parallel - check if coincident
        # Cross product of diff with d1 should be zero for coincident lines
        cross_diff = diff_x * d1_y - diff_y * d1_x
        if abs(cross_diff) < eps:
            # Coincident lines - return midpoint of p1 and p3 as representative
            return (INTERSECTION_SAME, (p1_x + p3_x) / 2.0, (p1_y + p3_y) / 2.0)
        # Parallel but not coincident
        return (INTERSECTION_NONE, 0.0, 0.0)

    # Compute parameter t for line 1
    t = (diff_x * d2_y - diff_y * d2_x) / cross

    # Intersection point
    x = p1_x + t * d1_x
    y = p1_y + t * d1_y

    return (INTERSECTION_ONE, x, y)


def bounding_box(locus: list[Coord]) -> tuple[float, float, float, float]:
    """
    Compute the bounding box of a locus.

    :param locus: A list of points or any iterable with the same structure.

    :returns: Bounding box as (y_min, x_max, y_max, x_min).
    """
    y_min = float("inf")
    x_min = float("inf")
    y_max = -float("inf")
    x_max = -float("inf")
    for point in locus:
        y_min = min(y_min, point[1])
        x_min = min(x_min, point[0])
        y_max = max(y_max, point[1])
        x_max = max(x_max, point[0])
    return y_min, x_max, y_max, x_min
