#!/usr/bin/env python3
"""
The geometry module provides general geometry functions.

It is used extensively, so each function should be highly optimized.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .._types import Circle, Coord, Line

# Type for circle intersection results
# Note: Using string annotations for forward references
CircleIntersectionType = Union[
    tuple[int],
    "tuple[int, Coord]",
    "tuple[int, Coord, Coord]",
    "tuple[int, Circle]",
]


def secant_circles_intersections(
    distance: float,
    dist_x: float,
    dist_y: float,
    mid_dist: float,
    radius1: float,
    projected: Coord,
) -> tuple[int, Coord, Coord]:
    """Return the TWO intersections of secant circles."""
    # Distance between projected P and points
    # and the points of which P is projection
    # Clamp to handle floating-point precision issues near tangent cases
    height_squared = max(0.0, radius1 ** 2 - mid_dist ** 2)
    height = math.sqrt(height_squared) / distance
    inter1: Coord = (
        projected[0] + height * dist_y,
        projected[1] - height * dist_x,
    )
    inter2: Coord = (
        projected[0] - height * dist_y,
        projected[1] + height * dist_x,
    )
    return 2, inter1, inter2


def circle_intersect(
    circle1: Circle,
    circle2: Circle,
    tol: float = 0.0,
) -> CircleIntersectionType:
    """
    Get the intersections of two circles.

    Transcription of a Matt Woodhead program, method provided by Paul Bourke,
    1997. http://paulbourke.net/geometry/circlesphere/.

    :param circle1: First circle as (x, y, radius).
    :param circle2: Second circle as (x, y, radius).
    :param tol: Distance under which two points are considered equal (Default value = 0.0).

    :returns: The intersections of two circles. Can be:
        - (0, ) when no intersection.
        - (1, (float, float)) if the intersection is one point.
        - (2, (float, float), (float, float)) for two points of intersection.
        - (3, (float, float, float)) if the intersection is a circle.
    """
    x_1, y_1, radius1 = circle1
    x_2, y_2, radius2 = circle2

    dist_x, dist_y = x_2 - x_1, y_2 - y_1
    # Distance between circles centers
    distance = math.sqrt(dist_x ** 2 + dist_y ** 2)
    if distance > radius1 + radius2:
        # Circles too far
        return (0,)
    if distance < abs(radius2 - radius1):
        # One circle in the other
        return (0,)
    if distance <= tol and abs(radius1 - radius2) <= tol:
        # Same circle
        return 3, circle1

    dual = True
    if abs(abs(radius1 - distance) - radius2) <= tol:
        # Tangent circles
        dual = False

    # Distance from first circle's center to orthogonal projection
    # of circles intersections, on the axis between circles' centers
    mid_dist = (radius1 ** 2 - radius2 ** 2 + distance ** 2) / distance / 2

    # projected point is easy to compute now
    projected: Coord = (
        x_1 + (mid_dist * dist_x) / distance,
        y_1 + (mid_dist * dist_y) / distance,
    )

    if dual:
        return secant_circles_intersections(
            distance, dist_x, dist_y, mid_dist, radius1, projected
        )
    return 1, projected


def circle_line_from_points_intersection(
    circle: Circle,
    first_point: Coord,
    second_point: Coord,
) -> tuple[()] | tuple[Coord] | tuple[Coord, Coord]:
    """
    Intersection(s) of a circle and a line defined by two points.

    :param circle: Sequence of (abscissa, ordinate, radius).
    :param first_point: One point of the line.
    :param second_point: Another point on the line.

    :return: Either 0, 1 or two intersection points, the length indicates the intersection type.
    """
    # Move axis to circle center
    fp: Coord = first_point[0] - circle[0], first_point[1] - circle[1]
    sp: Coord = second_point[0] - circle[0], second_point[1] - circle[1]

    dx, dy = sp[0] - fp[0], sp[1] - fp[1]

    dr2 = dx ** 2 + dy ** 2

    cross = fp[0] * sp[1] - sp[0] * fp[1]

    discriminant = circle[2] ** 2 * dr2 - cross ** 2

    if discriminant < 0:
        # no intersection
        return ()

    reduced: Coord = cross / dr2, math.sqrt(discriminant) / dr2

    if discriminant == 0:
        # Tangent line
        return ((reduced[0] * dy + circle[0], -reduced[0] * dx + circle[1]),)

    # discriminant > 0, two intersections
    return (
        (
            reduced[0] * dy - (1 if dy >= 0 else -1) * dx * reduced[1] + circle[0],
            -reduced[0] * dx - abs(dy) * reduced[1] + circle[1],
        ),
        (
            reduced[0] * dy + (1 if dy >= 0 else -1) * dx * reduced[1] + circle[0],
            -reduced[0] * dx + abs(dy) * reduced[1] + circle[1],
        ),
    )


def circle_line_intersection(
    circle: Circle,
    line: Line,
) -> tuple[()] | tuple[Coord] | tuple[Coord, Coord]:
    """
    Return the intersection between a line and a circle.

    From https://mathworld.wolfram.com/Circle-LineIntersection.html

    Circle((x0,y0), r).intersection(Line(a*x+b*y+c)) # sympy

    :param circle: Sequence of (abscissa, ordinate, radius).
    :param line: Cartesian equation of a line (a, b, c) where ax + by + c = 0.

    :return: Nothing, one or two intersections. The length of the tuple gives the intersection type.
    """
    # Find two points on the line
    if line[1] != 0:
        first_point: Coord = 0, -line[2] / line[1]
        second_point: Coord = 1, -(line[2] + line[0]) / line[1]
    else:
        first_point = -line[2] / line[0], 0
        second_point = -(line[2] + line[1]) / line[0], 1

    return circle_line_from_points_intersection(circle, first_point, second_point)


def intersection(
    obj_1: Coord | Circle,
    obj_2: Coord | Circle,
    tol: float = 0.0,
) -> Coord | tuple[Coord, ...] | None:
    """Intersection of two arbitrary objects.

    The input objects should be points or circles.

    :param obj_1: First point or circle.
    :param obj_2: Second point or circle.
    :param tol: Absolute tolerance to use if provided. (Default value = 0.0).

    :returns: The intersection found, if any.
    """
    # Two points
    if len(obj_1) == 2 and len(obj_2) == 2:
        if obj_1 == obj_2 or (tol and math.dist(obj_1, obj_2) <= tol):
            return obj_1
        return None
    # Two circles
    if len(obj_1) == 3 and len(obj_2) == 3:
        return circle_intersect(obj_1, obj_2)[1:]  # type: ignore[return-value]
    # Point and circle
    if len(obj_1) == 2 and len(obj_2) == 3:
        if math.dist(obj_1, obj_2[:2]) - obj_2[2] <= tol:
            return obj_1
        return None
    # Circle and point
    if len(obj_1) == 3 and len(obj_2) == 2:
        return intersection(obj_1=obj_2, obj_2=obj_1, tol=tol)
    return None


def bounding_box(locus: list[Coord]) -> tuple[float, float, float, float]:
    """
    Compute the bounding box of a locus.

    :param locus: A list of points or any iterable with the same structure.

    :returns: Bounding box as (y_min, x_max, y_max, x_min).
    """
    y_min = float('inf')
    x_min = float('inf')
    y_max = -float('inf')
    x_max = -float('inf')
    for point in locus:
        y_min = min(y_min, point[1])
        x_min = min(x_min, point[0])
        y_max = max(y_max, point[1])
        x_max = max(x_max, point[0])
    return y_min, x_max, y_max, x_min
