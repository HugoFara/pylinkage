#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The geometry module provides general geometry functions.

It is used extensively, so each function should be highly optimized.

Created on Wed May 5, 17:34:45 2021.

@author: HugoFara
"""
import math

from .core import dist


def secant_circles_intersections(
    distance, dist_x, dist_y, mid_dist, radius1, projected
):
    """Return the TWO intersections of secant circles."""
    # Distance between projected P and points
    # and the points of which P is projection
    height = math.sqrt(radius1 ** 2 - mid_dist ** 2) / distance
    inter1 = (
        projected[0] + height * dist_y,
        projected[1] - height * dist_x
    )
    inter2 = (
        projected[0] - height * dist_y,
        projected[1] + height * dist_x
    )
    return 2, inter1, inter2


def circle_intersect(circle1, circle2, tol=0.0):
    """
    Get the intersections of two circles.

    Transcription of a Matt Woodhead program, method provided by Paul Bourke,
    1997. http://paulbourke.net/geometry/circlesphere/.

    :param circle1: first circle
    :type circle1: tuple[float, float, float]
    :param circle2: second circle
    :type circle2: tuple[float, float, float]
    :param tol: distance under which two points are considered equal (Default value = 0.0)
    :type tol: float

    :returns: the intersections of two circles. Can be:

        - (0, ) when no intersection.
        - (1, (float, float)) if the intersection is one point.
        - (2, (float, float), (float, float)) for two points of intersection.
        - (3, (float, float, float)) if the intersection is a circle.

    :rtype: tuple[int] | tuple[int, tuple[float, float]] |
    tuple[int, tuple[float, float], tuple[float, float]] | tuple[int, tuple[float, float, float]]

    """
    x_1, y_1, radius1 = circle1
    x_2, y_2, radius2 = circle2

    dist_x, dist_y = x_2 - x_1, y_2 - y_1
    # Distance between circles centers
    distance = math.sqrt(dist_x ** 2 + dist_y ** 2)
    if distance > radius1 + radius2:
        # Circles two far
        return (0, )
    if distance < abs(radius2 - radius1):
        # One circle in the other
        return (0, )
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
    projected = (
        x_1 + (mid_dist * dist_x) / distance,
        y_1 + (mid_dist * dist_y) / distance
    )

    if dual:
        return secant_circles_intersections(
            distance, dist_x, dist_y, mid_dist, radius1, projected
        )
    return 1, projected


def circle_line_from_points_intersection(circle, first_point, second_point):
    """
    Intersection(s) of a circle and a line defined by two points.

    :param circle: Sequence of (abscissa, ordinate, radius)
    :type circle: tuple[float, float, float]
    :param tuple[float, float] first_point: One point of the line.
    :param tuple[float, float] second_point: Another point on the line.

    :return: Either 0, 1 or two intersection points, the first elements indicates the intersection type.
    :rtype: tuple | tuple[tuple[float, float]] | tuple[tuple[float, float], tuple[float, float]]
    """
    # Move axis to circle center
    first_point = first_point[0] - circle[0], first_point[1] - circle[1]
    second_point = second_point[0] - circle[0], second_point[1] - circle[1]

    dx, dy = second_point[0] - first_point[0], second_point[1] - first_point[1]

    dr2 = dx ** 2 + dy ** 2

    cross = first_point[0] * second_point[1] - second_point[0] * first_point[1]

    discriminant = circle[2] ** 2 * dr2 - cross ** 2

    if 0 > discriminant:
        # no intersection
        return tuple()

    reduced = cross / dr2, math.sqrt(discriminant) / dr2

    if discriminant == 0:
        # Tangent line
        return ((reduced[0] * dy + circle[0], -reduced[0] * dx + circle[1]), )

    # discriminant > 0, two intersections
    return (
        (
            reduced[0] * dy - (1 if dy >= 0 else -1) * dx * reduced[1] + circle[0],
            -reduced[0] * dx - abs(dy) * reduced[1] + circle[1]
        ),
        (
            reduced[0] * dy + (1 if dy >= 0 else -1) * dx * reduced[1] + circle[0],
            -reduced[0] * dx + abs(dy) * reduced[1] + circle[1]
        )
    )


def circle_line_intersection(circle, line):
    """
    Return the intersection between a line and a circle.

    From https://mathworld.wolfram.com/Circle-LineIntersection.html

    Circle((x0,y0), r).intersection(Line(a*x+b*y+c)) # sympy

    :param circle: Sequence of (abscissa, ordinate, radius)
    :type circle: tuple[float, float, float]
    :param line: Cartesian equation of a line.
    :type line: tuple[float, float, float]

    :return: Nothing, one or two intersections. The length of the tuple gives the intersection type.
    :rtype: tuple | tuple[tuple[float, float]] | tuple[tuple[float, float], tuple[float, float]]
    """
    # Find two points on the line
    if line[1] != 0:
        first_point = 0, - line[2] / line[1]
        second_point = 1, - (line[2] + line[0]) / line[1]
    else:
        first_point = - line[2] / line[0], 0
        second_point = - (line[2] + line[1]) / line[0], 1

    return circle_line_from_points_intersection(circle, first_point, second_point)


def intersection(obj_1, obj_2, tol=0.0):
    """Intersection of two arbitrary objects.

    The input objects should be points or circles.

    :param obj_1: First point or circle
    :type obj_1: tuple[float, float] | tuple[float, float, float]
    :param obj_2: Second point or circle
    :type obj_2: tuple[float, float] | tuple[float, float, float]
    :param tol: absolute tolerance to use if provided. (Default value = 0.0)
    :type tol: float

    :returns: The intersection found, if any
    :rtype: tuple[float, float] | tuple[float, float, float] | None
    """
    # Two points
    if len(obj_1) == 2 and len(obj_2) == 2:
        if obj_1 == obj_2 or tol and dist(obj_1, obj_2) <= tol:
            return obj_1
        return
    # Two circles
    if len(obj_1) == 3 and len(obj_2) == 3:
        return circle_intersect(obj_1, obj_2)[1:]
    # Point and circle
    if len(obj_1) == 2 and len(obj_2) == 3:
        if dist(obj_1, obj_2[:2]) - obj_2[2] <= tol:
            return obj_1
        return
    # Circle and point
    if len(obj_1) == 3 and len(obj_2) == 2:
        return intersection(obj_1=obj_2, obj_2=obj_1, tol=tol)


def bounding_box(locus):
    """
    Compute the bounding box of a locus.

    Parameters
    ----------
    locus : list[tuple[float]]
        A list of point or any iterable with the same structure.

    Returns
    -------
    tuple[float]
        Bounding box as (y_min, x_max, y_max, x_min).
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
