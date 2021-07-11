#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The geometry modules provides general geometry functions.

It is used extensively, so each function should be highly optimized.

Created on Wed May  5 17:34:45 2021.

@author: HugoFara
"""
import math

def dist_builtin(point1, point2):
    """
    Euclidian distance between two 2D points.

    Legacy built-in unoptimized equivalent of math.dist in Python 3.8.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2
                     + (point1[1] - point2[1]) ** 2)


if hasattr(math, 'dist'):
    dist = math.dist
else:
    print('Unable to import dist from math. Using built-in function.')
    dist = dist_builtin


def sqr_dist(point1, point2):
    """
    Square of the distance between two points.

    Faster than dist.
    """
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2


def norm(vec):
    """Return the norm of a 2-dimensional vector."""
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)


def cyl_to_cart(radius, theta, ori=(0, 0)):
    """
    Convert polar coordinates into cartesian.

    Arguments
    ---------
    radius: distance from ori
    theta: angle is the angle starting from abscisses axis
    ori: origin point.
    """
    return radius * math.cos(theta) + ori[0], radius * math.sin(theta) + ori[1]


def __secant_circles_intersections__(distance, dist_x, dist_y, mid_dist,
                                     radius1, projected):
    """Return the TWO intersections of secante circles."""
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
    Return the intersections between two circles.

    Transcription of a Matt Woodhead program, method provided by Paul Bourke,
    1997. http://paulbourke.net/geometry/circlesphere/.

    Arguments
    ---------
    circle1: first circle, sequence of (abscisse, ordinate, radius)
    circle2: second circle, sequence of (abscisse, ordinate, radius)
    tol: distance under which two points are considered equal.
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
        return __secant_circles_intersections__(
            distance, dist_x, dist_y, mid_dist, radius1, projected
        )
    return 1, projected


def intersection(obj_1, obj_2, tol=0.0):
    """
    Return intersection between two objects (points or circles).

    Return nothing of no intersection.
    tol: absolute tolerance to use if provided.
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
    return (y_min, x_max, y_max, x_min)