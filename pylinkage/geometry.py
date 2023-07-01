#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The geometry module provides general geometry functions.

It is used extensively, so each function should be highly optimized.

Created on Wed May 5, 17:34:45 2021.

@author: HugoFara
"""
import math


def dist_builtin(point1, point2):
    """Euclidian distance between two 2D points.

    Legacy built-in unoptimized equivalent of math.dist in Python 3.8.

    :param point1:
    :param point2:

    """
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
    )


if hasattr(math, 'dist'):
    dist = math.dist
else:
    print('Unable to import dist from math. Using built-in function.')
    dist = dist_builtin


def sqr_dist(point1, point2):
    """Square of the distance between two points.

    Faster than dist.

    :param point1:
    :param point2:

    """
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2


def norm(vec):
    """

    :param vec:

    """
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)


def cyl_to_cart(radius, theta, ori=(0, 0)):
    """Convert polar coordinates into cartesian.

    :param radius: distance from ori
    :param theta: angle is the angle starting from abscissa axis
    :param ori: origin point (Default value = (0)).

    """
    return radius * math.cos(theta) + ori[0], radius * math.sin(theta) + ori[1]


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
    :returns: the intersections of two circles.
    Can be:
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


def line_from_points(point0, point1):
    """
    Return a cartesian equation of a line joining two points.

    Arguments
    ---------
    point0 : (float, float)
        On point of the line.
    point1 : (float, float)
        Another point on the line.

    Returns
    -------
    (float, float, float)
    A cartesian equation of this line.
    """
    director = (
        point1[0] - point0[0],
        point1[1] - point0[1]
    )
    # The barycenter should give more precision
    mean = (
        (point0[0] + point1[0]) / 2,
        (point0[1] + point1[1]) / 2
    )
    equilibrium = director[1] * mean[0] - director[0] * mean[1]
    return -director[1], director[0], equilibrium


def circle_line_intersection(circle, line):
    """
    Return the intersection between a line and a circle.

    Arguments
    ---------
    circle : (float, float, float)
        Sequence of (abscisse, ordinate, radius)
    line : (float, float, float)
        Cartesian equation of a line.

    Circle((x0,y0), r).intersection(Line(a*x+b*y+c)) # sympy
    """
    a, b, c = line
    x0, y0, r = circle
    discriminant = a ** 2 * r ** 2 - a ** 2 * x0 ** 2 - 2 * a * b * x0 * y0 - 2 * a * c * x0 + b ** 2 * r ** 2 - b ** 2 * y0 ** 2 - 2 * b * c * y0 - c ** 2
    A = a ** 2 + b ** 2
    if discriminant < 0:
        return tuple()
    if discriminant == 0:
        return tuple([
            -(b * ( -(-a ** 2 * y0 + a * b * x0 + b * c) / A) + c) / a,
            - (-a ** 2 * y0 + a * b * x0 + b * c) / A
        ])

    s_discri = math.sqrt(discriminant)
    return (
        (
            -(b * (a * s_discri / (a ** 2 + b ** 2) - (-a ** 2 * y0 + a * b * x0 + b * c) / (a ** 2 + b ** 2)) + c) / a,
            a * s_discri / (a ** 2 + b ** 2) - (-a ** 2 * y0 + a * b * x0 + b * c) / (a ** 2 + b ** 2)
         ),
        (
            -(b * (-a * s_discri / A - (-a ** 2 * y0 + a * b * x0 + b * c) / (a ** 2 + b ** 2)) + c) / a,
            -a * s_discri / (a ** 2 + b ** 2) - (-a ** 2 * y0 + a * b * x0 + b * c) / (a ** 2 + b ** 2)
        )
    )


    # We want to solve an equation of the type au²+bu+c=0
    a = sqr_dist(joint1.coord(), joint2.coord())
    b = 2 * (
        (joint2.x - joint1.x) * (joint1.x - joint0.x) +
        (joint2.y - joint1.y) * (joint1.y - joint0.y)
    )
    c = (
            self.joint0.x ** 2 + self.joint0.y ** 2 +
            self.joint1.x ** 2 + self.joint1.y ** 2 -
            2 * (
                    self.joint0.x * self.joint1.x +
                    self.joint0.y * self.joint1.y
            ) - self.distance0 ** 2
    )
    if b ** 2 < 4 * a * c:
        raise UnbuildableError(self)


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
