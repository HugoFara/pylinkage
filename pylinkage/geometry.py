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
    Return the intersections between two circles.

    Transcription of a Matt Woodhead program, method provided by Paul Bourke,
    1997. http://paulbourke.net/geometry/circlesphere/.

    Arguments
    ---------
    circle1 : (float, float, float)
        First circle, sequence of (abscisse, ordinate, radius)
    circle2 : (float, float, float)
        Ssecond circle, sequence of (abscisse, ordinate, radius)
    tol : float
        Distance under which two points are considered equal.

    Returns
    -------
    (int, None) or (int, squence of flaats)
    The first int gives the Intersection type, other values are the points of
    intersection.

    First int
        - 0: no intersection, second value is None
        - 1: on intersection (tangent circles), second value is the intersection
        point (float, float)
        - 2: two intersections, second value is the first intersection point
         (float, float), third value the second point (float, float)
        - 3: circles are the same one, second value is the (float, float, float)
        (the input circle.
    """
    x_1, y_1, radius1 = circle1
    x_2, y_2, radius2 = circle2

    dist_x, dist_y = x_2 - x_1, y_2 - y_1
    # Distance between circles centers
    distance = math.sqrt(dist_x ** 2 + dist_y ** 2)
    if distance > radius1 + radius2:
        # Circles two far
        return 0, None
    if distance < abs(radius2 - radius1):
        # One circle in the other
        return 0, None
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

    Circle((x0,y0), r).intersection(Line(a*x+b*y+c)) # sympy

    Arguments
    ---------
    circle : (float, float, float)
        Sequence of (abscisse, ordinate, radius)
    line : (float, float, float)
        Cartesian equation of a line.

    Returns
    -------
    (None, ) or ((float, float), ) or ((float, float), (float, float))
    The first int gives the intersection type

    """
    a, b, c = line
    x0, y0, r = circle
    discriminant = a ** 2 * r ** 2 - a ** 2 * x0 ** 2 - 2 * a * b * x0 * y0 - 2 * a * c * x0 + b ** 2 * r ** 2 - b ** 2 * y0 ** 2 - 2 * b * c * y0 - c ** 2
    A = a ** 2 + b ** 2
    if discriminant < 0:
        return tuple()
    if discriminant == 0:
        return tuple([(
            -(b * ( -(-a ** 2 * y0 + a * b * x0 + b * c) / A) + c) / a,
            - (-a ** 2 * y0 + a * b * x0 + b * c) / A
        )])

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