"""
Basic geometry features.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._types import Coord, Line


dist = math.dist


def sqr_dist(point1: Coord, point2: Coord) -> float:
    """
    Square of the distance between two points.

    Faster than dist.

    :param point1: First point to compare.
    :param point2: Second point.

    :return: Computed squared distance.

    Examples:
        >>> sqr_dist((0, 0), (3, 4))
        25
        >>> sqr_dist((1, 1), (1, 1))
        0
        >>> sqr_dist((0, 0), (1, 1))
        2
    """
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2


def get_nearest_point(
    reference_point: Coord,
    first_point: Coord,
    second_point: Coord,
) -> Coord:
    """
    Return the point closer to the reference.

    :param reference_point: Point to compare to.
    :param first_point: First point candidate.
    :param second_point: Second point candidate.
    :return: Either first point or second point.

    Examples:
        >>> get_nearest_point((0, 0), (1, 0), (5, 0))
        (1, 0)
        >>> get_nearest_point((0, 0), (3, 4), (0, 1))
        (0, 1)
        >>> get_nearest_point((1, 1), (1, 1), (2, 2))  # reference matches first
        (1, 1)
    """
    if reference_point in (first_point, second_point):
        return reference_point
    if sqr_dist(reference_point, first_point) < sqr_dist(reference_point, second_point):
        return first_point
    return second_point


def norm(vec: Coord) -> float:
    """
    Return the norm of a 2-dimensional vector.

    :param vec: Vector to get norm from.

    Examples:
        >>> norm((3, 4))
        5.0
        >>> norm((0, 0))
        0.0
        >>> norm((1, 0))
        1.0
    """
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)


def cyl_to_cart(
    radius: float,
    theta: float,
    ori: Coord = (0, 0),
) -> Coord:
    """Convert polar coordinates into cartesian.

    :param radius: Distance from ori.
    :param theta: Angle starting from abscissa axis.
    :param ori: Origin point (Default value = (0, 0)).

    Examples:
        >>> import math
        >>> cyl_to_cart(1, 0)  # Point at angle 0
        (1.0, 0.0)
        >>> x, y = cyl_to_cart(1, math.pi / 2)  # Point at 90 degrees
        >>> (round(x, 10), round(y, 10))
        (0.0, 1.0)
        >>> cyl_to_cart(2, 0, ori=(1, 1))  # With offset origin
        (3.0, 1.0)
    """
    return radius * math.cos(theta) + ori[0], radius * math.sin(theta) + ori[1]


def line_from_points(first_point: Coord, second_point: Coord) -> Line:
    """
    A cartesian equation of the line joining two points.

    :param first_point: One point of the line.
    :param second_point: Another point on the line.
    :return: A cartesian equation of this line (a, b, c) where ax + by + c = 0.

    Examples:
        >>> line_from_points((0, 0), (1, 0))  # Horizontal line y=0
        (0, 1, 0.0)
        >>> line_from_points((0, 0), (0, 1))  # Vertical line x=0
        (-1, 0, 0.0)
        >>> line_from_points((0, 0), (1, 1))  # Diagonal line y=x
        (-1, 1, 0.0)
        >>> line_from_points((0, 1), (1, 1))  # Horizontal line y=1
        (0, 1, -1.0)
    """
    if first_point == second_point:
        warnings.warn(
            "Cannot choose a line, inputs points are the same!",
            stacklevel=2,
        )
        return 0, 0, 0
    director = (
        second_point[0] - first_point[0],
        second_point[1] - first_point[1],
    )
    # The barycenter should give more precision
    mean = (
        (first_point[0] + second_point[0]) / 2,
        (first_point[1] + second_point[1]) / 2,
    )
    equilibrium = mean[0] * director[1] - mean[1] * director[0]
    return -director[1], director[0], equilibrium
