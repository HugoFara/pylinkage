"""
Basic geometry features with numba optimization.
"""


import math
from typing import TYPE_CHECKING

from numba import njit

if TYPE_CHECKING:
    pass


@njit(cache=True)  # type: ignore[untyped-decorator]
def dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Distance between two points.

    :param x1: X coordinate of first point.
    :param y1: Y coordinate of first point.
    :param x2: X coordinate of second point.
    :param y2: Y coordinate of second point.
    :return: Euclidean distance.
    """
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


@njit(cache=True)  # type: ignore[untyped-decorator]
def sqr_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Square of the distance between two points.

    Faster than dist when only comparing distances.

    :param x1: X coordinate of first point.
    :param y1: Y coordinate of first point.
    :param x2: X coordinate of second point.
    :param y2: Y coordinate of second point.
    :return: Squared distance.
    """
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy


@njit(cache=True)  # type: ignore[untyped-decorator]
def get_nearest_point(
    ref_x: float,
    ref_y: float,
    p1_x: float,
    p1_y: float,
    p2_x: float,
    p2_y: float,
) -> tuple[float, float]:
    """
    Return the point closer to the reference.

    :param ref_x: X coordinate of reference point.
    :param ref_y: Y coordinate of reference point.
    :param p1_x: X coordinate of first candidate.
    :param p1_y: Y coordinate of first candidate.
    :param p2_x: X coordinate of second candidate.
    :param p2_y: Y coordinate of second candidate.
    :return: Coordinates of the nearest point.
    """
    d1 = sqr_dist(ref_x, ref_y, p1_x, p1_y)
    d2 = sqr_dist(ref_x, ref_y, p2_x, p2_y)
    if d1 < d2:
        return (p1_x, p1_y)
    return (p2_x, p2_y)


@njit(cache=True)  # type: ignore[untyped-decorator]
def norm(x: float, y: float) -> float:
    """
    Return the norm of a 2-dimensional vector.

    :param x: X component.
    :param y: Y component.
    :return: Vector magnitude.
    """
    return math.sqrt(x * x + y * y)


@njit(cache=True)  # type: ignore[untyped-decorator]
def cyl_to_cart(
    radius: float,
    theta: float,
    ori_x: float = 0.0,
    ori_y: float = 0.0,
) -> tuple[float, float]:
    """Convert polar coordinates into cartesian.

    :param radius: Distance from origin.
    :param theta: Angle starting from abscissa axis.
    :param ori_x: Origin X coordinate (Default value = 0.0).
    :param ori_y: Origin Y coordinate (Default value = 0.0).
    :return: Cartesian coordinates (x, y).
    """
    return (radius * math.cos(theta) + ori_x, radius * math.sin(theta) + ori_y)


def line_from_points(
    first_x: float,
    first_y: float,
    second_x: float,
    second_y: float,
) -> tuple[float, float, float]:
    """
    A cartesian equation of the line joining two points.

    :param first_x: X coordinate of first point.
    :param first_y: Y coordinate of first point.
    :param second_x: X coordinate of second point.
    :param second_y: Y coordinate of second point.
    :return: A cartesian equation of this line (a, b, c) where ax + by + c = 0.
    """
    if first_x == second_x and first_y == second_y:
        return (0.0, 0.0, 0.0)
    director_x = second_x - first_x
    director_y = second_y - first_y
    # The barycenter should give more precision
    mean_x = (first_x + second_x) / 2
    mean_y = (first_y + second_y) / 2
    equilibrium = mean_x * director_y - mean_y * director_x
    return (-director_y, director_x, equilibrium)
