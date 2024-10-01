"""
Basic geometry features.
"""
import sys
import warnings
import math


def dist_builtin(point1, point2):
    """Euclidian distance between two 2D points.

    Legacy built-in unoptimized equivalent of `math.dist` in Python 3.8.

    :param tuple[float, float] point1: First point
    :param tuple[float, float] point2: Second point

    """
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
    )


if sys.version_info >= (3, 8, 0):
    dist = math.dist
else:
    warnings.warn('Unable to import dist from math. Using built-in function.')
    dist = dist_builtin


def sqr_dist(point1, point2):
    """
    Square of the distance between two points.

    Faster than dist.

    :param tuple[float, float] point1: First point to compare
    :param tuple[float, float] point2: Second point

    :return float: Computed distance
    """
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2


def get_nearest_point(reference_point, first_point, second_point):
    """
    Return the point closer to the reference.

    :param tuple[float, float] reference_point: Point to compare to
    :param tuple[float, float] first_point: First point candidate
    :param tuple[float, float] second_point: Second point candidate
    :return tuple[float, float]: Either first point or second point
    """
    if reference_point == first_point or reference_point == second_point:
        return reference_point
    if sqr_dist(reference_point, first_point) < sqr_dist(reference_point, second_point):
        return first_point
    return second_point


def norm(vec):
    """
    Return the norm of a 2-dimensional vector.

    :param tuple[float, float] vec: Vector to get norm from
    """
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)


def cyl_to_cart(radius, theta, ori=(0, 0)):
    """Convert polar coordinates into cartesian.

    :param radius: distance from ori
    :param theta: angle is the angle starting from abscissa axis
    :param ori: origin point (Default value = (0)).

    """
    return radius * math.cos(theta) + ori[0], radius * math.sin(theta) + ori[1]


def line_from_points(first_point, second_point):
    """
    A cartesian equation of the line joining two points.

    :param tuple[float, float] first_point: One point of the line.
    :param tuple[float, float] second_point: Another point on the line.
    :return tuple[float, float, float]: A cartesian equation of this line.
    """
    if first_point == second_point:
        warnings.warn("Cannot choose a line, inputs points are the same!")
        return 0, 0, 0
    director = (
        second_point[0] - first_point[0],
        second_point[1] - first_point[1]
    )
    # The barycenter should give more precision
    mean = (
        (first_point[0] + second_point[0]) / 2,
        (first_point[1] + second_point[1]) / 2
    )
    equilibrium = mean[0] * director[1] - mean[1] * director[0]
    return -director[1], director[0], equilibrium
