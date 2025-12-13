"""Symbolic geometry functions using SymPy.

This module provides symbolic versions of the geometry functions from
pylinkage.geometry, enabling closed-form expressions for linkage analysis.
"""

import sympy as sp

from ._types import SymCoord


def symbolic_dist(
    x1: sp.Expr,
    y1: sp.Expr,
    x2: sp.Expr,
    y2: sp.Expr,
) -> sp.Expr:
    """
    Compute the symbolic Euclidean distance between two points.

    :param x1: X coordinate of first point.
    :param y1: Y coordinate of first point.
    :param x2: X coordinate of second point.
    :param y2: Y coordinate of second point.
    :returns: Symbolic expression for the distance.
    """
    return sp.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def symbolic_sqr_dist(
    x1: sp.Expr,
    y1: sp.Expr,
    x2: sp.Expr,
    y2: sp.Expr,
) -> sp.Expr:
    """
    Compute the symbolic squared distance between two points.

    Useful for avoiding square roots in constraint equations.

    :param x1: X coordinate of first point.
    :param y1: Y coordinate of first point.
    :param x2: X coordinate of second point.
    :param y2: Y coordinate of second point.
    :returns: Symbolic expression for the squared distance.
    """
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def symbolic_cyl_to_cart(
    radius: sp.Expr,
    angle: sp.Expr,
    ori_x: sp.Expr | float = 0,
    ori_y: sp.Expr | float = 0,
) -> SymCoord:
    """
    Convert polar coordinates to cartesian coordinates symbolically.

    :param radius: Distance from origin.
    :param angle: Angle in radians (counterclockwise from positive x-axis).
    :param ori_x: X coordinate of origin. Default is 0.
    :param ori_y: Y coordinate of origin. Default is 0.
    :returns: Tuple of (x, y) symbolic expressions.
    """
    ori_x = sp.sympify(ori_x)
    ori_y = sp.sympify(ori_y)
    return (
        ori_x + radius * sp.cos(angle),
        ori_y + radius * sp.sin(angle),
    )


def symbolic_circle_intersect(
    x1: sp.Expr,
    y1: sp.Expr,
    r1: sp.Expr,
    x2: sp.Expr,
    y2: sp.Expr,
    r2: sp.Expr,
    branch: int = 1,
) -> SymCoord:
    """
    Compute the symbolic intersection of two circles.

    This function returns a closed-form expression for one of the two
    intersection points of two circles. The branch parameter selects
    which intersection point to return.

    Algorithm (Paul Bourke, 1997):
    - d = distance between centers
    - a = (r1^2 - r2^2 + d^2) / (2*d) : distance from center1 to radical line
    - h = sqrt(r1^2 - a^2) : half-chord length
    - The intersection points are at distance h perpendicular to the line
      connecting the centers.

    :param x1: X coordinate of first circle center.
    :param y1: Y coordinate of first circle center.
    :param r1: Radius of first circle.
    :param x2: X coordinate of second circle center.
    :param y2: Y coordinate of second circle center.
    :param r2: Radius of second circle.
    :param branch: +1 or -1 to select which intersection point.
        Default is +1.
    :returns: Tuple of (x, y) symbolic expressions for the intersection.

    Note:
        The expressions may be complex (imaginary) for parameter values
        where the circles do not intersect. This is mathematically correct
        but will fail during numeric evaluation.
    """
    # Ensure all inputs are SymPy expressions
    x1, y1, r1 = sp.sympify(x1), sp.sympify(y1), sp.sympify(r1)
    x2, y2, r2 = sp.sympify(x2), sp.sympify(y2), sp.sympify(r2)

    # Distance between centers
    d = symbolic_dist(x1, y1, x2, y2)

    # Distance from center1 to the radical line (point of perpendicular)
    a = (r1**2 - r2**2 + d**2) / (2 * d)

    # Height from the line connecting centers to intersection points
    h = sp.sqrt(r1**2 - a**2)

    # Direction vector from center1 to center2 (normalized)
    dx = (x2 - x1) / d
    dy = (y2 - y1) / d

    # Projection point on the radical line
    px = x1 + a * dx
    py = y1 + a * dy

    # Perpendicular offset to get intersection points
    # branch = +1 gives one point, branch = -1 gives the other
    return (
        px + branch * h * (-dy),
        py + branch * h * dx,
    )


def symbolic_circle_line_intersect(
    cx: sp.Expr,
    cy: sp.Expr,
    r: sp.Expr,
    p1_x: sp.Expr,
    p1_y: sp.Expr,
    p2_x: sp.Expr,
    p2_y: sp.Expr,
    branch: int = 1,
) -> SymCoord:
    """
    Compute the symbolic intersection of a circle and a line.

    The line is defined by two points on it. The branch parameter
    selects which of the two intersection points to return.

    :param cx: X coordinate of circle center.
    :param cy: Y coordinate of circle center.
    :param r: Radius of the circle.
    :param p1_x: X coordinate of first point on line.
    :param p1_y: Y coordinate of first point on line.
    :param p2_x: X coordinate of second point on line.
    :param p2_y: Y coordinate of second point on line.
    :param branch: +1 or -1 to select which intersection point.
        Default is +1.
    :returns: Tuple of (x, y) symbolic expressions for the intersection.

    Note:
        The expressions may be complex (imaginary) for parameter values
        where the circle and line do not intersect.
    """
    # Ensure all inputs are SymPy expressions
    cx, cy, r = sp.sympify(cx), sp.sympify(cy), sp.sympify(r)
    p1_x, p1_y = sp.sympify(p1_x), sp.sympify(p1_y)
    p2_x, p2_y = sp.sympify(p2_x), sp.sympify(p2_y)

    # Translate to circle center
    fp_x = p1_x - cx
    fp_y = p1_y - cy
    sp_x = p2_x - cx
    sp_y = p2_y - cy

    # Direction vector
    dx = sp_x - fp_x
    dy = sp_y - fp_y

    # Squared length of direction
    dr2 = dx**2 + dy**2

    # Cross product (signed area)
    cross = fp_x * sp_y - sp_x * fp_y

    # Discriminant
    discriminant = r**2 * dr2 - cross**2

    # Reduced coordinates
    reduced_x = cross / dr2
    reduced_y = sp.sqrt(discriminant) / dr2

    # Sign function for dy (use sign for symbolic)
    sign_dy = sp.sign(dy)

    # Intersection points
    # Note: We use branch to select between + and - in reduced_y term
    if branch == 1:
        x = reduced_x * dy - sign_dy * dx * reduced_y + cx
        y = -reduced_x * dx - sp.Abs(dy) * reduced_y + cy
    else:
        x = reduced_x * dy + sign_dy * dx * reduced_y + cx
        y = -reduced_x * dx + sp.Abs(dy) * reduced_y + cy

    return (x, y)
