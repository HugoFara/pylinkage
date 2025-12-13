"""Per-joint numba solvers for the simulation loop.

Each function solves a specific joint type given the parent positions
and constraints. All functions are numba-compiled for maximum performance.
"""


import math

from numba import njit

from ..geometry.core import cyl_to_cart, get_nearest_point
from ..geometry.secants import (
    INTERSECTION_NONE,
    INTERSECTION_ONE,
    circle_intersect,
    circle_line_from_points_intersection,
)


@njit(cache=True)  # type: ignore[untyped-decorator]
def solve_crank(
    current_x: float,
    current_y: float,
    anchor_x: float,
    anchor_y: float,
    radius: float,
    angle_rate: float,
    dt: float,
) -> tuple[float, float]:
    """Solve crank position using polar rotation.

    The crank rotates around its anchor point at a constant angular rate.

    Args:
        current_x: Current X position of the crank.
        current_y: Current Y position of the crank.
        anchor_x: X position of the anchor (rotation center).
        anchor_y: Y position of the anchor (rotation center).
        radius: Distance from anchor to crank.
        angle_rate: Angular velocity (radians per unit time).
        dt: Time step.

    Returns:
        New (x, y) position of the crank.
    """
    current_angle = math.atan2(current_y - anchor_y, current_x - anchor_x)
    new_angle = current_angle + angle_rate * dt
    return cyl_to_cart(radius, new_angle, anchor_x, anchor_y)  # type: ignore[no-any-return]


@njit(cache=True)  # type: ignore[untyped-decorator]
def solve_revolute(
    current_x: float,
    current_y: float,
    p0_x: float,
    p0_y: float,
    r0: float,
    p1_x: float,
    p1_y: float,
    r1: float,
) -> tuple[float, float]:
    """Solve revolute joint using circle-circle intersection.

    The revolute joint is positioned at the intersection of two circles:
    - Circle 1: centered at parent 0 with radius r0
    - Circle 2: centered at parent 1 with radius r1

    When two solutions exist, picks the nearest to current position (hysteresis).

    Args:
        current_x: Current X position (for disambiguation).
        current_y: Current Y position (for disambiguation).
        p0_x: X position of first parent (circle 1 center).
        p0_y: Y position of first parent (circle 1 center).
        r0: Distance to first parent (circle 1 radius).
        p1_x: X position of second parent (circle 2 center).
        p1_y: Y position of second parent (circle 2 center).
        r1: Distance to second parent (circle 2 radius).

    Returns:
        New (x, y) position, or (NaN, NaN) if unbuildable.
    """
    result = circle_intersect(p0_x, p0_y, r0, p1_x, p1_y, r1)

    if result[0] == INTERSECTION_NONE:
        return (math.nan, math.nan)

    if result[0] == INTERSECTION_ONE:
        return (result[1], result[2])

    # Two or more solutions - pick nearest to current position
    return get_nearest_point(  # type: ignore[no-any-return]
        current_x, current_y, result[1], result[2], result[3], result[4]
    )


@njit(cache=True)  # type: ignore[untyped-decorator]
def solve_fixed(
    p0_x: float,
    p0_y: float,
    p1_x: float,
    p1_y: float,
    radius: float,
    angle: float,
) -> tuple[float, float]:
    """Solve fixed joint using polar projection.

    The fixed joint is at a fixed distance and angle from parent 0,
    with the angle measured relative to the line from parent 0 to parent 1.

    Args:
        p0_x: X position of first parent (origin).
        p0_y: Y position of first parent (origin).
        p1_x: X position of second parent (angle reference).
        p1_y: Y position of second parent (angle reference).
        radius: Distance from first parent.
        angle: Angle offset from the parent-to-parent direction.

    Returns:
        New (x, y) position (always deterministic).
    """
    base_angle = math.atan2(p1_y - p0_y, p1_x - p0_x)
    return cyl_to_cart(radius, angle + base_angle, p0_x, p0_y)  # type: ignore[no-any-return]


@njit(cache=True)  # type: ignore[untyped-decorator]
def solve_linear(
    current_x: float,
    current_y: float,
    circle_x: float,
    circle_y: float,
    radius: float,
    line_p1_x: float,
    line_p1_y: float,
    line_p2_x: float,
    line_p2_y: float,
) -> tuple[float, float]:
    """Solve linear joint using circle-line intersection.

    The linear joint slides along a line while maintaining a fixed distance
    from a revolute anchor:
    - Circle: centered at revolute anchor with given radius
    - Line: passing through line_p1 and line_p2

    When two solutions exist, picks the nearest to current position (hysteresis).

    Args:
        current_x: Current X position (for disambiguation).
        current_y: Current Y position (for disambiguation).
        circle_x: X position of revolute anchor (circle center).
        circle_y: Y position of revolute anchor (circle center).
        radius: Distance to revolute anchor (circle radius).
        line_p1_x: X position of first line-defining point.
        line_p1_y: Y position of first line-defining point.
        line_p2_x: X position of second line-defining point.
        line_p2_y: Y position of second line-defining point.

    Returns:
        New (x, y) position, or (NaN, NaN) if unbuildable.
    """
    result = circle_line_from_points_intersection(
        circle_x, circle_y, radius, line_p1_x, line_p1_y, line_p2_x, line_p2_y
    )

    if result[0] == INTERSECTION_NONE:
        return (math.nan, math.nan)

    if result[0] == INTERSECTION_ONE:
        return (result[1], result[2])

    # Two solutions - pick nearest to current position
    return get_nearest_point(  # type: ignore[no-any-return]
        current_x, current_y, result[1], result[2], result[3], result[4]
    )
