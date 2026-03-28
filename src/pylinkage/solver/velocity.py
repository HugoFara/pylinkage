"""Per-joint numba velocity solvers for kinematic analysis.

Each function computes the linear velocity of a specific joint type given
the parent velocities, positions, and constraints. All functions are
numba-compiled for maximum performance.

Velocity equations are derived by analytic differentiation of position
equations, using implicit differentiation for constraint-based joints.
"""

import math

from numba import njit


@njit(cache=True)  # type: ignore[untyped-decorator]
def solve_crank_velocity(
    x: float,
    y: float,
    anchor_x: float,
    anchor_y: float,
    anchor_vx: float,
    anchor_vy: float,
    radius: float,
    omega: float,
) -> tuple[float, float]:
    """Compute crank velocity using direct differentiation.

    The crank position is: (anchor + r*cos(θ), anchor + r*sin(θ))
    Differentiating: v = anchor_v + r*ω*(-sin(θ), cos(θ))

    Args:
        x: Current X position of the crank.
        y: Current Y position of the crank.
        anchor_x: X position of the anchor.
        anchor_y: Y position of the anchor.
        anchor_vx: X velocity of the anchor (usually 0).
        anchor_vy: Y velocity of the anchor (usually 0).
        radius: Distance from anchor to crank.
        omega: Angular velocity in rad/s.

    Returns:
        Velocity (vx, vy) of the crank.
    """
    if math.isnan(x) or math.isnan(y):
        return (math.nan, math.nan)

    theta = math.atan2(y - anchor_y, x - anchor_x)
    vx = anchor_vx - radius * omega * math.sin(theta)
    vy = anchor_vy + radius * omega * math.cos(theta)
    return (vx, vy)


@njit(cache=True)  # type: ignore[untyped-decorator]
def solve_revolute_velocity(
    x: float,
    y: float,
    p0_x: float,
    p0_y: float,
    p0_vx: float,
    p0_vy: float,
    p1_x: float,
    p1_y: float,
    p1_vx: float,
    p1_vy: float,
) -> tuple[float, float]:
    """Compute revolute joint velocity using implicit differentiation.

    The revolute joint satisfies two distance constraints:
        (x - p0_x)² + (y - p0_y)² = r0²
        (x - p1_x)² + (y - p1_y)² = r1²

    Differentiating with respect to time yields a 2x2 linear system:
        A * [vx, vy]ᵀ = b

    Where:
        A = [[x - p0_x, y - p0_y],
             [x - p1_x, y - p1_y]]
        b = [(x - p0_x)*p0_vx + (y - p0_y)*p0_vy,
             (x - p1_x)*p1_vx + (y - p1_y)*p1_vy]

    Args:
        x: Current X position of the joint.
        y: Current Y position of the joint.
        p0_x: X position of first parent.
        p0_y: Y position of first parent.
        p0_vx: X velocity of first parent.
        p0_vy: Y velocity of first parent.
        p1_x: X position of second parent.
        p1_y: Y position of second parent.
        p1_vx: X velocity of second parent.
        p1_vy: Y velocity of second parent.

    Returns:
        Velocity (vx, vy) of the joint, or (NaN, NaN) if singular.
    """
    if math.isnan(x) or math.isnan(y):
        return (math.nan, math.nan)

    # Jacobian matrix elements
    a11 = x - p0_x
    a12 = y - p0_y
    a21 = x - p1_x
    a22 = y - p1_y

    # Right-hand side
    b1 = a11 * p0_vx + a12 * p0_vy
    b2 = a21 * p1_vx + a22 * p1_vy

    # Determinant
    det = a11 * a22 - a12 * a21

    # Singularity check (joint collinear with parents = lock-up)
    if abs(det) < 1e-12:
        return (math.nan, math.nan)

    # Solve 2x2 system using Cramer's rule
    vx = (a22 * b1 - a12 * b2) / det
    vy = (a11 * b2 - a21 * b1) / det

    return (vx, vy)


@njit(cache=True)  # type: ignore[untyped-decorator]
def solve_fixed_velocity(
    x: float,
    y: float,
    p0_x: float,
    p0_y: float,
    p0_vx: float,
    p0_vy: float,
    p1_x: float,
    p1_y: float,
    p1_vx: float,
    p1_vy: float,
    radius: float,
    angle: float,
) -> tuple[float, float]:
    """Compute fixed joint velocity using direct differentiation.

    The fixed joint position is:
        x = p0_x + r * cos(α + γ)
        y = p0_y + r * sin(α + γ)

    Where α = atan2(p1_y - p0_y, p1_x - p0_x) is the base angle.

    Differentiating:
        dα/dt = [(p1_x - p0_x)(p1_vy - p0_vy) - (p1_y - p0_y)(p1_vx - p0_vx)] / d²
        vx = p0_vx - r * (dα/dt) * sin(α + γ)
        vy = p0_vy + r * (dα/dt) * cos(α + γ)

    Args:
        x: Current X position of the joint (unused, for consistency).
        y: Current Y position of the joint (unused, for consistency).
        p0_x: X position of first parent (origin).
        p0_y: Y position of first parent (origin).
        p0_vx: X velocity of first parent.
        p0_vy: Y velocity of first parent.
        p1_x: X position of second parent (angle reference).
        p1_y: Y position of second parent (angle reference).
        p1_vx: X velocity of second parent.
        p1_vy: Y velocity of second parent.
        radius: Distance from first parent.
        angle: Angle offset from the parent-to-parent direction.

    Returns:
        Velocity (vx, vy) of the joint, or (NaN, NaN) if singular.
    """
    if math.isnan(x) or math.isnan(y):
        return (math.nan, math.nan)

    # Vector from p0 to p1
    dx = p1_x - p0_x
    dy = p1_y - p0_y
    d_sq = dx * dx + dy * dy

    # Singularity: p0 and p1 coincide (undefined angle)
    if d_sq < 1e-24:
        return (math.nan, math.nan)

    # Base angle and its derivative
    base_angle = math.atan2(dy, dx)
    # d/dt[atan2(y,x)] = (x*dy/dt - y*dx/dt) / (x² + y²)
    rel_vx = p1_vx - p0_vx
    rel_vy = p1_vy - p0_vy
    d_base_angle_dt = (dx * rel_vy - dy * rel_vx) / d_sq

    # Total angle
    total_angle = angle + base_angle

    # Velocity
    vx = p0_vx - radius * d_base_angle_dt * math.sin(total_angle)
    vy = p0_vy + radius * d_base_angle_dt * math.cos(total_angle)

    return (vx, vy)


@njit(cache=True)  # type: ignore[untyped-decorator]
def solve_prismatic_velocity(
    x: float,
    y: float,
    circle_x: float,
    circle_y: float,
    circle_vx: float,
    circle_vy: float,
    radius: float,
    line_p1_x: float,
    line_p1_y: float,
    line_p1_vx: float,
    line_p1_vy: float,
    line_p2_x: float,
    line_p2_y: float,
    line_p2_vx: float,
    line_p2_vy: float,
) -> tuple[float, float]:
    """Compute prismatic joint velocity using implicit differentiation.

    The prismatic joint satisfies:
    1. Circle constraint: (x - cx)² + (y - cy)² = r²
    2. Line constraint: point lies on line through p1 and p2

    The line constraint in 2D:
        (x - p1_x)(p2_y - p1_y) - (y - p1_y)(p2_x - p1_x) = 0

    Differentiating both constraints yields a 2x2 linear system.

    Args:
        x: Current X position of the joint.
        y: Current Y position of the joint.
        circle_x: X position of circle center (revolute anchor).
        circle_y: Y position of circle center.
        circle_vx: X velocity of circle center.
        circle_vy: Y velocity of circle center.
        radius: Circle radius.
        line_p1_x: X position of first line point.
        line_p1_y: Y position of first line point.
        line_p1_vx: X velocity of first line point.
        line_p1_vy: Y velocity of first line point.
        line_p2_x: X position of second line point.
        line_p2_y: Y position of second line point.
        line_p2_vx: X velocity of second line point.
        line_p2_vy: Y velocity of second line point.

    Returns:
        Velocity (vx, vy) of the joint, or (NaN, NaN) if singular.
    """
    if math.isnan(x) or math.isnan(y):
        return (math.nan, math.nan)

    # Circle constraint derivative:
    # 2(x - cx)(vx - cvx) + 2(y - cy)(vy - cvy) = 0
    # => (x - cx)*vx + (y - cy)*vy = (x - cx)*cvx + (y - cy)*cvy
    a11 = x - circle_x
    a12 = y - circle_y
    b1 = a11 * circle_vx + a12 * circle_vy

    # Line constraint: (x - p1_x)(p2_y - p1_y) - (y - p1_y)(p2_x - p1_x) = 0
    # Let dx = p2_x - p1_x, dy = p2_y - p1_y
    # Constraint: (x - p1_x)*dy - (y - p1_y)*dx = 0
    #
    # Differentiating:
    # (vx - p1_vx)*dy + (x - p1_x)*(p2_vy - p1_vy)
    # - (vy - p1_vy)*dx - (y - p1_y)*(p2_vx - p1_vx) = 0
    #
    # Rearranging:
    # dy*vx - dx*vy = p1_vx*dy - p1_vy*dx
    #                 - (x - p1_x)*(p2_vy - p1_vy)
    #                 + (y - p1_y)*(p2_vx - p1_vx)
    dx = line_p2_x - line_p1_x
    dy = line_p2_y - line_p1_y
    d_dx = line_p2_vx - line_p1_vx
    d_dy = line_p2_vy - line_p1_vy

    a21 = dy
    a22 = -dx
    b2 = line_p1_vx * dy - line_p1_vy * dx - (x - line_p1_x) * d_dy + (y - line_p1_y) * d_dx

    # Determinant
    det = a11 * a22 - a12 * a21

    # Singularity check
    if abs(det) < 1e-12:
        return (math.nan, math.nan)

    # Solve 2x2 system using Cramer's rule
    vx = (a22 * b1 - a12 * b2) / det
    vy = (a11 * b2 - a21 * b1) / det

    return (vx, vy)
