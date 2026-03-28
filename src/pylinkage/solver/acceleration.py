"""Per-joint numba acceleration solvers for kinematic analysis.

Each function computes the linear acceleration of a specific joint type given
parent accelerations, velocities, positions, and constraints. All functions are
numba-compiled for maximum performance.

Acceleration equations are derived by differentiating velocity equations,
including centripetal and Coriolis terms where applicable.
"""

import math

from .._numba_compat import njit


@njit(cache=True)  # type: ignore[untyped-decorator]
def solve_crank_acceleration(
    x: float,
    y: float,
    vx: float,
    vy: float,
    anchor_x: float,
    anchor_y: float,
    anchor_vx: float,
    anchor_vy: float,
    anchor_ax: float,
    anchor_ay: float,
    radius: float,
    omega: float,
    alpha: float,
) -> tuple[float, float]:
    """Compute crank acceleration including centripetal term.

    The crank velocity is: v = anchor_v + r*ω*(-sin(θ), cos(θ))
    Differentiating: a = anchor_a + r*α*(-sin(θ), cos(θ)) + r*ω²*(-cos(θ), -sin(θ))

    The second term is tangential acceleration, third is centripetal.

    Args:
        x: Current X position of the crank.
        y: Current Y position of the crank.
        vx: Current X velocity of the crank (unused, for consistency).
        vy: Current Y velocity of the crank (unused, for consistency).
        anchor_x: X position of the anchor.
        anchor_y: Y position of the anchor.
        anchor_vx: X velocity of the anchor (unused, for consistency).
        anchor_vy: Y velocity of the anchor (unused, for consistency).
        anchor_ax: X acceleration of the anchor (usually 0).
        anchor_ay: Y acceleration of the anchor (usually 0).
        radius: Distance from anchor to crank.
        omega: Angular velocity in rad/s.
        alpha: Angular acceleration in rad/s².

    Returns:
        Acceleration (ax, ay) of the crank.
    """
    if math.isnan(x) or math.isnan(y):
        return (math.nan, math.nan)

    theta = math.atan2(y - anchor_y, x - anchor_x)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    # Tangential: r * alpha * (-sin, cos)
    # Centripetal: -r * omega² * (cos, sin)
    ax = anchor_ax - radius * alpha * sin_theta - radius * omega * omega * cos_theta
    ay = anchor_ay + radius * alpha * cos_theta - radius * omega * omega * sin_theta

    return (ax, ay)


@njit(cache=True)  # type: ignore[untyped-decorator]
def solve_revolute_acceleration(
    x: float,
    y: float,
    vx: float,
    vy: float,
    p0_x: float,
    p0_y: float,
    p0_vx: float,
    p0_vy: float,
    p0_ax: float,
    p0_ay: float,
    p1_x: float,
    p1_y: float,
    p1_vx: float,
    p1_vy: float,
    p1_ax: float,
    p1_ay: float,
) -> tuple[float, float]:
    """Compute revolute joint acceleration using implicit differentiation.

    Starting from the velocity constraint equations and differentiating again
    with respect to time.

    From the constraint (x - p0_x)² + (y - p0_y)² = r0²:
    Velocity: (x - p0_x)(vx - p0_vx) + (y - p0_y)(vy - p0_vy) = 0
    Acceleration (differentiating velocity constraint):
    (vx - p0_vx)² + (vy - p0_vy)² + (x - p0_x)(ax - p0_ax) + (y - p0_y)(ay - p0_ay) = 0

    Rearranging for ax, ay gives a 2x2 linear system.

    Args:
        x: Current X position of the joint.
        y: Current Y position of the joint.
        vx: Current X velocity of the joint.
        vy: Current Y velocity of the joint.
        p0_x, p0_y: Position of first parent.
        p0_vx, p0_vy: Velocity of first parent.
        p0_ax, p0_ay: Acceleration of first parent.
        p1_x, p1_y: Position of second parent.
        p1_vx, p1_vy: Velocity of second parent.
        p1_ax, p1_ay: Acceleration of second parent.

    Returns:
        Acceleration (ax, ay) of the joint, or (NaN, NaN) if singular.
    """
    if math.isnan(x) or math.isnan(y) or math.isnan(vx) or math.isnan(vy):
        return (math.nan, math.nan)

    # Jacobian matrix elements (same as velocity)
    a11 = x - p0_x
    a12 = y - p0_y
    a21 = x - p1_x
    a22 = y - p1_y

    # Relative velocities
    rel_vx_0 = vx - p0_vx
    rel_vy_0 = vy - p0_vy
    rel_vx_1 = vx - p1_vx
    rel_vy_1 = vy - p1_vy

    # Right-hand side includes velocity squared terms and parent accelerations
    # From: (vx - p0_vx)² + (vy - p0_vy)² + (x - p0_x)(ax - p0_ax) + (y - p0_y)(ay - p0_ay) = 0
    # => a11*ax + a12*ay = a11*p0_ax + a12*p0_ay - rel_vx_0² - rel_vy_0²
    b1 = a11 * p0_ax + a12 * p0_ay - rel_vx_0 * rel_vx_0 - rel_vy_0 * rel_vy_0
    b2 = a21 * p1_ax + a22 * p1_ay - rel_vx_1 * rel_vx_1 - rel_vy_1 * rel_vy_1

    # Determinant
    det = a11 * a22 - a12 * a21

    # Singularity check
    if abs(det) < 1e-12:
        return (math.nan, math.nan)

    # Solve 2x2 system using Cramer's rule
    ax = (a22 * b1 - a12 * b2) / det
    ay = (a11 * b2 - a21 * b1) / det

    return (ax, ay)


@njit(cache=True)  # type: ignore[untyped-decorator]
def solve_fixed_acceleration(
    x: float,
    y: float,
    vx: float,
    vy: float,
    p0_x: float,
    p0_y: float,
    p0_vx: float,
    p0_vy: float,
    p0_ax: float,
    p0_ay: float,
    p1_x: float,
    p1_y: float,
    p1_vx: float,
    p1_vy: float,
    p1_ax: float,
    p1_ay: float,
    radius: float,
    angle: float,
) -> tuple[float, float]:
    """Compute fixed joint acceleration by differentiating velocity equations.

    The velocity is:
        vx = p0_vx - r * (dα/dt) * sin(α + γ)
        vy = p0_vy + r * (dα/dt) * cos(α + γ)

    Differentiating gives acceleration including angular acceleration term.

    Args:
        x: Current X position of the joint.
        y: Current Y position of the joint.
        vx: Current X velocity of the joint.
        vy: Current Y velocity of the joint.
        p0_x, p0_y: Position of first parent (origin).
        p0_vx, p0_vy: Velocity of first parent.
        p0_ax, p0_ay: Acceleration of first parent.
        p1_x, p1_y: Position of second parent (angle reference).
        p1_vx, p1_vy: Velocity of second parent.
        p1_ax, p1_ay: Acceleration of second parent.
        radius: Distance from first parent.
        angle: Angle offset from the parent-to-parent direction.

    Returns:
        Acceleration (ax, ay) of the joint, or (NaN, NaN) if singular.
    """
    if math.isnan(x) or math.isnan(y):
        return (math.nan, math.nan)

    # Vector from p0 to p1
    dx = p1_x - p0_x
    dy = p1_y - p0_y
    d_sq = dx * dx + dy * dy

    # Singularity: p0 and p1 coincide
    if d_sq < 1e-24:
        return (math.nan, math.nan)

    # Base angle
    base_angle = math.atan2(dy, dx)
    total_angle = angle + base_angle

    # Relative velocities
    rel_vx = p1_vx - p0_vx
    rel_vy = p1_vy - p0_vy

    # dα/dt = (dx * rel_vy - dy * rel_vx) / d_sq
    d_alpha_dt = (dx * rel_vy - dy * rel_vx) / d_sq

    # Relative accelerations
    rel_ax = p1_ax - p0_ax
    rel_ay = p1_ay - p0_ay

    # d²α/dt² by differentiating d_alpha_dt:
    # d_alpha_dt = (dx * rel_vy - dy * rel_vx) / d_sq
    # Need: d/dt[(dx * rel_vy - dy * rel_vx) / d_sq]

    # d(d_sq)/dt = 2*(dx*rel_vx + dy*rel_vy)
    d_dsq_dt = 2.0 * (dx * rel_vx + dy * rel_vy)

    # numerator = dx * rel_vy - dy * rel_vx
    # d(numerator)/dt = rel_vx * rel_vy + dx * rel_ay - rel_vy * rel_vx - dy * rel_ax
    #                 = dx * rel_ay - dy * rel_ax
    d_numer_dt = dx * rel_ay - dy * rel_ax

    # d²α/dt² = (d_numer_dt * d_sq - (dx * rel_vy - dy * rel_vx) * d_dsq_dt) / d_sq²
    numer = dx * rel_vy - dy * rel_vx
    d2_alpha_dt2 = (d_numer_dt * d_sq - numer * d_dsq_dt) / (d_sq * d_sq)

    # Acceleration: differentiate velocity
    # vx = p0_vx - r * d_alpha_dt * sin(total_angle)
    # ax = p0_ax - r*d2_alpha_dt2*sin(θ) - r*d_alpha_dt*cos(θ)*d_alpha_dt
    #    = p0_ax - r*d2_alpha_dt2*sin(θ) - r*d_alpha_dt²*cos(θ)
    sin_angle = math.sin(total_angle)
    cos_angle = math.cos(total_angle)

    ax = p0_ax - radius * d2_alpha_dt2 * sin_angle - radius * d_alpha_dt * d_alpha_dt * cos_angle
    ay = p0_ay + radius * d2_alpha_dt2 * cos_angle - radius * d_alpha_dt * d_alpha_dt * sin_angle

    return (ax, ay)


@njit(cache=True)  # type: ignore[untyped-decorator]
def solve_prismatic_acceleration(
    x: float,
    y: float,
    vx: float,
    vy: float,
    circle_x: float,
    circle_y: float,
    circle_vx: float,
    circle_vy: float,
    circle_ax: float,
    circle_ay: float,
    radius: float,
    line_p1_x: float,
    line_p1_y: float,
    line_p1_vx: float,
    line_p1_vy: float,
    line_p1_ax: float,
    line_p1_ay: float,
    line_p2_x: float,
    line_p2_y: float,
    line_p2_vx: float,
    line_p2_vy: float,
    line_p2_ax: float,
    line_p2_ay: float,
) -> tuple[float, float]:
    """Compute prismatic joint acceleration using implicit differentiation.

    Differentiates the velocity constraint equations to get acceleration.

    Args:
        x, y: Current position of the joint.
        vx, vy: Current velocity of the joint.
        circle_x, circle_y: Position of circle center.
        circle_vx, circle_vy: Velocity of circle center.
        circle_ax, circle_ay: Acceleration of circle center.
        radius: Circle radius.
        line_p1_x, line_p1_y: Position of first line point.
        line_p1_vx, line_p1_vy: Velocity of first line point.
        line_p1_ax, line_p1_ay: Acceleration of first line point.
        line_p2_x, line_p2_y: Position of second line point.
        line_p2_vx, line_p2_vy: Velocity of second line point.
        line_p2_ax, line_p2_ay: Acceleration of second line point.

    Returns:
        Acceleration (ax, ay) of the joint, or (NaN, NaN) if singular.
    """
    if math.isnan(x) or math.isnan(y) or math.isnan(vx) or math.isnan(vy):
        return (math.nan, math.nan)

    # Circle constraint: (x - cx)² + (y - cy)² = r²
    # Velocity: (x - cx)(vx - cvx) + (y - cy)(vy - cvy) = 0
    # Acceleration: (vx - cvx)² + (vy - cvy)² + (x - cx)(ax - cax) + (y - cy)(ay - cay) = 0

    a11 = x - circle_x
    a12 = y - circle_y

    rel_vx_c = vx - circle_vx
    rel_vy_c = vy - circle_vy

    b1 = a11 * circle_ax + a12 * circle_ay - rel_vx_c * rel_vx_c - rel_vy_c * rel_vy_c

    # Line constraint derivatives
    dx = line_p2_x - line_p1_x
    dy = line_p2_y - line_p1_y
    d_dx = line_p2_vx - line_p1_vx
    d_dy = line_p2_vy - line_p1_vy
    d2_dx = line_p2_ax - line_p1_ax
    d2_dy = line_p2_ay - line_p1_ay

    a21 = dy
    a22 = -dx

    # Line velocity constraint was:
    # dy*vx - dx*vy = p1_vx*dy - p1_vy*dx - (x - p1_x)*d_dy + (y - p1_y)*d_dx
    # Differentiating:
    # d_dy*vx + dy*ax - d_dx*vy - dx*ay = ...

    # Full differentiation of RHS:
    # d/dt[p1_vx*dy] = p1_ax*dy + p1_vx*d_dy
    # d/dt[p1_vy*dx] = p1_ay*dx + p1_vy*d_dx
    # d/dt[(x - p1_x)*d_dy] = (vx - p1_vx)*d_dy + (x - p1_x)*d2_dy
    # d/dt[(y - p1_y)*d_dx] = (vy - p1_vy)*d_dx + (y - p1_y)*d2_dx

    b2 = (
        d_dy * vx
        - d_dx * vy
        + line_p1_ax * dy
        + line_p1_vx * d_dy
        - line_p1_ay * dx
        - line_p1_vy * d_dx
        - (vx - line_p1_vx) * d_dy
        - (x - line_p1_x) * d2_dy
        + (vy - line_p1_vy) * d_dx
        + (y - line_p1_y) * d2_dx
    )

    # Adjustments for our matrix form: a21*ax + a22*ay = b2_adjusted
    # From: d_dy*vx + dy*ax - d_dx*vy - dx*ay = b2
    # => dy*ax - dx*ay = b2 - d_dy*vx + d_dx*vy
    b2_adjusted = b2 - d_dy * vx + d_dx * vy

    # Determinant
    det = a11 * a22 - a12 * a21

    if abs(det) < 1e-12:
        return (math.nan, math.nan)

    # Solve 2x2 system
    ax = (a22 * b1 - a12 * b2_adjusted) / det
    ay = (a11 * b2_adjusted - a21 * b1) / det

    return (ax, ay)
