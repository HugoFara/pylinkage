"""Numba-compiled cam profile evaluation functions.

These functions are optimized for use in simulation hot loops.
All profile evaluation is done through precomputed parameters
to avoid Python object overhead in the simulation loop.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

# Profile type codes for dispatcher
PROFILE_HARMONIC = 0
PROFILE_CYCLOIDAL = 1
PROFILE_POLYNOMIAL = 2
PROFILE_MODIFIED_TRAPEZOIDAL = 3
PROFILE_SPLINE = 4


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_harmonic(u: float) -> float:
    """Evaluate simple harmonic motion law.

    s(u) = (1 - cos(pi * u)) / 2

    Smooth displacement but non-zero acceleration at boundaries.

    Args:
        u: Normalized angle in [0, 1].

    Returns:
        Normalized displacement in [0, 1].
    """
    return (1.0 - math.cos(math.pi * u)) / 2.0


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_harmonic_velocity(u: float) -> float:
    """Evaluate simple harmonic motion law velocity.

    ds/du = (pi / 2) * sin(pi * u)

    Args:
        u: Normalized angle in [0, 1].

    Returns:
        Normalized velocity.
    """
    return (math.pi / 2.0) * math.sin(math.pi * u)


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_harmonic_acceleration(u: float) -> float:
    """Evaluate simple harmonic motion law acceleration.

    d2s/du2 = (pi^2 / 2) * cos(pi * u)

    Args:
        u: Normalized angle in [0, 1].

    Returns:
        Normalized acceleration.
    """
    return (math.pi**2 / 2.0) * math.cos(math.pi * u)


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_cycloidal(u: float) -> float:
    """Evaluate cycloidal motion law.

    s(u) = u - sin(2 * pi * u) / (2 * pi)

    Zero velocity and acceleration at boundaries.

    Args:
        u: Normalized angle in [0, 1].

    Returns:
        Normalized displacement in [0, 1].
    """
    return u - math.sin(2.0 * math.pi * u) / (2.0 * math.pi)


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_cycloidal_velocity(u: float) -> float:
    """Evaluate cycloidal motion law velocity.

    ds/du = 1 - cos(2 * pi * u)

    Args:
        u: Normalized angle in [0, 1].

    Returns:
        Normalized velocity.
    """
    return 1.0 - math.cos(2.0 * math.pi * u)


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_cycloidal_acceleration(u: float) -> float:
    """Evaluate cycloidal motion law acceleration.

    d2s/du2 = 2 * pi * sin(2 * pi * u)

    Args:
        u: Normalized angle in [0, 1].

    Returns:
        Normalized acceleration.
    """
    return 2.0 * math.pi * math.sin(2.0 * math.pi * u)


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_polynomial(u: float, coeffs: np.ndarray) -> float:
    """Evaluate polynomial motion law.

    s(u) = sum(coeffs[i] * u^i)

    Args:
        u: Normalized angle in [0, 1].
        coeffs: Polynomial coefficients [a0, a1, a2, ...].

    Returns:
        Normalized displacement.
    """
    result = 0.0
    power = 1.0
    for i in range(len(coeffs)):
        result += coeffs[i] * power
        power *= u
    return result


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_polynomial_velocity(u: float, coeffs: np.ndarray) -> float:
    """Evaluate polynomial motion law velocity (derivative).

    ds/du = sum(i * coeffs[i] * u^(i-1))

    Args:
        u: Normalized angle in [0, 1].
        coeffs: Polynomial coefficients [a0, a1, a2, ...].

    Returns:
        Normalized velocity.
    """
    result = 0.0
    power = 1.0
    for i in range(1, len(coeffs)):
        result += i * coeffs[i] * power
        power *= u
    return result


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_modified_trapezoidal(u: float) -> float:
    """Evaluate modified trapezoidal motion law.

    Uses sinusoidal acceleration segments at start/end with
    constant acceleration in the middle. Provides lower peak
    acceleration than harmonic.

    Args:
        u: Normalized angle in [0, 1].

    Returns:
        Normalized displacement in [0, 1].
    """
    if u <= 0.125:
        # First acceleration segment (0 to 1/8)
        return (
            u / 4.0
            - math.sin(4.0 * math.pi * u) / (8.0 * math.pi)
        )
    elif u <= 0.375:
        # Constant acceleration segment (1/8 to 3/8)
        return 2.0 * u**2 - u / 4.0 + 1.0 / 32.0
    elif u <= 0.5:
        # Deceleration to constant velocity (3/8 to 1/2)
        return (
            u / 4.0
            + 1.0 / 4.0
            + math.sin(4.0 * math.pi * u - math.pi) / (8.0 * math.pi)
        )
    elif u <= 0.625:
        # Symmetric: deceleration from constant velocity (1/2 to 5/8)
        return (
            3.0 / 4.0
            - u / 4.0
            - math.sin(4.0 * math.pi * u - math.pi) / (8.0 * math.pi)
        )
    elif u <= 0.875:
        # Constant deceleration segment (5/8 to 7/8)
        return -2.0 * u**2 + 9.0 * u / 4.0 - 17.0 / 32.0
    else:
        # Final deceleration segment (7/8 to 1)
        return (
            3.0 / 4.0
            + u / 4.0
            + math.sin(4.0 * math.pi * u) / (8.0 * math.pi)
        )


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_modified_trapezoidal_velocity(u: float) -> float:
    """Evaluate modified trapezoidal motion law velocity.

    Args:
        u: Normalized angle in [0, 1].

    Returns:
        Normalized velocity.
    """
    if u <= 0.125:
        return 0.25 * (1.0 - math.cos(4.0 * math.pi * u))
    elif u <= 0.375:
        return 4.0 * u - 0.25
    elif u <= 0.625:
        # Constant velocity segment (0.5 to 0.625) and deceleration start (0.375 to 0.5)
        return 0.25 * (1.0 + math.cos(4.0 * math.pi * u - math.pi))
    elif u <= 0.875:
        return -4.0 * u + 2.25
    else:
        return 0.25 * (1.0 - math.cos(4.0 * math.pi * u))


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_cubic_spline(
    angle: float,
    angles: np.ndarray,
    coeffs: np.ndarray,
) -> float:
    """Evaluate precomputed cubic spline at angle.

    Each segment: r(t) = a + b*t + c*t^2 + d*t^3
    where t = (angle - angles[i]) / (angles[i+1] - angles[i])

    Args:
        angle: Cam rotation angle in radians.
        angles: Knot angles (sorted, ascending).
        coeffs: Spline coefficients, shape (n_segments, 4) for [a, b, c, d].

    Returns:
        Interpolated radius value.
    """
    n = len(angles) - 1

    # Handle periodicity - wrap angle to [0, 2*pi)
    two_pi = 2.0 * math.pi
    angle = angle % two_pi
    if angle < 0:
        angle += two_pi

    # Binary search for segment
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if angles[mid] <= angle:
            lo = mid
        else:
            hi = mid - 1

    i = lo

    # Handle wrap-around for last segment
    if i == n - 1 and angle >= angles[n - 1]:
        # Last segment may wrap to angles[0]
        h = (two_pi - angles[i]) + angles[0]
        if angle >= angles[i]:  # noqa: SIM108
            t = (angle - angles[i]) / h
        else:
            t = (two_pi - angles[i] + angle) / h
    else:
        h = angles[i + 1] - angles[i]
        t = (angle - angles[i]) / h

    # Evaluate cubic polynomial
    a = coeffs[i, 0]
    b = coeffs[i, 1]
    c = coeffs[i, 2]
    d = coeffs[i, 3]

    return a + t * (b + t * (c + t * d))


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_cubic_spline_derivative(
    angle: float,
    angles: np.ndarray,
    coeffs: np.ndarray,
) -> float:
    """Evaluate derivative of precomputed cubic spline at angle.

    dr/dt = b + 2*c*t + 3*d*t^2
    dr/dtheta = dr/dt / h

    Args:
        angle: Cam rotation angle in radians.
        angles: Knot angles (sorted, ascending).
        coeffs: Spline coefficients, shape (n_segments, 4) for [a, b, c, d].

    Returns:
        Derivative dr/dtheta.
    """
    n = len(angles) - 1
    two_pi = 2.0 * math.pi
    angle = angle % two_pi
    if angle < 0:
        angle += two_pi

    # Binary search for segment
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if angles[mid] <= angle:
            lo = mid
        else:
            hi = mid - 1

    i = lo

    # Compute h and t
    if i == n - 1 and angle >= angles[n - 1]:
        h = (two_pi - angles[i]) + angles[0]
        if angle >= angles[i]:  # noqa: SIM108
            t = (angle - angles[i]) / h
        else:
            t = (two_pi - angles[i] + angle) / h
    else:
        h = angles[i + 1] - angles[i]
        t = (angle - angles[i]) / h

    # Derivative coefficients
    b = coeffs[i, 1]
    c = coeffs[i, 2]
    d = coeffs[i, 3]

    # dr/dt
    dr_dt = b + t * (2.0 * c + 3.0 * d * t)

    # Convert to dr/dtheta
    return dr_dt / h


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_profile_displacement(
    angle: float,
    profile_type: int,
    base_radius: float,
    lift: float,
    rise_start: float,
    rise_end: float,
    dwell_high_end: float,
    fall_end: float,
    coeffs: np.ndarray,
) -> float:
    """Evaluate cam profile radius at given angle.

    This is the main dispatcher for profile evaluation in simulation.

    Args:
        angle: Cam rotation angle in radians.
        profile_type: Profile type code (PROFILE_HARMONIC, etc.).
        base_radius: Base circle radius.
        lift: Total follower lift (max displacement).
        rise_start: Angle where rise begins.
        rise_end: Angle where rise ends (dwell-high begins).
        dwell_high_end: Angle where dwell-high ends (fall begins).
        fall_end: Angle where fall ends (dwell-low begins).
        coeffs: Additional coefficients for polynomial/spline profiles.

    Returns:
        Cam radius at the given angle.
    """
    # Normalize angle to [0, 2*pi)
    two_pi = 2.0 * math.pi
    angle = angle % two_pi
    if angle < 0:
        angle += two_pi

    # Determine which phase we're in
    if angle < rise_start or angle >= fall_end:
        # Dwell low
        return base_radius

    elif angle < rise_end:
        # Rise phase
        u = (angle - rise_start) / (rise_end - rise_start)
        if profile_type == PROFILE_HARMONIC:
            s = evaluate_harmonic(u)
        elif profile_type == PROFILE_CYCLOIDAL:
            s = evaluate_cycloidal(u)
        elif profile_type == PROFILE_POLYNOMIAL:
            s = evaluate_polynomial(u, coeffs)
        elif profile_type == PROFILE_MODIFIED_TRAPEZOIDAL:
            s = evaluate_modified_trapezoidal(u)
        else:
            s = 0.0
        return base_radius + lift * s

    elif angle < dwell_high_end:
        # Dwell high
        return base_radius + lift

    else:
        # Fall phase
        u = (angle - dwell_high_end) / (fall_end - dwell_high_end)
        if profile_type == PROFILE_HARMONIC:
            s = 1.0 - evaluate_harmonic(u)
        elif profile_type == PROFILE_CYCLOIDAL:
            s = 1.0 - evaluate_cycloidal(u)
        elif profile_type == PROFILE_POLYNOMIAL:
            s = 1.0 - evaluate_polynomial(u, coeffs)
        elif profile_type == PROFILE_MODIFIED_TRAPEZOIDAL:
            s = 1.0 - evaluate_modified_trapezoidal(u)
        else:
            s = 1.0
        return base_radius + lift * s


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_spline_profile_displacement(
    angle: float,
    spline_angles: np.ndarray,
    spline_coeffs: np.ndarray,
) -> float:
    """Evaluate spline-based cam profile radius at given angle.

    Args:
        angle: Cam rotation angle in radians.
        spline_angles: Array of knot angles.
        spline_coeffs: Spline coefficients, shape (n_segments, 4) for [a, b, c, d].

    Returns:
        Cam radius at the given angle.
    """
    return evaluate_cubic_spline(angle, spline_angles, spline_coeffs)


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_spline_profile_derivative(
    angle: float,
    spline_angles: np.ndarray,
    spline_coeffs: np.ndarray,
) -> float:
    """Evaluate spline-based cam profile derivative at given angle.

    Args:
        angle: Cam rotation angle in radians.
        spline_angles: Array of knot angles.
        spline_coeffs: Spline coefficients, shape (n_segments, 4) for [a, b, c, d].

    Returns:
        Derivative dr/dtheta at the given angle.
    """
    return evaluate_cubic_spline_derivative(angle, spline_angles, spline_coeffs)


@njit(cache=True)  # type: ignore[untyped-decorator]
def evaluate_profile_derivative(
    angle: float,
    profile_type: int,
    base_radius: float,
    lift: float,
    rise_start: float,
    rise_end: float,
    dwell_high_end: float,
    fall_end: float,
    coeffs: np.ndarray,
) -> float:
    """Evaluate cam profile radius derivative (dr/dtheta) at given angle.

    Used for pressure angle and pitch curve calculations.

    Args:
        angle: Cam rotation angle in radians.
        profile_type: Profile type code.
        base_radius: Base circle radius (unused for derivative).
        lift: Total follower lift.
        rise_start: Angle where rise begins.
        rise_end: Angle where rise ends.
        dwell_high_end: Angle where dwell-high ends.
        fall_end: Angle where fall ends.
        coeffs: Additional coefficients for polynomial/spline profiles.

    Returns:
        Derivative dr/dtheta at the given angle.
    """
    two_pi = 2.0 * math.pi
    angle = angle % two_pi
    if angle < 0:
        angle += two_pi

    # Dwell phases have zero derivative
    if angle < rise_start or angle >= fall_end:
        return 0.0

    elif angle < rise_end:
        # Rise phase
        delta = rise_end - rise_start
        u = (angle - rise_start) / delta
        if profile_type == PROFILE_HARMONIC:
            ds_du = evaluate_harmonic_velocity(u)
        elif profile_type == PROFILE_CYCLOIDAL:
            ds_du = evaluate_cycloidal_velocity(u)
        elif profile_type == PROFILE_POLYNOMIAL:
            ds_du = evaluate_polynomial_velocity(u, coeffs)
        elif profile_type == PROFILE_MODIFIED_TRAPEZOIDAL:
            ds_du = evaluate_modified_trapezoidal_velocity(u)
        else:
            ds_du = 0.0
        # Chain rule: dr/dtheta = lift * ds/du * du/dtheta
        return lift * ds_du / delta

    elif angle < dwell_high_end:
        # Dwell high
        return 0.0

    else:
        # Fall phase (negative of rise derivative)
        delta = fall_end - dwell_high_end
        u = (angle - dwell_high_end) / delta
        if profile_type == PROFILE_HARMONIC:
            ds_du = -evaluate_harmonic_velocity(u)
        elif profile_type == PROFILE_CYCLOIDAL:
            ds_du = -evaluate_cycloidal_velocity(u)
        elif profile_type == PROFILE_POLYNOMIAL:
            ds_du = -evaluate_polynomial_velocity(u, coeffs)
        elif profile_type == PROFILE_MODIFIED_TRAPEZOIDAL:
            ds_du = -evaluate_modified_trapezoidal_velocity(u)
        else:
            ds_du = 0.0
        return lift * ds_du / delta


@njit(cache=True)  # type: ignore[untyped-decorator]
def compute_pitch_radius(
    cam_radius: float,
    cam_derivative: float,
    roller_radius: float,
) -> float:
    """Compute pitch curve radius for roller follower.

    The pitch curve is offset from the cam profile by the roller radius
    in the direction normal to the profile.

    For a translating radial follower:
    pitch_radius = cam_radius + roller_radius

    For more accurate contact, the offset depends on profile curvature,
    but this simple approximation works for most practical cases.

    Args:
        cam_radius: Cam profile radius r(theta).
        cam_derivative: dr/dtheta at this angle.
        roller_radius: Roller follower radius.

    Returns:
        Pitch curve radius (distance from cam center to roller center).
    """
    # For a radial translating follower, the pitch curve is simply
    # offset by the roller radius along the radial direction
    return cam_radius + roller_radius
