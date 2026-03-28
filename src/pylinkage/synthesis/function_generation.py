"""Function generation synthesis for four-bar linkages.

Function generation synthesizes a linkage where the input crank angle
(theta2) produces a desired output rocker angle (theta4). This is based
on Freudenstein's equation, which relates the input/output angles to
the link length ratios.

The Freudenstein equation:
    R1*cos(theta4) - R2*cos(theta2) + R3 - cos(theta2 - theta4) = 0

Where:
    R1 = d/a  (ground/crank ratio)
    R2 = d/c  (ground/rocker ratio)
    R3 = (a^2 - b^2 + c^2 + d^2) / (2*a*c)

And a, b, c, d are crank, coupler, rocker, ground lengths respectively.

Common applications:
- Mechanical computing devices
- Coordinated motion mechanisms
- Transfer functions in machinery

References:
    - Freudenstein, F. "Approximate Synthesis of Four-Bar Linkages" (1955)
    - Sandor & Erdman, Chapter 3: "Analytical Linkage Synthesis"
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg as scipy_linalg

from ._types import AnglePair, FourBarSolution, Point2D, SynthesisType
from .core import SynthesisProblem, SynthesisResult
from .utils import grashof_check, validate_fourbar

if TYPE_CHECKING:
    from ..linkage import Linkage


def freudenstein_equation(
    theta2: float,
    theta4: float,
    R1: float,
    R2: float,
    R3: float,
) -> float:
    """Evaluate Freudenstein's equation.

    The Freudenstein equation relates input angle theta2 (crank) to
    output angle theta4 (rocker) through the link length ratios R1, R2, R3.

    Args:
        theta2: Input crank angle in radians.
        theta4: Output rocker angle in radians.
        R1: d/a ratio (ground/crank).
        R2: d/c ratio (ground/rocker).
        R3: (a^2 - b^2 + c^2 + d^2) / (2*a*c) ratio.

    Returns:
        Residual of the Freudenstein equation (0 if satisfied exactly).
    """
    return R1 * math.cos(theta4) - R2 * math.cos(theta2) + R3 - math.cos(theta2 - theta4)


def solve_freudenstein_3_positions(
    angle_pairs: list[AnglePair],
) -> tuple[float, float, float]:
    """Solve for Freudenstein coefficients from 3 precision positions.

    With exactly 3 angle pairs, we have a linear system of 3 equations
    in 3 unknowns (R1, R2, R3). This gives an exact solution.

    Args:
        angle_pairs: Exactly 3 (theta2, theta4) pairs in radians.

    Returns:
        Tuple (R1, R2, R3) of Freudenstein coefficients.

    Raises:
        ValueError: If not exactly 3 angle pairs provided.
        scipy.linalg.LinAlgError: If the system is singular.

    Example:
        >>> pairs = [(0.0, 0.0), (0.5, 0.6), (1.0, 1.1)]
        >>> R1, R2, R3 = solve_freudenstein_3_positions(pairs)
    """
    if len(angle_pairs) != 3:
        raise ValueError(f"Need exactly 3 angle pairs, got {len(angle_pairs)}")

    # Build coefficient matrix A and RHS b for A @ [R1, R2, R3].T = b
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)

    for i, (theta2, theta4) in enumerate(angle_pairs):
        A[i, 0] = math.cos(theta4)
        A[i, 1] = -math.cos(theta2)
        A[i, 2] = 1.0
        b[i] = math.cos(theta2 - theta4)

    # Check condition number
    cond = np.linalg.cond(A)
    if cond > 1e12:
        raise scipy_linalg.LinAlgError(
            f"System is ill-conditioned (cond={cond:.2e}). "
            "Angle pairs may be too similar or degenerate."
        )

    # Solve linear system
    R = scipy_linalg.solve(A, b)

    return float(R[0]), float(R[1]), float(R[2])


def solve_freudenstein_least_squares(
    angle_pairs: list[AnglePair],
) -> tuple[float, float, float, float]:
    """Solve over-determined Freudenstein system via least squares.

    For more than 3 angle pairs, find the coefficients that minimize
    the sum of squared residuals.

    Args:
        angle_pairs: List of (theta2, theta4) pairs (4 or more).

    Returns:
        Tuple (R1, R2, R3, residual_norm) where residual_norm is the
        fitting error.

    Raises:
        ValueError: If fewer than 3 angle pairs provided.
    """
    n = len(angle_pairs)
    if n < 3:
        raise ValueError(f"Need at least 3 angle pairs, got {n}")

    A = np.zeros((n, 3), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    for i, (theta2, theta4) in enumerate(angle_pairs):
        A[i, 0] = math.cos(theta4)
        A[i, 1] = -math.cos(theta2)
        A[i, 2] = 1.0
        b[i] = math.cos(theta2 - theta4)

    # Least squares solution
    R, residues, rank, s = scipy_linalg.lstsq(A, b)

    # Compute residual norm
    residuals = A @ R - b
    residual_norm = float(scipy_linalg.norm(residuals))

    return float(R[0]), float(R[1]), float(R[2]), residual_norm


def coefficients_to_link_lengths(
    R1: float,
    R2: float,
    R3: float,
    ground_length: float = 1.0,
) -> tuple[float, float, float, float]:
    """Convert Freudenstein coefficients to link lengths.

    Given R1, R2, R3 and choosing ground length d, solve for the
    link lengths a (crank), b (coupler), c (rocker).

    The relationships are:
        R1 = d/a  =>  a = d/R1
        R2 = d/c  =>  c = d/R2
        R3 = (a^2 - b^2 + c^2 + d^2) / (2*a*c)
            =>  b^2 = a^2 + c^2 + d^2 - 2*a*c*R3

    Args:
        R1: d/a ratio.
        R2: d/c ratio.
        R3: Combined ratio.
        ground_length: Desired ground link length d.

    Returns:
        Tuple (crank_length, coupler_length, rocker_length, ground_length).

    Raises:
        ValueError: If coefficients produce invalid link lengths.
    """
    d = ground_length

    # Check for valid R1, R2 (must be non-zero)
    if abs(R1) < 1e-12:
        raise ValueError(f"R1={R1} is too small; would give infinite crank length")
    if abs(R2) < 1e-12:
        raise ValueError(f"R2={R2} is too small; would give infinite rocker length")

    a = d / R1  # crank
    c = d / R2  # rocker

    # Solve for coupler length b
    b_squared = a**2 + c**2 + d**2 - 2.0 * a * c * R3

    if b_squared < 0:
        raise ValueError(
            f"Invalid coefficients: coupler length squared = {b_squared:.6f} < 0. "
            "The specified angle pairs may not be achievable by a physical 4-bar."
        )

    b = math.sqrt(b_squared)

    # Check for valid positive lengths
    if a <= 0:
        raise ValueError(f"Computed crank length {a:.6f} <= 0 (R1 has wrong sign)")
    if c <= 0:
        raise ValueError(f"Computed rocker length {c:.6f} <= 0 (R2 has wrong sign)")

    return (a, b, c, d)


def _compute_initial_joint_positions(
    ground_pivot_a: Point2D,
    crank_length: float,
    coupler_length: float,
    rocker_length: float,
    ground_length: float,
    initial_crank_angle: float,
    initial_rocker_angle: float,
) -> tuple[Point2D, Point2D, Point2D, Point2D]:
    """Compute initial joint positions for a four-bar.

    Args:
        ground_pivot_a: Position of crank ground pivot A.
        crank_length: Length a.
        coupler_length: Length b.
        rocker_length: Length c.
        ground_length: Length d.
        initial_crank_angle: Initial angle of crank from horizontal.
        initial_rocker_angle: Initial angle of rocker from horizontal.

    Returns:
        Tuple (A, D, B, C) of joint positions.

    Raises:
        ValueError: If no valid coupler position exists.
    """
    from ..geometry.secants import circle_intersect

    A = ground_pivot_a

    # Ground pivot D is to the right of A by ground_length
    D = (A[0] + ground_length, A[1])

    # Crank pivot B
    B = (
        A[0] + crank_length * math.cos(initial_crank_angle),
        A[1] + crank_length * math.sin(initial_crank_angle),
    )

    # Coupler pivot C must satisfy both constraints:
    # - Distance from B = coupler_length
    # - Distance from D = rocker_length
    # Use circle-circle intersection to find valid C position
    n_intersections, x1, y1, x2, y2 = circle_intersect(
        B[0], B[1], coupler_length, D[0], D[1], rocker_length
    )

    if n_intersections == 0:
        # No valid coupler position - linkage is unbuildable
        raise ValueError(
            f"Cannot assemble four-bar at angles ({initial_crank_angle:.3f}, "
            f"{initial_rocker_angle:.3f}). Coupler circles do not intersect."
        )

    # Choose the solution that best matches the expected rocker angle
    # Standard convention: θ4 is angle from ground link (positive x) to rocker (DC)
    # So C = D + rocker_length * (cos(θ4), sin(θ4))
    expected_C = (
        D[0] + rocker_length * math.cos(initial_rocker_angle),
        D[1] + rocker_length * math.sin(initial_rocker_angle),
    )

    if n_intersections == 1:
        C = (x1, y1)
    else:
        # Choose the intersection point closer to expected position
        dist1 = (x1 - expected_C[0]) ** 2 + (y1 - expected_C[1]) ** 2
        dist2 = (x2 - expected_C[0]) ** 2 + (y2 - expected_C[1]) ** 2
        C = (x1, y1) if dist1 <= dist2 else (x2, y2)

    return A, D, B, C


def function_generation(
    angle_pairs: list[AnglePair],
    ground_length: float = 1.0,
    ground_pivot_a: Point2D = (0.0, 0.0),
    require_grashof: bool = True,
    require_crank_rocker: bool = False,
) -> SynthesisResult:
    """Synthesize a four-bar for function generation.

    Given input/output angle pairs, find four-bar linkages where
    the crank angle theta2 produces rocker angle theta4 according
    to the Freudenstein equation.

    For 3 angle pairs: Unique exact solution (if physically valid)
    For 4+ angle pairs: Least-squares approximate solution

    Args:
        angle_pairs: List of (input_angle, output_angle) in radians.
            At least 3 pairs required.
        ground_length: Desired ground link length (scaling factor).
        ground_pivot_a: Position of input (crank) ground pivot.
        require_grashof: If True, reject non-Grashof solutions.
        require_crank_rocker: If True, only accept crank-rocker type.

    Returns:
        SynthesisResult containing valid linkages.

    Raises:
        ValueError: If fewer than 3 angle pairs provided.

    Example:
        >>> pairs = [(0, 0), (math.pi/6, math.pi/4), (math.pi/3, math.pi/2)]
        >>> result = function_generation(pairs)
        >>> for linkage in result.solutions:
        ...     print(f"Crank: {linkage.joints[2].r:.3f}")
    """
    problem = SynthesisProblem(
        synthesis_type=SynthesisType.FUNCTION,
        angle_pairs=list(angle_pairs),
        ground_pivot_a=ground_pivot_a,
        ground_length=ground_length,
    )

    warnings: list[str] = []
    raw_solutions: list[FourBarSolution] = []

    n = len(angle_pairs)

    if n < 3:
        raise ValueError(f"Function generation needs at least 3 angle pairs, got {n}")

    try:
        if n == 3:
            R1, R2, R3 = solve_freudenstein_3_positions(angle_pairs)
            residual = 0.0
        else:
            R1, R2, R3, residual = solve_freudenstein_least_squares(angle_pairs)
            if residual > 0.01:
                warnings.append(
                    f"Least squares fit residual = {residual:.4f}. "
                    "The specified angles may not be exactly achievable."
                )

        # Convert to link lengths
        try:
            crank, coupler, rocker, ground = coefficients_to_link_lengths(R1, R2, R3, ground_length)
        except ValueError as e:
            warnings.append(str(e))
            return SynthesisResult(
                solutions=[],
                raw_solutions=[],
                problem=problem,
                warnings=warnings,
            )

        # Check Grashof criterion
        grashof_type = grashof_check(crank, coupler, rocker, ground)

        if require_crank_rocker:
            from .utils import GrashofType

            if grashof_type not in (
                GrashofType.GRASHOF_CRANK_ROCKER,
                GrashofType.CHANGE_POINT,
            ):
                warnings.append(f"Solution is not crank-rocker type (got {grashof_type.name})")
                return SynthesisResult(
                    solutions=[],
                    raw_solutions=[],
                    problem=problem,
                    warnings=warnings,
                )

        if require_grashof:
            from .utils import GrashofType

            if grashof_type == GrashofType.NON_GRASHOF:
                warnings.append(
                    "Solution is non-Grashof (no link can fully rotate). "
                    "Set require_grashof=False to accept."
                )
                return SynthesisResult(
                    solutions=[],
                    raw_solutions=[],
                    problem=problem,
                    warnings=warnings,
                )

        # Compute joint positions at first precision position
        theta2_0, theta4_0 = angle_pairs[0]

        try:
            A, D, B, C = _compute_initial_joint_positions(
                ground_pivot_a,
                crank,
                coupler,
                rocker,
                ground,
                theta2_0,
                theta4_0,
            )
        except ValueError as pos_err:
            warnings.append(str(pos_err))
            return SynthesisResult(
                solutions=[],
                raw_solutions=[],
                problem=problem,
                warnings=warnings,
            )

        raw_solution = FourBarSolution(
            ground_pivot_a=A,
            ground_pivot_d=D,
            crank_pivot_b=B,
            coupler_pivot_c=C,
            crank_length=crank,
            coupler_length=coupler,
            rocker_length=rocker,
            ground_length=ground,
        )

        # Validate solution
        is_valid, errors = validate_fourbar(raw_solution)
        if not is_valid:
            warnings.extend(errors)
        else:
            raw_solutions.append(raw_solution)

    except (np.linalg.LinAlgError, scipy_linalg.LinAlgError) as e:
        warnings.append(f"Linear algebra error: {e}")

    # Convert raw solutions to Linkage objects
    from .conversion import solutions_to_linkages

    linkages = solutions_to_linkages(raw_solutions, SynthesisType.FUNCTION)

    return SynthesisResult(
        solutions=linkages,
        raw_solutions=raw_solutions,
        problem=problem,
        warnings=warnings,
    )


def verify_function_generation(
    linkage: Linkage,
    angle_pairs: list[AnglePair],
    tolerance: float = 0.05,
) -> tuple[bool, list[float]]:
    """Verify that a linkage satisfies function generation requirements.

    Simulates the linkage at each input angle and checks if the
    output angle matches the specification.

    Args:
        linkage: The linkage to verify.
        angle_pairs: Expected (input_angle, output_angle) pairs.
        tolerance: Maximum acceptable angle error in radians.

    Returns:
        Tuple of (all_satisfied, list of actual errors for each pair).
    """
    from ..geometry.secants import circle_intersect
    from ..joints import Crank, Revolute, Static

    errors: list[float] = []

    # Find the joints by type
    # Expected structure: [Static A, Static D, Crank B, Revolute C]
    joint_A: Static | None = None
    joint_D: Static | None = None
    joint_B: Crank | None = None
    joint_C: Revolute | None = None

    for joint in linkage.joints:
        if isinstance(joint, Crank):
            joint_B = joint
        elif isinstance(joint, Revolute):
            joint_C = joint
        elif isinstance(joint, Static):
            # Determine which static is A (connected to crank) vs D
            if joint_A is None:
                joint_A = joint
            else:
                joint_D = joint

    # Validate structure
    if joint_A is None or joint_D is None or joint_B is None or joint_C is None:
        return False, [float("inf")] * len(angle_pairs)

    # Ensure A is the one connected to the crank
    if joint_B.joint0 is joint_D:
        joint_A, joint_D = joint_D, joint_A

    # Get fixed parameters
    assert joint_A.x is not None and joint_A.y is not None
    assert joint_D.x is not None and joint_D.y is not None
    assert joint_B.r is not None
    assert joint_C.r0 is not None and joint_C.r1 is not None
    assert joint_C.x is not None and joint_C.y is not None

    A = (joint_A.x, joint_A.y)
    D = (joint_D.x, joint_D.y)
    crank_length = joint_B.r
    coupler_length = joint_C.r0
    rocker_length = joint_C.r1

    # Determine the configuration (which branch) from initial C position
    # This is needed to consistently choose the same branch during verification

    # Test each angle pair
    for theta2, expected_theta4 in angle_pairs:
        # Calculate crank position B at input angle theta2
        B = (
            A[0] + crank_length * math.cos(theta2),
            A[1] + crank_length * math.sin(theta2),
        )

        # Solve for coupler position C using circle-circle intersection
        # C is at distance coupler_length from B and rocker_length from D
        try:
            n_intersections, x1, y1, x2, y2 = circle_intersect(
                B[0], B[1], coupler_length, D[0], D[1], rocker_length
            )
        except Exception:
            errors.append(float("inf"))
            continue

        if n_intersections == 0:
            # Check if circles are nearly tangent (numerical precision issue)
            # Compute distance between circle centers
            dist_BD = math.sqrt((B[0] - D[0]) ** 2 + (B[1] - D[1]) ** 2)
            sum_radii = coupler_length + rocker_length
            diff_radii = abs(coupler_length - rocker_length)

            # Allow small tolerance for tangent circles
            tol = 1e-6 * max(dist_BD, sum_radii, 1.0)

            if abs(dist_BD - sum_radii) < tol or abs(dist_BD - diff_radii) < tol:
                # Circles are tangent - compute tangent point
                if abs(dist_BD - sum_radii) < tol:
                    # External tangent
                    t = coupler_length / (coupler_length + rocker_length)
                else:
                    # Internal tangent
                    t = coupler_length / abs(coupler_length - rocker_length)
                    if coupler_length > rocker_length:
                        t = 1 - t

                x1 = B[0] + t * (D[0] - B[0])
                y1 = B[1] + t * (D[1] - B[1])
                n_intersections = 1
            else:
                # Linkage truly cannot be assembled at this angle
                errors.append(float("inf"))
                continue

        # Choose the solution that gives θ4 closest to the expected value
        # This handles branch selection correctly for synthesis verification
        theta4_1 = math.atan2(y1 - D[1], x1 - D[0])
        if n_intersections == 2:
            theta4_2 = math.atan2(y2 - D[1], x2 - D[0])

            # Compute angle errors for both solutions (with wraparound)
            err1 = theta4_1 - expected_theta4
            while err1 > math.pi:
                err1 -= 2 * math.pi
            while err1 < -math.pi:
                err1 += 2 * math.pi

            err2 = theta4_2 - expected_theta4
            while err2 > math.pi:
                err2 -= 2 * math.pi
            while err2 < -math.pi:
                err2 += 2 * math.pi

            if abs(err2) < abs(err1):
                theta4_1 = theta4_2

        # Calculate actual rocker angle theta4
        actual_theta4 = theta4_1

        # Compute angle error (handle wraparound)
        error = actual_theta4 - expected_theta4
        # Normalize to [-π, π]
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi

        errors.append(abs(error))

    # Check if all errors are within tolerance
    all_satisfied = all(e <= tolerance for e in errors)

    return all_satisfied, errors
