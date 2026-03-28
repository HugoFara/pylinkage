"""Conversion between synthesis results and Linkage objects.

This module provides functions to convert raw mathematical solutions
from synthesis algorithms into pylinkage Linkage objects that can
be simulated and visualized.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ._types import FourBarSolution, Point2D, SynthesisType

if TYPE_CHECKING:
    from ..linkage import Linkage


def _compute_coupler_point_params(
    B: Point2D,
    C: Point2D,
    P: Point2D,
) -> tuple[float, float]:
    """Compute Fixed joint parameters (distance, angle) for a coupler point.

    The coupler point P is on the rigid coupler link B-C. We express
    P in polar coordinates relative to B, with the angle measured from
    the B→C direction.

    Args:
        B: Crank-coupler joint position.
        C: Coupler-rocker joint position.
        P: Coupler point (traced point) position.

    Returns:
        Tuple of (distance_from_B, angle_from_BC_direction).
    """
    # Vector B→P
    bp_x = P[0] - B[0]
    bp_y = P[1] - B[1]
    distance = math.sqrt(bp_x * bp_x + bp_y * bp_y)

    # Angle of B→P
    bp_angle = math.atan2(bp_y, bp_x)

    # Angle of B→C (reference direction for the Fixed joint)
    bc_x = C[0] - B[0]
    bc_y = C[1] - B[1]
    bc_angle = math.atan2(bc_y, bc_x)

    # Relative angle
    angle = bp_angle - bc_angle

    return distance, angle


def solution_to_linkage(
    solution: FourBarSolution,
    name: str = "synthesized",
    iterations: int = 100,
) -> Linkage:
    """Convert a single FourBarSolution to a Linkage object.

    Creates a four-bar linkage with:
    - Two Static joints (A, D) as ground pivots
    - One Crank joint (B) connected to A
    - One Revolute joint (C) connecting B to D
    - Optionally, a Fixed joint (P) for the coupler point that traces
      the target path (only if solution.coupler_point is set)

    Args:
        solution: FourBarSolution containing geometry.
        name: Name for the linkage.
        iterations: Number of simulation steps per rotation.

    Returns:
        Linkage object ready for simulation.
    """
    from ..joints import Crank, Fixed, Revolute, Static
    from ..linkage import Linkage

    # Create ground pivots
    A = solution.ground_pivot_a
    D = solution.ground_pivot_d

    joint_A = Static(x=A[0], y=A[1], name="A")
    joint_D = Static(x=D[0], y=D[1], name="D")

    # Create crank joint
    B = solution.crank_pivot_b

    # Angle step per iteration (full rotation in `iterations` steps)
    angle_step = 2 * math.pi / iterations

    joint_B = Crank(
        x=B[0],
        y=B[1],
        joint0=joint_A,
        distance=solution.crank_length,
        angle=angle_step,
        name="B",
    )

    # Create revolute joint connecting crank to rocker
    C = solution.coupler_pivot_c

    joint_C = Revolute(
        x=C[0],
        y=C[1],
        joint0=joint_B,
        joint1=joint_D,
        distance0=solution.coupler_length,
        distance1=solution.rocker_length,
        name="C",
    )

    joints = [joint_A, joint_D, joint_B, joint_C]
    order = [joint_B, joint_C]

    # Add coupler point tracker if present
    if solution.coupler_point is not None:
        P = solution.coupler_point
        dist_bp, angle_bp = _compute_coupler_point_params(B, C, P)

        joint_P = Fixed(
            x=P[0],
            y=P[1],
            joint0=joint_B,
            joint1=joint_C,
            distance=dist_bp,
            angle=angle_bp,
            name="P",
        )
        joints.append(joint_P)
        order.append(joint_P)

    # Create linkage with solve order
    linkage = Linkage(
        joints=joints,
        order=order,
        name=name,
    )

    return linkage


def solutions_to_linkages(
    solutions: list[FourBarSolution],
    synthesis_type: SynthesisType,
    iterations: int = 100,
) -> list[Linkage]:
    """Convert multiple solutions to Linkage objects.

    Args:
        solutions: List of FourBarSolution objects.
        synthesis_type: Type of synthesis (for naming).
        iterations: Number of simulation steps per rotation.

    Returns:
        List of Linkage objects.
    """
    linkages: list[Linkage] = []

    for i, sol in enumerate(solutions):
        name = f"{synthesis_type.name.lower()}_{i}"
        try:
            linkage = solution_to_linkage(sol, name=name, iterations=iterations)
            linkages.append(linkage)
        except Exception:
            # Skip invalid solutions
            continue

    return linkages


def linkage_to_synthesis_params(
    linkage: Linkage,
) -> FourBarSolution:
    """Extract synthesis parameters from an existing four-bar linkage.

    Analyzes a Linkage object to extract the four-bar geometry
    parameters as a FourBarSolution.

    Args:
        linkage: A four-bar Linkage object.

    Returns:
        FourBarSolution tuple with geometry parameters.

    Raises:
        ValueError: If linkage is not a valid four-bar.
    """
    from ..joints import Crank, Fixed, Revolute, Static

    # Find joints by type
    statics: list[Static] = []
    cranks: list[Crank] = []
    revolutes: list[Revolute] = []
    fixed_joints: list[Fixed] = []

    for joint in linkage.joints:
        if isinstance(joint, Crank):
            cranks.append(joint)
        elif isinstance(joint, Fixed):
            fixed_joints.append(joint)
        elif isinstance(joint, Revolute):
            revolutes.append(joint)
        elif isinstance(joint, Static):
            statics.append(joint)

    # Validate structure
    if len(statics) < 2:
        raise ValueError(
            f"Four-bar needs 2 static joints, found {len(statics)}"
        )
    if len(cranks) < 1:
        raise ValueError(f"Four-bar needs 1 crank joint, found {len(cranks)}")
    if len(revolutes) < 1:
        raise ValueError(
            f"Four-bar needs 1 revolute joint, found {len(revolutes)}"
        )

    # Identify joints
    crank = cranks[0]
    revolute = revolutes[0]

    # Find which static is A (connected to crank)
    joint_A = None
    joint_D = None

    for static in statics:
        if crank.joint0 is static:
            joint_A = static
        else:
            joint_D = static

    if joint_A is None:
        joint_A = statics[0]
    if joint_D is None:
        joint_D = statics[1] if len(statics) > 1 else statics[0]

    # Extract positions (assert not None for type checker)
    assert joint_A.x is not None and joint_A.y is not None
    assert joint_D.x is not None and joint_D.y is not None
    assert crank.x is not None and crank.y is not None
    assert revolute.x is not None and revolute.y is not None

    A: Point2D = (joint_A.x, joint_A.y)
    D: Point2D = (joint_D.x, joint_D.y)
    B: Point2D = (crank.x, crank.y)
    C: Point2D = (revolute.x, revolute.y)

    # Extract lengths
    crank_length_val: float = crank.r if hasattr(crank, "r") and crank.r is not None else math.sqrt(
        (B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2
    )

    coupler_length_val: float = revolute.r0 if hasattr(revolute, "r0") and revolute.r0 is not None else math.sqrt(
        (C[0] - B[0]) ** 2 + (C[1] - B[1]) ** 2
    )

    rocker_length_val: float = revolute.r1 if hasattr(revolute, "r1") and revolute.r1 is not None else math.sqrt(
        (C[0] - D[0]) ** 2 + (C[1] - D[1]) ** 2
    )

    ground_length = math.sqrt((D[0] - A[0]) ** 2 + (D[1] - A[1]) ** 2)

    # Extract coupler point if present
    coupler_point: Point2D | None = None
    for fj in fixed_joints:
        if fj.name == "P" and fj.x is not None and fj.y is not None:
            coupler_point = (fj.x, fj.y)
            break

    return FourBarSolution(
        ground_pivot_a=A,
        ground_pivot_d=D,
        crank_pivot_b=B,
        coupler_pivot_c=C,
        crank_length=crank_length_val,
        coupler_length=coupler_length_val,
        rocker_length=rocker_length_val,
        ground_length=ground_length,
        coupler_point=coupler_point,
    )


def fourbar_from_lengths(
    crank_length: float,
    coupler_length: float,
    rocker_length: float,
    ground_length: float,
    ground_pivot_a: Point2D = (0.0, 0.0),
    initial_crank_angle: float = 0.0,
    iterations: int = 100,
    name: str = "fourbar",
) -> Linkage:
    """Create a four-bar linkage from link lengths.

    Convenience function to create a four-bar linkage given only
    the link lengths. The linkage is placed with ground pivot A
    at the specified position.

    Args:
        crank_length: Length of crank (a).
        coupler_length: Length of coupler (b).
        rocker_length: Length of rocker (c).
        ground_length: Length of ground link (d).
        ground_pivot_a: Position of first ground pivot.
        initial_crank_angle: Initial crank angle in radians.
        iterations: Number of simulation steps per rotation.
        name: Name for the linkage.

    Returns:
        Linkage object.

    Example:
        >>> linkage = fourbar_from_lengths(1.0, 3.0, 3.0, 4.0)
        >>> linkage.show()
    """
    from ..joints import Crank, Revolute, Static
    from ..linkage import Linkage

    A = ground_pivot_a
    D = (A[0] + ground_length, A[1])

    # Initial crank position
    B = (
        A[0] + crank_length * math.cos(initial_crank_angle),
        A[1] + crank_length * math.sin(initial_crank_angle),
    )

    # Initial coupler/rocker position (solve for C)
    # C is at distance coupler_length from B and rocker_length from D
    # Use circle-circle intersection
    from ..geometry.secants import circle_intersect

    n_intersections, x1, y1, x2, y2 = circle_intersect(
        B[0], B[1], coupler_length, D[0], D[1], rocker_length
    )

    if n_intersections == 0:
        raise ValueError(
            f"Cannot assemble four-bar with given lengths at angle {initial_crank_angle}"
        )

    # Choose the upper solution (positive y relative to BD line)
    C = (x1, y1)
    if n_intersections == 2 and y2 > y1:
        # Pick the one with larger y (upper configuration)
        C = (x2, y2)

    # Create joints
    joint_A = Static(x=A[0], y=A[1], name="A")
    joint_D = Static(x=D[0], y=D[1], name="D")

    angle_step = 2 * math.pi / iterations
    joint_B = Crank(
        x=B[0],
        y=B[1],
        joint0=joint_A,
        distance=crank_length,
        angle=angle_step,
        name="B",
    )

    joint_C = Revolute(
        x=C[0],
        y=C[1],
        joint0=joint_B,
        joint1=joint_D,
        distance0=coupler_length,
        distance1=rocker_length,
        name="C",
    )

    return Linkage(
        joints=[joint_A, joint_D, joint_B, joint_C],
        order=[joint_B, joint_C],
        name=name,
    )
