"""
Kinematic analysis for linkage mechanisms.

This module provides analysis tools for:

1. Transmission angle analysis (for Revolute/RRR joints):
   - The angle between coupler and output links at their connecting joint
   - Ideal range: 40° to 140°, optimal at 90°

2. Stroke analysis (for Prismatic/RRP joints):
   - The slide position along the prismatic axis
   - Tracks min/max/range of travel over a motion cycle
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .._types import Coord

if TYPE_CHECKING:
    from .linkage import Linkage


@dataclass(frozen=True)
class TransmissionAngleAnalysis:
    """Results of transmission angle analysis over a motion cycle.

    Attributes:
        min_angle: Minimum transmission angle in degrees.
        max_angle: Maximum transmission angle in degrees.
        mean_angle: Mean transmission angle in degrees.
        angles: Array of transmission angles at each step (degrees).
        is_acceptable: True if angle always in acceptable range.
        min_deviation: Minimum deviation from 90 degrees.
        max_deviation: Maximum deviation from 90 degrees.
        min_angle_step: Step index where minimum angle occurs.
        max_angle_step: Step index where maximum angle occurs.
    """

    min_angle: float
    max_angle: float
    mean_angle: float
    angles: NDArray[np.float64]
    is_acceptable: bool
    min_deviation: float
    max_deviation: float
    min_angle_step: int
    max_angle_step: int

    @property
    def acceptable_range(self) -> tuple[float, float]:
        """Return the standard acceptable range [40, 140] degrees."""
        return (40.0, 140.0)

    def worst_angle(self) -> float:
        """Return the angle with maximum deviation from 90 degrees."""
        if abs(self.max_angle - 90.0) >= abs(self.min_angle - 90.0):
            return self.max_angle
        return self.min_angle


def compute_transmission_angle(
    coupler_joint: Coord,
    output_joint: Coord,
    rocker_pivot: Coord,
) -> float:
    """Compute transmission angle in degrees.

    The transmission angle is the angle between:
    - Coupler link: from coupler_joint (B) to output_joint (C)
    - Rocker link: from rocker_pivot (D) to output_joint (C)

    Args:
        coupler_joint: Position of the coupler input joint (B).
        output_joint: Position where angle is measured (C).
        rocker_pivot: Position of the rocker ground pivot (D).

    Returns:
        Transmission angle in degrees (0-180).
    """
    # Vector from B to C (coupler direction at C)
    bc_x = output_joint[0] - coupler_joint[0]
    bc_y = output_joint[1] - coupler_joint[1]

    # Vector from D to C (rocker direction at C)
    dc_x = output_joint[0] - rocker_pivot[0]
    dc_y = output_joint[1] - rocker_pivot[1]

    # Magnitudes
    mag_bc = math.sqrt(bc_x * bc_x + bc_y * bc_y)
    mag_dc = math.sqrt(dc_x * dc_x + dc_y * dc_y)

    # Handle degenerate case (zero-length vector)
    if mag_bc < 1e-10 or mag_dc < 1e-10:
        return 90.0

    # Dot product
    dot = bc_x * dc_x + bc_y * dc_y

    # Cosine of angle (clamped for numerical stability)
    cos_angle = dot / (mag_bc * mag_dc)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def _get_joint_coord(joint: object) -> Coord | None:
    """Extract coordinates from a joint or tuple.

    Args:
        joint: A Joint object or a (x, y) tuple.

    Returns:
        Coordinates as (x, y) or None if not available.
    """
    if joint is None:
        return None
    if isinstance(joint, tuple):
        return joint
    # Joint object with x, y attributes
    x = getattr(joint, "x", None)
    y = getattr(joint, "y", None)
    if x is None or y is None:
        return None
    return (x, y)


def _auto_detect_fourbar_joints(
    linkage: Linkage,
) -> tuple[object, object, object]:
    """Auto-detect coupler, output, and rocker pivot joints for a four-bar.

    For a standard four-bar:
    - Crank (B): The driver joint rotating around ground
    - Revolute (C): The output joint connecting coupler to rocker
    - Rocker pivot (D): The ground joint where rocker connects

    Args:
        linkage: The linkage to analyze.

    Returns:
        Tuple of (crank, revolute, rocker_pivot) joints.

    Raises:
        ValueError: If joints cannot be auto-detected.
    """
    # Import here to avoid circular imports
    from ..joints import Crank, Revolute

    crank = None
    revolute = None

    for joint in linkage.joints:
        if isinstance(joint, Crank):
            crank = joint
        elif isinstance(joint, Revolute):
            revolute = joint

    if crank is None:
        raise ValueError(
            "Cannot auto-detect: no Crank joint found. "
            "Please specify joints explicitly."
        )
    if revolute is None:
        raise ValueError(
            "Cannot auto-detect: no Revolute joint found. "
            "Please specify joints explicitly."
        )

    # The rocker pivot is joint1 of the Revolute
    rocker_pivot = revolute.joint1
    if rocker_pivot is None:
        raise ValueError(
            "Cannot auto-detect: Revolute.joint1 (rocker pivot) is None."
        )

    return crank, revolute, rocker_pivot


def transmission_angle_at_position(
    linkage: Linkage,
    coupler_joint: object | None = None,
    output_joint: object | None = None,
    rocker_pivot: object | None = None,
) -> float:
    """Compute transmission angle at the current linkage position.

    Args:
        linkage: The linkage to analyze.
        coupler_joint: Joint at coupler input (B). Auto-detected if None.
        output_joint: Joint where angle is measured (C). Auto-detected if None.
        rocker_pivot: Ground pivot of rocker (D). Auto-detected if None.

    Returns:
        Transmission angle in degrees.

    Raises:
        ValueError: If joints cannot be determined or positions are invalid.
    """
    if coupler_joint is None or output_joint is None or rocker_pivot is None:
        coupler_joint, output_joint, rocker_pivot = _auto_detect_fourbar_joints(
            linkage
        )

    b_coord = _get_joint_coord(coupler_joint)
    c_coord = _get_joint_coord(output_joint)
    d_coord = _get_joint_coord(rocker_pivot)

    if b_coord is None or c_coord is None or d_coord is None:
        raise ValueError("Could not determine joint positions.")

    return compute_transmission_angle(b_coord, c_coord, d_coord)


def analyze_transmission(
    linkage: Linkage,
    iterations: int | None = None,
    acceptable_range: tuple[float, float] = (40.0, 140.0),
) -> TransmissionAngleAnalysis:
    """Analyze transmission angle over a full motion cycle.

    For a standard four-bar linkage, joints are auto-detected:
    - coupler_joint: The Crank joint (B)
    - output_joint: The Revolute joint (C)
    - rocker_pivot: The Static joint at Revolute.joint1 (D)

    Args:
        linkage: The Linkage to analyze.
        iterations: Number of simulation steps. Defaults to one full rotation.
        acceptable_range: (min, max) acceptable angles in degrees.

    Returns:
        TransmissionAngleAnalysis with statistics over the motion cycle.

    Raises:
        ValueError: If joints cannot be detected or no valid positions found.
    """
    coupler_joint, output_joint, rocker_pivot = _auto_detect_fourbar_joints(linkage)

    # Save initial state
    initial_coords = linkage.get_coords()

    if iterations is None:
        iterations = linkage.get_rotation_period()

    # Collect angles over the cycle
    angles: list[float] = []

    try:
        for _ in linkage.step(iterations=iterations):
            b_coord = _get_joint_coord(coupler_joint)
            c_coord = _get_joint_coord(output_joint)
            d_coord = _get_joint_coord(rocker_pivot)

            if b_coord is None or c_coord is None or d_coord is None:
                continue

            angle = compute_transmission_angle(b_coord, c_coord, d_coord)
            angles.append(angle)
    finally:
        # Restore initial state
        linkage.set_coords(initial_coords)

    if not angles:
        raise ValueError("No valid configurations found during simulation.")

    angles_array = np.array(angles, dtype=np.float64)
    min_angle = float(np.min(angles_array))
    max_angle = float(np.max(angles_array))
    mean_angle = float(np.mean(angles_array))

    min_acceptable, max_acceptable = acceptable_range
    is_acceptable = min_angle >= min_acceptable and max_angle <= max_acceptable

    # Deviation from optimal (90 degrees)
    deviations = np.abs(angles_array - 90.0)
    min_deviation = float(np.min(deviations))
    max_deviation = float(np.max(deviations))

    return TransmissionAngleAnalysis(
        min_angle=min_angle,
        max_angle=max_angle,
        mean_angle=mean_angle,
        angles=angles_array,
        is_acceptable=is_acceptable,
        min_deviation=min_deviation,
        max_deviation=max_deviation,
        min_angle_step=int(np.argmin(angles_array)),
        max_angle_step=int(np.argmax(angles_array)),
    )


# =============================================================================
# Stroke Analysis for Prismatic Joints
# =============================================================================


@dataclass(frozen=True)
class StrokeAnalysis:
    """Results of stroke analysis for a prismatic joint over a motion cycle.

    The stroke is the position of the slider along its axis, measured as
    the signed distance from the first line point (joint1) projected onto
    the slide axis.

    Attributes:
        min_position: Minimum slide position along the axis.
        max_position: Maximum slide position along the axis.
        mean_position: Mean slide position.
        stroke_range: Total travel distance (max - min).
        positions: Array of slide positions at each step.
        min_position_step: Step index where minimum position occurs.
        max_position_step: Step index where maximum position occurs.
    """

    min_position: float
    max_position: float
    mean_position: float
    stroke_range: float
    positions: NDArray[np.float64]
    min_position_step: int
    max_position_step: int

    @property
    def amplitude(self) -> float:
        """Return half the stroke range (useful for oscillating mechanisms)."""
        return self.stroke_range / 2.0

    @property
    def center_position(self) -> float:
        """Return the center of the stroke range."""
        return (self.min_position + self.max_position) / 2.0


def compute_slide_position(
    slider_pos: Coord,
    line_point1: Coord,
    line_point2: Coord,
) -> float:
    """Compute the slide position along a prismatic axis.

    The position is the signed distance from line_point1 to the projection
    of slider_pos onto the line, measured in the direction of line_point2.

    Args:
        slider_pos: Current position of the slider joint.
        line_point1: First point defining the slide axis (origin).
        line_point2: Second point defining the slide axis (direction).

    Returns:
        Signed distance along the axis from line_point1.
    """
    # Direction vector of the line
    dx = line_point2[0] - line_point1[0]
    dy = line_point2[1] - line_point1[1]

    # Length of line segment
    line_length = math.sqrt(dx * dx + dy * dy)

    if line_length < 1e-10:
        return 0.0

    # Unit direction vector
    ux = dx / line_length
    uy = dy / line_length

    # Vector from line_point1 to slider
    px = slider_pos[0] - line_point1[0]
    py = slider_pos[1] - line_point1[1]

    # Project onto line direction (dot product)
    slide_distance = px * ux + py * uy

    return slide_distance


def _auto_detect_prismatic_joint(linkage: Linkage) -> object:
    """Auto-detect a Prismatic joint in the linkage.

    Args:
        linkage: The linkage to analyze.

    Returns:
        The first Prismatic joint found.

    Raises:
        ValueError: If no Prismatic joint is found.
    """
    from ..joints import Prismatic

    for joint in linkage.joints:
        if isinstance(joint, Prismatic):
            return joint

    raise ValueError(
        "Cannot auto-detect: no Prismatic joint found. "
        "Please specify the joint explicitly."
    )


def stroke_at_position(
    linkage: Linkage,
    prismatic_joint: object | None = None,
) -> float:
    """Compute the slide position of a prismatic joint at current position.

    Args:
        linkage: The linkage to analyze.
        prismatic_joint: The Prismatic joint to analyze. Auto-detected if None.

    Returns:
        Slide position along the prismatic axis.

    Raises:
        ValueError: If joint cannot be determined or positions are invalid.
    """
    if prismatic_joint is None:
        prismatic_joint = _auto_detect_prismatic_joint(linkage)

    # Get the slider position
    slider_coord = _get_joint_coord(prismatic_joint)
    if slider_coord is None:
        raise ValueError("Could not determine slider position.")

    # Get the line points (joint1 and joint2 of Prismatic)
    joint1 = getattr(prismatic_joint, "joint1", None)
    joint2 = getattr(prismatic_joint, "joint2", None)

    if joint1 is None or joint2 is None:
        raise ValueError("Prismatic joint does not have joint1 and joint2 defined.")

    line_point1 = _get_joint_coord(joint1)
    line_point2 = _get_joint_coord(joint2)

    if line_point1 is None or line_point2 is None:
        raise ValueError("Could not determine line point positions.")

    return compute_slide_position(slider_coord, line_point1, line_point2)


def analyze_stroke(
    linkage: Linkage,
    prismatic_joint: object | None = None,
    iterations: int | None = None,
) -> StrokeAnalysis:
    """Analyze stroke/slide position over a full motion cycle.

    For a linkage with a Prismatic joint, this tracks the slide position
    along the prismatic axis throughout the motion cycle.

    Args:
        linkage: The Linkage to analyze.
        prismatic_joint: The Prismatic joint to analyze. Auto-detected if None.
        iterations: Number of simulation steps. Defaults to one full rotation.

    Returns:
        StrokeAnalysis with statistics over the motion cycle.

    Raises:
        ValueError: If joint cannot be detected or no valid positions found.
    """
    if prismatic_joint is None:
        prismatic_joint = _auto_detect_prismatic_joint(linkage)

    # Get the line points
    joint1 = getattr(prismatic_joint, "joint1", None)
    joint2 = getattr(prismatic_joint, "joint2", None)

    if joint1 is None or joint2 is None:
        raise ValueError("Prismatic joint does not have joint1 and joint2 defined.")

    # Save initial state
    initial_coords = linkage.get_coords()

    if iterations is None:
        iterations = linkage.get_rotation_period()

    # Collect positions over the cycle
    positions: list[float] = []

    try:
        for _ in linkage.step(iterations=iterations):
            slider_coord = _get_joint_coord(prismatic_joint)
            line_point1 = _get_joint_coord(joint1)
            line_point2 = _get_joint_coord(joint2)

            if slider_coord is None or line_point1 is None or line_point2 is None:
                continue

            pos = compute_slide_position(slider_coord, line_point1, line_point2)
            positions.append(pos)
    finally:
        # Restore initial state
        linkage.set_coords(initial_coords)

    if not positions:
        raise ValueError("No valid configurations found during simulation.")

    positions_array = np.array(positions, dtype=np.float64)
    min_pos = float(np.min(positions_array))
    max_pos = float(np.max(positions_array))
    mean_pos = float(np.mean(positions_array))
    stroke_range = max_pos - min_pos

    return StrokeAnalysis(
        min_position=min_pos,
        max_position=max_pos,
        mean_position=mean_pos,
        stroke_range=stroke_range,
        positions=positions_array,
        min_position_step=int(np.argmin(positions_array)),
        max_position_step=int(np.argmax(positions_array)),
    )
