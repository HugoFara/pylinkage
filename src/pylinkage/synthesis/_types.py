"""Type definitions for the synthesis module.

This module defines type aliases used throughout the synthesis module
for mechanism synthesis operations.

Type Categories:
    Points: Point2D, ComplexPoint - 2D point representations
    Precision Specs: AnglePair, PrecisionPoint - synthesis inputs
    Solutions: FourBarSolution, DyadSolution - synthesis outputs
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple, TypeAlias

from .._types import Coord

# Re-export Coord as Point2D for clarity in synthesis context
Point2D: TypeAlias = Coord
"""A 2D point as (x, y)."""

# Complex number representation of 2D points
ComplexPoint: TypeAlias = complex
"""A 2D point represented as complex number x + iy."""

# Precision specifications for synthesis
PrecisionPoint: TypeAlias = Point2D
"""A target point for path generation."""

AnglePair: TypeAlias = tuple[float, float]
"""Input/output angle pair (theta_in, theta_out) in radians."""


class SynthesisType(Enum):
    """Type of synthesis problem."""

    FUNCTION = auto()
    """Function generation: match input/output angle relationships."""
    PATH = auto()
    """Path generation: coupler curve through precision points."""
    MOTION = auto()
    """Motion generation: rigid body guidance through poses."""


@dataclass(frozen=True, slots=True)
class Pose:
    """A rigid body pose: position + orientation.

    Attributes:
        x: X-coordinate of the pose origin.
        y: Y-coordinate of the pose origin.
        angle: Orientation angle in radians.
    """

    x: float
    y: float
    angle: float

    def to_complex(self) -> ComplexPoint:
        """Convert position to complex representation."""
        return complex(self.x, self.y)

    @classmethod
    def from_point_angle(cls, point: Point2D, angle: float) -> "Pose":
        """Create a Pose from a point and angle."""
        return cls(x=point[0], y=point[1], angle=angle)


class FourBarSolution(NamedTuple):
    """A four-bar linkage solution from synthesis.

    The four-bar consists of:
    - Ground link: from A to D (fixed frame)
    - Crank (input link): from A to B
    - Coupler: from B to C
    - Rocker (output link): from D to C

    Attributes:
        ground_pivot_a: Position of fixed pivot A (crank base).
        ground_pivot_d: Position of fixed pivot D (rocker base).
        crank_pivot_b: Initial position of crank pin B.
        coupler_pivot_c: Initial position of coupler-rocker pin C.
        crank_length: Length of crank link (A to B).
        coupler_length: Length of coupler link (B to C).
        rocker_length: Length of rocker link (D to C).
        ground_length: Length of ground link (A to D).
    """

    ground_pivot_a: Point2D
    ground_pivot_d: Point2D
    crank_pivot_b: Point2D
    coupler_pivot_c: Point2D
    crank_length: float
    coupler_length: float
    rocker_length: float
    ground_length: float


class DyadSolution(NamedTuple):
    """A dyad (two-link chain) from Burmester synthesis.

    A dyad connects a moving point (circle point on coupler)
    to a fixed pivot (center point on frame).

    Attributes:
        circle_point: Point on the moving body (complex representation).
        center_point: Fixed pivot point (complex representation).
    """

    circle_point: ComplexPoint
    center_point: ComplexPoint

    @property
    def link_length(self) -> float:
        """Length of the dyad link."""
        return abs(self.circle_point - self.center_point)

    def to_cartesian(self) -> tuple[Point2D, Point2D]:
        """Convert to Cartesian coordinate tuples."""
        return (
            (self.circle_point.real, self.circle_point.imag),
            (self.center_point.real, self.center_point.imag),
        )
