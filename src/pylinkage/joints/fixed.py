"""
Fixed joint (deterministic constraint).

A fixed joint has a position fully determined by its two parent joints
with no ambiguity. It uses polar coordinates relative to the line
between the two parents.

This is a thin wrapper around the solver's solve_fixed function.
"""

from .. import exceptions as pl_exceptions
from .._types import Coord
from ..solver.joints import solve_fixed
from . import joint as pl_joint


class Fixed(pl_joint.Joint):
    """Fixed joint - deterministic polar constraint.

    The position is fully determined by:
    - joint0: Origin point
    - joint1: Reference point for angle measurement
    - r: Distance from joint0
    - angle: Angle offset from the joint0→joint1 direction

    Unlike Revolute (RRR dyad), this has a unique solution.
    """

    __slots__ = "r", "angle"

    r: float | None
    angle: float | None

    def __init__(
        self,
        x: float | None = None,
        y: float | None = None,
        joint0: pl_joint.Joint | Coord | None = None,
        joint1: pl_joint.Joint | Coord | None = None,
        distance: float | None = None,
        angle: float | None = None,
        name: str | None = None,
    ) -> None:
        """
        Create a point, of position fully defined by its two references.

        :param x: Position on horizontal axis. The default is 0.
        :param y: Position on vertical axis. The default is O.
        :param name: Friendly name for human readability. The default is None.
        :param joint0: Linked revolute joint 1 (geometric constraints). The default is None.
        :param joint1: Other revolute joint linked. The default is None.
        :param distance: Distance to keep constant between joint0 and self. The default is
            None.
        :param angle: Angle (joint1, joint0, self). Should be in radian and in trigonometric
            order. The default is None.
        """
        super().__init__(x, y, joint0, joint1, name)
        self.angle = angle
        self.r = distance

    def reload(self, dt: float = 1) -> None:
        """Compute position using solver (deterministic constraint).

        :param dt: Unused, but preserves the object structure.
        """
        if self.joint0 is None or self.joint1 is None:
            raise pl_exceptions.UnderconstrainedError(f'Not enough constraints for {self}')
        if self.joint0.x is None or self.joint0.y is None:
            raise pl_exceptions.UnderconstrainedError(
                f'{self.joint0} has None coordinates.'
            )
        if self.joint1.x is None or self.joint1.y is None:
            raise pl_exceptions.UnderconstrainedError(
                f'{self.joint1} has None coordinates.'
            )
        if self.r is None or self.angle is None:
            raise pl_exceptions.UnderconstrainedError(
                f'{self} has None constraints (r={self.r}, angle={self.angle}).'
            )

        # Delegate to solver function (single source of truth)
        self.x, self.y = solve_fixed(
            self.joint0.x, self.joint0.y,
            self.joint1.x, self.joint1.y,
            self.r, self.angle
        )

    def get_constraints(self) -> tuple[float | None, float | None]:
        """Return the constraining distance and angle parameters."""
        return self.r, self.angle

    def set_constraints(
        self,
        distance: float | None = None,
        angle: float | None = None,
        *args: float | None,
    ) -> None:
        """Set geometric constraints.

        :param distance: Distance from joint0. (Default value = None).
        :param angle: Angle in radians. (Default value = None).
        :param args: Unused, but preserves the object structure.
        """
        self.r, self.angle = distance or self.r, angle or self.angle

    def set_anchor0(
        self,
        joint: pl_joint.Joint | Coord,
        distance: float | None = None,
        angle: float | None = None,
    ) -> None:
        """First joint anchor and characteristics.

        :param joint: Joint to set as anchor.
        :param distance: Distance from the joint. (Default value = None).
        :param angle: Angle in radians. (Default value = None).
        """
        self.joint0 = pl_joint.joint_syntax_parser(joint)
        self.set_constraints(distance, angle)

    def set_anchor1(self, joint: pl_joint.Joint | Coord) -> None:
        """Second joint anchor.

        :param joint: Joint to set as anchor.
        """
        self.joint1 = pl_joint.joint_syntax_parser(joint)
