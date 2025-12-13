"""
Crank joint definition.

A crank is a driver joint that rotates around a fixed anchor point.
This is a thin wrapper around the solver's solve_crank function.
"""

from .. import exceptions as pl_exceptions
from .._types import Coord
from ..solver.joints import solve_crank
from . import joint as pl_joint


class Crank(pl_joint.Joint):
    """Define a crank (rotary driver) joint.

    The crank rotates around its anchor point (joint0) at a constant
    angular velocity. It is the primary driver for linkage mechanisms.
    """

    __slots__ = "r", "angle"

    r: float | None
    angle: float | None

    def __init__(
        self,
        x: float | None = None,
        y: float | None = None,
        joint0: pl_joint.Joint | Coord | None = None,
        distance: float | None = None,
        angle: float | None = None,
        name: str | None = None,
    ) -> None:
        """
        Define a crank (circular motor).

        :param x: Initial horizontal position, won't be used thereafter.
            The default is None.
        :param y: Initial vertical position. The default is None.
        :param joint0: First reference joint. The default is None.
        :param distance: Distance to keep between joint0 and self. The default is None.
        :param angle: It is the angle (horizontal axis, joint0, self).
            Should be in radian and in trigonometric order.
            The default is None.
        :param name: Human-readable name. The default is None.
        """
        super().__init__(x, y, joint0, name=name)
        self.r, self.angle = distance, angle

    def reload(self, dt: float = 1) -> None:
        """Make a step of crank using solver function.

        :param dt: Fraction of steps to take (Default value = 1).
        """
        if self.joint0 is None:
            return
        if None in self.joint0.coord():
            raise pl_exceptions.UnderconstrainedError(
                f'{self.joint0} has None coordinates. '
                f'{self} cannot be calculated'
            )
        if self.x is None or self.y is None:
            raise pl_exceptions.UnderconstrainedError(
                f'{self} has None coordinates.'
            )
        if self.r is None or self.angle is None:
            raise pl_exceptions.UnderconstrainedError(
                f'{self} has None constraints (r={self.r}, angle={self.angle}).'
            )

        # Delegate to solver function (single source of truth)
        j0x = self.joint0.x if self.joint0 is not None else 0.0
        j0y = self.joint0.y if self.joint0 is not None else 0.0
        self.x, self.y = solve_crank(
            self.x, self.y,
            j0x or 0.0, j0y or 0.0,
            self.r, self.angle, dt
        )

    def get_constraints(self) -> tuple[float | None]:
        """Return the distance to the center of rotation."""
        return (self.r,)

    def set_constraints(self, distance: float | None = None, *args: float | None) -> None:
        """Set geometric constraints, only self.r is affected.

        :param distance: Distance from the reference point.
            (Default value = None).
        :param args: Unused, but preserves the object structure.
        """
        self.r = distance or self.r

    def set_anchor0(
        self,
        joint: pl_joint.Joint | Coord,
        distance: float | None = None,
    ) -> None:
        """First joint anchor and fixed distance.

        :param joint: Joint to set as anchor.
        :param distance: Distance from the joint. (Default value = None).
        """
        self.joint0 = pl_joint.joint_syntax_parser(joint)
        self.set_constraints(distance=distance)
