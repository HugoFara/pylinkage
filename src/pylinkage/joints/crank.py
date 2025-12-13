"""
Crank joint definition.
"""

from __future__ import annotations

from math import atan2
from typing import TYPE_CHECKING

from .. import exceptions as pl_exceptions
from .. import geometry as pl_geom
from . import joint as pl_joint

if TYPE_CHECKING:
    from .._types import Coord


class Crank(pl_joint.Joint):
    """Define a crank joint."""

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
        """Make a step of crank.

        :param dt: Fraction of steps to take (Default value = 1).
        """
        if self.joint0 is None:
            return
        if None in self.joint0.coord():
            raise pl_exceptions.UnderconstrainedError(
                f'{self.joint0} has None coordinates. '
                f'{self} cannot be calculated'
            )
        # Type assertions after validation
        assert self.joint0.x is not None and self.joint0.y is not None
        assert self.x is not None and self.y is not None
        assert self.r is not None and self.angle is not None
        # Rotation angle of local space relative to global
        rot = atan2(self.y - self.joint0.y, self.x - self.joint0.x)
        self.x, self.y = pl_geom.cyl_to_cart(
            self.r, rot + self.angle * dt,
            self.joint0.x, self.joint0.y
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
