"""
Fixed joint.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .. import exceptions as pl_exceptions
from .. import geometry as pl_geom
from . import joint as pl_joint

if TYPE_CHECKING:
    from .._types import Coord


class Fixed(pl_joint.Joint):
    """Define a joint using parents locations only, with no ambiguity."""

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
        """Compute point coordinates.

        We know point position relative to its two parents, which gives a local
        space.
        We know the orientation of local space, so we can solve the
        whole. Local space is defined by link[0] as the origin and
        (link[0], link[1]) as abscissas axis.

        :param dt: Unused, but preserves the object structure.
        """
        if self.joint0 is None or self.joint1 is None:
            raise pl_exceptions.HypostaticError(f'Not enough constraints for {self}')
        # Type assertions after validation
        assert self.joint0.x is not None and self.joint0.y is not None
        assert self.joint1.x is not None and self.joint1.y is not None
        assert self.r is not None and self.angle is not None
        # Rotation angle of local space relative to global
        rot = math.atan2(
            self.joint1.y - self.joint0.y,
            self.joint1.x - self.joint0.x,
        )
        # Position in global space
        self.x, self.y = pl_geom.cyl_to_cart(
            self.r, self.angle + rot, (self.joint0.x, self.joint0.y)
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
