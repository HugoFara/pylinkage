"""
Crank joint definition.
"""
from math import atan2

from . import joint as pl_joint
from .. import geometry as pl_geom
from .. import exceptions as pl_exceptions


class Crank(pl_joint.Joint):
    """Define a crank joint."""

    __slots__ = "r", "angle"

    def __init__(
        self,
        x=None,
        y=None,
        joint0=None,
        distance=None,
        angle=None,
        name=None
    ):
        """
        Define a crank (circular motor).

        :param x: Initial horizontal position, won't be used thereafter.
            The default is None.
        :type x: float | None
        :param y: Initial vertical position. The default is None.
        :type y: float | None
        :param joint0: First reference joint. The default is None.
        :type joint0: pylinkage.Joint | tuple[float, float] | None
        :param distance: Distance to keep between joint0 and self. The default is None.
        :type distance: float | None
        :param angle: It is the angle (horizontal axis, joint0, self).
            Should be in radian and in trigonometric order.
            The default is None.
        :type angle: float | None
        :param str | None name: Human-readable name. The default is None.
        """
        super().__init__(x, y, joint0, name=name)
        self.r, self.angle = distance, angle

    def reload(self, dt=1):
        """Make a step of crank.

        :param dt: Fraction of steps to take (Default value = 1)
        :type dt: float
        """
        if self.joint0 is None:
            return
        if None in self.joint0.coord():
            raise pl_exceptions.HypostaticError(
                f'{self.joint0} has None coordinates. '
                f'{self} cannot be calculated'
            )
        # Rotation angle of local space relative to global
        rot = atan2(self.y - self.joint0.y, self.x - self.joint0.x)
        self.x, self.y = pl_geom.cyl_to_cart(
            self.r, rot + self.angle * dt,
            self.joint0.coord()
        )

    def get_constraints(self):
        """Return the distance to the center of rotation."""
        return (self.r,)

    def set_constraints(self, distance=None, *args):
        """Set geometric constraints, only self.r is affected.

        :param distance: Distance from the reference point.
            (Default value = None)
        :type distance: float
        :param args: Unused, but preserves the object structure.

        """
        self.r = distance or self.r

    def set_anchor0(self, joint, distance=None):
        """First joint anchor and fixed distance.

        :param joint:
        :param distance:  (Default value = None)

        """
        self.joint0 = joint
        self.set_constraints(distance=distance)
