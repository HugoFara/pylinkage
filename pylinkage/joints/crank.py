"""
Crank joint definition.
"""
from math import atan2

from ..interface import joint as pl_joint
from .. import geometry as pl_geom
from ..interface import exceptions as pl_exceptions


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

        Parameters
        ----------
        x : float, optional
            initial horizontal position, won't be used thereafter.
            The default is None.
        y : float, optional
            initial vertical position. The default is None.
        joint0 : Union[Joint, tuple[float]], optional
            first reference joint. The default is None.
        distance : float, optional
            distance to keep between joint0 and self. The default is None.
        angle : float, optional
            It is the angle (horizontal axis, joint0, self).
            Should be in radian and in trigonometric order.
            The default is None.
        name : str, optional
            user-friendly name. The default is None.

        Returns
        -------
        None.

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
