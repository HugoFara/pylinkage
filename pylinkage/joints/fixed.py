"""
Fixed joint.
"""
import math

from .. import geometry as pl_geom
from ..interface import exceptions as pl_exceptions
from . import joint as pl_joint


class Fixed(pl_joint.Joint):
    """Define a joint using parents locations only, with no ambiguity."""

    __slots__ = "r", "angle"

    def __init__(self, x=None, y=None, joint0=None, joint1=None,
                 distance=None, angle=None, name=None):
        """
        Create a point, of position fully defined by its two references.

        Arguments
        ---------
        x : float, optional
            Position on horizontal axis. The default is 0.
        y : float, optional
            Position on vertical axis. The default is O.
        name : str, optional
            Friendly name for human readability. The default is None.
        joint0 : Union[Joint, tuple[float]], optional
            Linked revolute joint 1 (geometric constraints). The default is None.
        joint1 : Union[Joint, tuple[float]], optional
            Other revolute joint linked. The default is None.
        distance : float, optional
            Distance to keep constant between joint0 and self. The default is
            None.
        angle : float, optional
         Angle (joint1, joint0, self). Should be in radian and in trigonometric
         order. The default is None.
        """
        super().__init__(x, y, joint0, joint1, name)
        self.angle = angle
        self.r = distance

    def reload(self):
        """Compute point coordinates.

        We know point position relative to its two parents, which gives a local
        space.
        We know the orientation of local space, so we can solve the
        whole. Local space is defined by link[0] as the origin and
        (link[0], link[1]) as abscissas axis.
        """
        if self.joint0 is None:
            return
        if self.joint0 is None or self.joint1 is None:
            raise pl_exceptions.HypostaticError(f'Not enough constraints for {self}')
        # Rotation angle of local space relative to global
        rot = math.atan2(self.joint1.y - self.joint0.y,
                    self.joint1.x - self.joint0.x)
        # Position in global space
        self.x, self.y = pl_geom.cyl_to_cart(
            self.r, self.angle + rot, self.joint0.coord()
        )

    def get_constraints(self):
        """Return the constraining distance and angle parameters."""
        return self.r, self.angle

    def set_constraints(self, distance=None, angle=None):
        """Set geometric constraints.

        :param distance:  (Default value = None)
        :param angle:  (Default value = None)

        """
        self.r, self.angle = distance or self.r, angle or self.angle

    def set_anchor0(self, joint, distance=None, angle=None):
        """First joint anchor and characteristics.

        :param joint:
        :param distance:  (Default value = None)
        :param angle:  (Default value = None)

        """
        self.joint0 = joint
        self.set_constraints(distance, angle)

    def set_anchor1(self, joint):
        """Second joint anchor.

        :param joint: Joint to set as anchor

        """
        self.joint1 = joint
