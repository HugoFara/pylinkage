"""
Definition of a revolute joint.

It is also called pin joint or hinge joint, and used to be
called a pivot joint in this project.
"""
import warnings
from math import atan2

from .. import geometry as pl_geom
from ..interface import exceptions as pl_exceptions
from . import joint as pl_joint


class Revolute(pl_joint.Joint):
    """Center of a revolute joint."""

    __slots__ = "r0", "r1"

    def __init__(
            self,
            x=0,
            y=0,
            joint0=None,
            joint1=None,
            distance0=None,
            distance1=None,
            name=None
    ):
        """
        Set point position, parents, and if it is fixed for this turn.

        :param x: Position on the horizontal axis.
            (Default value = 0).
        :type x: float
        :param y: Position on vertical axis.
            (Default value = 0).
        :type y: float
        :param joint0: Linked revolute joint 0 (geometric constraints).
            (Default value = None).
        :type joint0: Joint | tuple
        :param joint1: Linked revolute joint 1 (geometric constraints).
            (Default value = None).
        :type joint1: Joint | tuple
        :param distance0: Distance from joint0 to the current Joint.
            (Default value = None).
        :type distance0: float
        :param distance1: Distance from joint1 to the current Joint.
            (Default value = None).
        :type distance1: float
        :param name: Friendly name for human readability.
            (Default value = None).
        :type name: str
        """
        super().__init__(x, y, joint0, joint1, name)
        self.r0, self.r1 = distance0, distance1

    def __get_joint_as_circle__(self, index):
        """
        A circle representing the reference anchor as (ref.x, ref.y, ref.distance).

        :param index: Get circle from the first of the second anchor.
        :type index: int

        :return: Constraint as a circle
        :rtype: tuple[float, float, float]
        """
        if index == 0:
            return self.joint0.x, self.joint0.y, self.r0
        if index == 1:
            return self.joint1.x, self.joint1.y, self.r1
        raise ValueError(f'index should be 0 or 1, not {index}')

    def circle(self, joint):
        """
        Return the first link between self and parent as a circle.

        :param joint: Parent joint you want to use
        :type joint: Joint
        :returns: Circle is a tuple (abscissa, ordinate, radius).
        :rtype: tuple[float, float, float]

        """
        if self.joint0 is joint:
            return joint.x, joint.y, self.r0
        if self.joint1 is joint:
            return joint.x, joint.y, self.r1
        raise ValueError(f'{joint} is not in joints of {self}')

    def reload(self):
        """Compute the position of revolute joint, use the two linked joints."""
        if self.joint0 is None:
            return
        # Fixed joint as reference. In links, we only keep fixed objects
        ref = tuple(x for x in self.__get_joints__() if None not in x.coord())
        if len(ref) == 0:
            # Don't change coordinates (irrelevant)
            return
        if len(ref) == 1:
            warnings.warn(
                "Unable to set coordinates of revolute joint {}:"
                "Only one constraint is set."
                "Coordinates unchanged".format(self.name)
            )
        elif len(ref) == 2:
            # Most common case, optimized here
            intersections = pl_geom.circle_intersect(
                self.__get_joint_as_circle__(0),
                self.__get_joint_as_circle__(1)
            )
            if intersections[0] == 0:
                raise pl_exceptions.UnbuildableError(self)
            if intersections[0] == 1:
                self.x, self.y = intersections[1]
            elif intersections[0] == 2:
                self.x, self.y = pl_geom.get_nearest_point(self.coord(), intersections[1], intersections[2])
            elif intersections[0] == 3:
                warnings.warn(
                    f"Joint {self.name} has an infinite number of"
                    "solutions, position will be arbitrary"
                )
                # We project position on circle of possible positions
                angle = atan2(self.y - self.joint0.y, self.x - self.joint0.x)
                self.x, self.y = pl_geom.cyl_to_cart(self.r0, angle, self.joint0.coord())

    def get_constraints(self):
        """Return the two constraining distances of this joint."""
        return self.r0, self.r1

    def set_constraints(self, distance0=None, distance1=None):
        """Set geometric constraints.

        :param distance0: Distance to the first reference (Default value = None)
        :type distance0: float
        :param distance1: Distance to the second reference (Default value = None)
        :type distance1: float

        """
        self.r0, self.r1 = distance0 or self.r0, distance1 or self.r1

    def set_anchor0(self, joint, distance=None):
        """Set the first anchor for this Joint.

        :param joint: The joint to use as anchor.
        :type joint: Joint | tuple[float]
        :param distance: Distance to keep constant from the anchor. The default is None.
        :type distance: float


        """
        self.joint0 = joint
        self.set_constraints(distance0=distance)

    def set_anchor1(self, joint, distance=None):
        """Set the second anchor for this Joint.

        :param joint: The joint to use as anchor.
        :type joint: Joint | tuple[float]
        :param distance: Distance to keep constant from the anchor. The default is None.
        :type distance: float
        """
        self.joint1 = joint
        self.set_constraints(distance1=distance)


class Pivot(Revolute):
    """
    Revolute Joint definition.

    .. deprecated :: 0.6.0
        This class has been de@recated in favor of `Revolute` which has a standard name.
        It will be removed in PyLinkage 0.7.0.
    """

    def __init__(
        self,
        x=0,
        y=0,
        joint0=None,
        joint1=None,
        distance0=None,
        distance1=None,
        name=None
    ):
        """
        Set point position, parents, and if it is fixed for this turn.

        :param x: Position on the horizontal axis.
            (Default value = 0).
        :type x: float
        :param y: Position on vertical axis.
            (Default value = 0).
        :type y: float
        :param joint0: Linked pivot joint 0 (geometric constraints).
            (Default value = None).
        :type joint0: Joint
        :param joint1: Linked pivot joint 1 (geometric constraints).
            (Default value = None).
        :type joint1: Joint
        :param distance0: Distance from joint0 to the current Joint.
            (Default value = None).
        :type distance0: float
        :param distance1: Distance from joint1 to the current Joint.
            (Default value = None).
        :type distance1: float
        :param name: Friendly name for human readability.
            (Default value = None).
        :type name: str
        """
        warnings.warn("The Pivot class is deprecated in favor of the Revolute class.")
        super().__init__(
            x=x,
            y=y,
            joint0=joint0,
            joint1=joint1,
            distance0=distance0,
            distance1=distance1,
            name=name
        )
