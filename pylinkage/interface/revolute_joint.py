"""
Definition of a revolute joint.

It is also called pin joint or hinge joint, and used to be
called a pivot joint in this project.
"""
import warnings
from math import atan2

from ..geometry import sqr_dist, circle_intersect, cyl_to_cart
from .exceptions import UnbuildableError
from .joint import Joint


class Pivot(Joint):
    """Center of pivot joint."""

    __slots__ = "r0", "r1"

    def __init__(self, x=0, y=0, joint0=None, joint1=None, distance0=None,
                 distance1=None, name=None):
        """
        Set point position, parents, and if it is fixed for this turn.

        Arguments
        ---------
        x : float, optional
            Position on horizontal axis. The default is 0.
        y : float, optional
            Position on vertical axis. The default is O.
        name : str, optional
            Friendly name for human readability. The default is None.
        joint0 : Union[Joint, tuple[float]], optional
            Linked pivot joint 1 (geometric constraints). The default is None.
        joint1 : Union[Joint, tuple[float]], optional
            Other pivot joint linked. The default is None.
        distance0 : float, optional
            Distance from joint0 to the current Joint. The default is None.
        distance1 : float, optional
            Distance from joint1 to the current Joint. The default is None.
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

        :param joint: Parent joint
        :type joint: Joint
        :returns: Circle is a tuple (abscisse, ordinate, radius).
        :rtype: tuple[float, float, float]

        """
        if self.joint0 is joint:
            return joint.x, joint.y, self.r0
        if self.joint1 is joint:
            return joint.x, joint.y, self.r1
        raise ValueError(f'{joint} is not in joints of {self}')

    def reload(self):
        """Compute the position of pivot joint, use the two linked joints."""
        if self.joint0 is None:
            return
        # Fixed joint as reference. In links, we only keep fixed objects
        ref = tuple(x for x in self.__get_joints__() if None not in x.coord())
        if len(ref) == 0:
            # Don't modify coordinates (irrelevant)
            return
        if len(ref) == 1:
            warnings.warn(
                "Unable to set coordinates of pivot joint {}:"
                "Only one constraint is set."
                "Coordinates unchanged".format(self.name)
            )
        elif len(ref) == 2:
            # Most common case, optimized here
            coco = circle_intersect(
                self.__get_joint_as_circle__(0),
                self.__get_joint_as_circle__(1)
            )
            if coco[0] == 0:
                raise UnbuildableError(self)
            if coco[0] == 1:
                self.x, self.y = coco[1]
            elif coco[0] == 2:
                if sqr_dist(
                        self.coord(), coco[1]
                ) < sqr_dist(self.coord(), coco[2]):
                    self.x, self.y = coco[1]
                else:
                    self.x, self.y = coco[2]
            elif coco[0] == 3:
                warnings.warn(
                    f"Joint {self.name} has an infinite number of"
                    "solutions, position will be arbitrary"
                )
                # We project position on circle of possible positions
                angle = atan2(self.y - self.joint0.y, self.x - self.joint0.x)
                self.x, self.y = cyl_to_cart(self.r0, angle, self.joint0.coord())

    def get_constraints(self):
        """Return the two constraining distances of this joint."""
        return self.r0, self.r1

    def set_constraints(self, distance0=None, distance1=None):
        """Set geometric constraints.

        :param distance0:  (Default value = None)
        :param distance1:  (Default value = None)

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
