"""
Definition of the different joints used for pylinkage.
"""
import abc
import warnings
from math import atan2
from .exceptions import HypostaticError, UnbuildableError
from ..geometry import sqr_dist, circle_intersect, cyl_to_cart


class Joint(abc.ABC):
    """Geometric constraint expressed by two joints.
    
    Abstract class should always be inherited.
    """

    __slots__ = "x", "y", "joint0", "joint1", "name"

    def __init__(self, x=0, y=0, joint0=None, joint1=None, name=None):
        """
        Create Joint.

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
        """
        self.x, self.y = x, y
        if joint0 is None or isinstance(joint0, Joint):
            self.joint0 = joint0
        else:
            self.joint0 = Static(*joint0)
        if joint1 is None or isinstance(joint1, Joint):
            self.joint1 = joint1
        else:
            self.joint1 = Static(*joint1)
        self.name = name
        if name is None:
            self.name = str(id(self))

    def __repr__(self):
        """Represent an object with class name, coordinates, name and state."""
        return "{}(x={}, y={}, name={})".format(
            self.__class__.__name__, self.x, self.y, self.name
        )

    def __get_joints__(self):
        """Return constraint joints as a tuple."""
        return self.joint0, self.joint1

    def coord(self):
        """Return cartesian coordinates."""
        return self.x, self.y

    def set_coord(self, *args):
        """Take a sequence or two scalars, and assign them to object x, y.

        :param args: Coordinates to set, either as two elements or as a tuple of 2 elements
        :type args: tuple[float, float] | tuple[tuple[float, float]]

        """
        if len(args) == 1:
            self.x, self.y = args[0]
        else:
            self.x, self.y = args[0], args[1]

    @abc.abstractmethod
    def get_constraints(self):
        """Return geometric constraints applying to this Joint."""
        raise NotImplementedError(
            "You can't call a constraint from an abstract class."
        )

    @abc.abstractmethod
    def set_constraints(self, *args):
        """Set geometric constraints applying to this Joint."""
        raise NotImplementedError(
            "You can't set a constraint from an abstract class."
        )


class Static(Joint):
    """Special case of Joint that should not move.
    
    Mostly used for the frame.
    """

    __slots__ = tuple()

    def __init__(self, x=0, y=0, name=None):
        """
        A Static joint is a point in space to use as anchor by other joints.

        It is NOT a kind of joint as viewed in engineering terms!

        x : float, optional
            Position on horizontal axis. The default is 0.
        y : float, optional
            Position on vertical axis. The default is O.
        name : str, optional
            Friendly name for human readability. The default is None.
        """
        super().__init__(x, y, name=name)

    def reload(self):
        """Do nothing, for consistency only."""
        pass

    def get_constraints(self):
        """Return an empty tuple."""
        return tuple()

    def set_constraints(self, *args):
        """Do nothing, for consistency only.

        :param args: Unused

        """
        pass

    def set_anchor0(self, joint):
        """First joint anchor.

        :param joint: 

        """
        self.joint0 = joint

    def set_anchor1(self, joint):
        """Second joint anchor.

        :param joint: 

        """
        self.joint1 = joint


class Fixed(Joint):
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
            Linked pivot joint 1 (geometric constraints). The default is None.
        joint1 : Union[Joint, tuple[float]], optional
            Other pivot joint linked. The default is None.
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
            raise HypostaticError(f'Not enough constraints for {self}')
        # Rotation angle of local space relative to global
        rot = atan2(self.joint1.y - self.joint0.y,
                    self.joint1.x - self.joint0.x)
        # Position in global space
        self.x, self.y = cyl_to_cart(self.r, self.angle + rot,
                                     self.joint0.coord())

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

        :param joint: 

        """
        self.joint1 = joint


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
        """Return reference as (ref.x, ref.y, ref.distance)."""
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
                if sqr_dist(self.coord(), coco[1]
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
                vect = (
                    (j-i) / abs(j-i) for i, j in zip(coco[1], self.coord())
                )
                self.x, self.y = [
                    i + j * coco[1][2] for i, j in zip(coco[1], vect)
                ]

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


class Crank(Joint):
    """Define a crank joint."""

    __slots__ = "r", "angle"

    def __init__(self, x=None, y=None, joint0=None,
                 distance=None, angle=None, name=None):
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

        :param dt: Fraction of step to do (Default value = 1)
        :type dt: float

        """
        if self.joint0 is None:
            return
        if None in self.joint0.coord():
            raise HypostaticError(
                f'{self.joint0} has None coordinates. '
                f'{self} cannot be calculated'
            )
        # Rotation angle of local space relative to global
        rot = atan2(self.y - self.joint0.y, self.x - self.joint0.x)
        self.x, self.y = cyl_to_cart(
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
