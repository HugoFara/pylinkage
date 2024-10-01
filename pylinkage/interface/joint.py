"""
Definition of the different joints used for pylinkage.
"""
import abc
from math import atan2

from ..geometry import (
    cyl_to_cart, circle_line_from_points_intersection, get_nearest_point
)
from .exceptions import HypostaticError, UnbuildableError, NotCompletelyDefinedError


def joint_syntax_parser(joint):
    """
    Syntactic parser that understand a joint definition.

    :param joint: Input joint definition to be parsed.
    :type joint: Joint | tuple[float, float] | None

    :return: New static joint definition if possible, or None.
    :rtype: Static | None
    """
    if joint is None or isinstance(joint, Joint):
        return joint
    return Static(*joint)


class Joint(abc.ABC):
    """
    Geometric constraint expressed by two joints.
    
    Abstract class should always be inherited.
    """

    __slots__ = "x", "y", "joint0", "joint1", "name"

    def __init__(self, x=0, y=0, joint0=None, joint1=None, name=None):
        """
        Create a Joint abstract object.

        :param x: Position on horizontal axis. The default is 0.
        :type x: float | None
        :param y: Position on vertical axis. The default is O.
        :type y: float | None
        :param name: Friendly name for human readability. Will default to object if None.
        :type name: str | None
        :param joint0: First linked joint (geometric constraints). The default is None.
        :type joint0: Joint | tuple[float, float] | None
        :param joint1: Other revolute joint linked. The default is None.
        :type joint1: Joint | tuple[float, float] | None
        """
        self.x, self.y = x, y
        self.joint0 = joint_syntax_parser(joint0)
        self.joint1 = joint_syntax_parser(joint1)
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
        """
        Return cartesian coordinates.

        :rtype: tuple[float | None, float | None]
        """
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


class Crank(Joint):
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


class Linear(Joint):
    """Define a point constrained by a prismatic joint and a revolute joint."""

    __slots__ = "revolute_radius", "joint2"

    def __init__(
        self,
        x=0,
        y=0,
        joint0=None,
        joint1=None,
        joint2=None,
        revolute_radius=None,
        name=None
    ):
        """
        Set point position, parents, and if it is fixed for this turn.

        :param x: Position on horizontal axis. The default is 0.
        :type x: float | None
        :param y: Position on vertical axis. The default is 0.
        :type y: float | None
        :param joint0: Linked revolute joint 1 (geometric constraints). The default is None.
        :type joint0: Joint | tuple[float, float]
        :param joint1: First joint or point defining the axis. The default is None.
        :type joint1: Joint | tuple[float, float]
        :param joint2: Second joint or point defining the axis. The default is None.
        :type joint2: Joint | tuple[float, float]
        :param revolute_radius: Distance from joint0 to the current Joint. The default is None.
        :type revolute_radius: float | None
        :param name: Friendly name for human readability. The default is None.
        :type name: str | None
        """
        super().__init__(x, y, joint0, joint1, name)
        self.revolute_radius = revolute_radius
        self.joint2 = joint_syntax_parser(joint2)

    def reload(self):
        """Compute position of revolute joint, with the three linked joints."""
        if self.joint0 is None:
            raise NotCompletelyDefinedError(self, "joint0 is not defined")
        positions_circle = (*self.joint0.coord(), self.revolute_radius)
        if None in positions_circle:
            raise UnbuildableError(
                self,
                message="Joint has missing constraints. "
                        "Current constraints are " + str(positions_circle)
            )
        positions = circle_line_from_points_intersection(
            positions_circle,
            self.joint1.coord(),
            self.joint2.coord()
        )
        if len(positions) == 0:
            raise UnbuildableError(self)
        elif len(positions) == 1:
            self.x, self.y = positions[0]
        else:
            # Got to the nearest point
            self.x, self.y = get_nearest_point(self.coord(), positions[0], positions[1])

    def get_constraints(self):
        """Return the only distance constraint for this joint."""
        return tuple([self.revolute_radius])

    def set_constraints(self, distance0=None):
        """Set the only distance constraint for this joint."""
        self.revolute_radius = distance0 or self.revolute_radius
