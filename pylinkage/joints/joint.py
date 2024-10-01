"""
Definition of the different joints used for pylinkage.
"""
import abc


def joint_syntax_parser(joint):
    """
    Syntactic parser that understand a joint definition.

    :param joint: Input joint definition to be parsed.
    :type joint: Joint | tuple[float, float] | None

    :return: New static joint definition if possible, or None.
    :rtype: Joint | Static | None
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

        :param joint: Joint to set as anchor.

        """
        self.joint0 = joint

    def set_anchor1(self, joint):
        """Second joint anchor.

        :param joint: Joint to set as anchor.

        """
        self.joint1 = joint
