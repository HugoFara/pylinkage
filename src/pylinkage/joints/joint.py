"""
Definition of the different joints used for pylinkage.
"""


import abc

from .._types import Constraints, Coord, MaybeCoord


def joint_syntax_parser(joint: "Joint | Coord | None") -> "Joint | Static | None":
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

    x: float | None
    y: float | None
    joint0: "Joint | Static | None"
    joint1: "Joint | Static | None"
    name: str

    def __init__(
        self,
        x: float | None = 0,
        y: float | None = 0,
        joint0: "Joint | Coord | None" = None,
        joint1: "Joint | Coord | None" = None,
        name: str | None = None,
    ) -> None:
        """
        Create a Joint abstract object.

        :param x: Position on horizontal axis. The default is 0.
        :param y: Position on vertical axis. The default is O.
        :param name: Friendly name for human readability. Will default to object if None.
        :param joint0: First linked joint (geometric constraints). The default is None.
        :param joint1: Other revolute joint linked. The default is None.
        """
        self.x, self.y = x, y
        self.joint0 = joint_syntax_parser(joint0)
        self.joint1 = joint_syntax_parser(joint1)
        self.name = name if name is not None else str(id(self))

    def __repr__(self) -> str:
        """Represent an object with class name, coordinates, name and state."""
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, name={self.name})"

    def __get_joints__(self) -> "tuple[Joint | Static | None, Joint | Static | None]":
        """Return constraint joints as a tuple."""
        return self.joint0, self.joint1

    def coord(self) -> tuple[float | None, float | None]:
        """Return cartesian coordinates."""
        return self.x, self.y

    def set_coord(
        self,
        *args: float | None | Coord | MaybeCoord,
    ) -> None:
        """Take a sequence or two scalars, and assign them to object x, y.

        :param args: Coordinates to set, either as two elements or as a tuple of 2 elements.
        """
        if len(args) == 1:
            coord = args[0]
            if isinstance(coord, (tuple, list)):
                self.x, self.y = coord[0], coord[1]
            else:
                raise TypeError("Single argument must be a sequence of two floats")
        else:
            x, y = args[0], args[1]
            self.x = float(x) if x is not None else None  # type: ignore[arg-type]
            self.y = float(y) if y is not None else None  # type: ignore[arg-type]

    @abc.abstractmethod
    def get_constraints(self) -> Constraints:
        """Return geometric constraints applying to this Joint."""
        raise NotImplementedError(
            "You can't call a constraint from an abstract class."
        )

    @abc.abstractmethod
    def set_constraints(self, *args: float | None) -> None:
        """Set geometric constraints applying to this Joint."""
        raise NotImplementedError(
            "You can't set a constraint from an abstract class."
        )

    @abc.abstractmethod
    def reload(self, dt: float = 1) -> None:
        """Recompute joint coordinates based on constraints and parent joints.

        :param dt: Time step or fraction of movement. The default is 1.
        """
        raise NotImplementedError(
            "You can't reload from an abstract class."
        )


class Static(Joint):
    """Special case of Joint that should not move.

    Mostly used for the frame.
    """

    __slots__ = ()

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        name: str | None = None,
    ) -> None:
        """
        A Static joint is a point in space to use as anchor by other joints.

        It is NOT a kind of joint as viewed in engineering terms!

        :param x: Position on horizontal axis. The default is 0.
        :param y: Position on vertical axis. The default is O.
        :param name: Friendly name for human readability. The default is None.
        """
        super().__init__(x, y, name=name)

    def reload(self, dt: float = 1) -> None:
        """Do nothing, for consistency only.

        :param dt: Unused, but preserves the object structure.
        """
        pass

    def get_constraints(self) -> tuple[()]:
        """Return an empty tuple."""
        return ()

    def set_constraints(self, *args: float | None) -> None:
        """Do nothing, for consistency only.

        :param args: Unused.
        """
        pass

    def set_anchor0(self, joint: "Joint | Coord") -> None:
        """First joint anchor.

        :param joint: Joint to set as anchor.
        """
        self.joint0 = joint_syntax_parser(joint)

    def set_anchor1(self, joint: "Joint | Coord") -> None:
        """Second joint anchor.

        :param joint: Joint to set as anchor.
        """
        self.joint1 = joint_syntax_parser(joint)
