"""
Definition of a revolute joint.

It is also called pin joint or hinge joint, and used to be
called a pivot joint in this project.
"""

from __future__ import annotations

import warnings
from math import atan2
from typing import TYPE_CHECKING

from .. import exceptions as pl_exceptions
from .. import geometry as pl_geom
from . import joint as pl_joint

if TYPE_CHECKING:
    from .._types import Circle, Coord


class Revolute(pl_joint.Joint):
    """Center of a revolute joint."""

    __slots__ = "r0", "r1"

    r0: float | None
    r1: float | None

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        joint0: pl_joint.Joint | Coord | None = None,
        joint1: pl_joint.Joint | Coord | None = None,
        distance0: float | None = None,
        distance1: float | None = None,
        name: str | None = None,
    ) -> None:
        """
        Set point position, parents, and if it is fixed for this turn.

        :param x: Position on the horizontal axis.
            (Default value = 0).
        :param y: Position on vertical axis.
            (Default value = 0).
        :param joint0: Linked revolute joint 0 (geometric constraints).
            (Default value = None).
        :param joint1: Linked revolute joint 1 (geometric constraints).
            (Default value = None).
        :param distance0: Distance from joint0 to the current Joint.
            (Default value = None).
        :param distance1: Distance from joint1 to the current Joint.
            (Default value = None).
        :param name: Friendly name for human readability.
            (Default value = None).
        """
        super().__init__(x, y, joint0, joint1, name)
        self.r0, self.r1 = distance0, distance1

    def __get_joint_as_circle__(self, index: int) -> Circle:
        """
        A circle representing the reference anchor as (ref.x, ref.y, ref.distance).

        :param index: Get circle from the first or the second anchor.
        :return: Constraint as a circle.
        """
        if index == 0:
            assert self.joint0 is not None
            assert self.joint0.x is not None and self.joint0.y is not None
            assert self.r0 is not None
            return self.joint0.x, self.joint0.y, self.r0
        if index == 1:
            assert self.joint1 is not None
            assert self.joint1.x is not None and self.joint1.y is not None
            assert self.r1 is not None
            return self.joint1.x, self.joint1.y, self.r1
        raise ValueError(f'index should be 0 or 1, not {index}')

    def circle(self, joint: pl_joint.Joint) -> Circle:
        """
        Return the first link between self and parent as a circle.

        :param joint: Parent joint you want to use.
        :returns: Circle is a tuple (abscissa, ordinate, radius).
        """
        assert joint.x is not None and joint.y is not None
        if self.joint0 is joint:
            assert self.r0 is not None
            return joint.x, joint.y, self.r0
        if self.joint1 is joint:
            assert self.r1 is not None
            return joint.x, joint.y, self.r1
        raise ValueError(f'{joint} is not in joints of {self}')

    def reload(self, dt: float = 1) -> None:
        """Compute the position of revolute joint, use the two linked joints.

        :param dt: Unused, but preserves the object structure.
        """
        if self.joint0 is None:
            return
        # Fixed joint as reference. In links, we only keep fixed objects
        joints = self.__get_joints__()
        ref = tuple(x for x in joints if x is not None and None not in x.coord())
        if len(ref) == 0:
            # Don't change coordinates (irrelevant)
            return
        if len(ref) == 1:
            warnings.warn(
                f"Unable to set coordinates of revolute joint {self.name}:"
                "Only one constraint is set."
                "Coordinates unchanged",
                stacklevel=2,
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
                coord = intersections[1]  # type: ignore[misc]
                assert isinstance(coord, tuple) and len(coord) == 2
                self.x, self.y = coord
            elif intersections[0] == 2:
                assert self.x is not None and self.y is not None
                coord1 = intersections[1]  # type: ignore[misc]
                coord2 = intersections[2]  # type: ignore[misc]
                assert isinstance(coord1, tuple) and len(coord1) == 2
                assert isinstance(coord2, tuple) and len(coord2) == 2
                self.x, self.y = pl_geom.core.get_nearest_point(
                    (self.x, self.y), coord1, coord2
                )
            elif intersections[0] == 3:
                warnings.warn(
                    f"Joint {self.name} has an infinite number of"
                    "solutions, position will be arbitrary",
                    stacklevel=2,
                )
                # We project position on circle of possible positions
                assert self.joint0.x is not None and self.joint0.y is not None
                assert self.x is not None and self.y is not None
                assert self.r0 is not None
                angle = atan2(self.y - self.joint0.y, self.x - self.joint0.x)
                self.x, self.y = pl_geom.cyl_to_cart(
                    self.r0, angle, (self.joint0.x, self.joint0.y)
                )

    def get_constraints(self) -> tuple[float | None, float | None]:
        """Return the two constraining distances of this joint."""
        return self.r0, self.r1

    def set_constraints(
        self,
        distance0: float | None = None,
        distance1: float | None = None,
        *args: float | None,
    ) -> None:
        """Set geometric constraints.

        :param distance0: Distance to the first reference (Default value = None).
        :param distance1: Distance to the second reference (Default value = None).
        :param args: Unused, but preserves the object structure.
        """
        self.r0, self.r1 = distance0 or self.r0, distance1 or self.r1

    def set_anchor0(
        self,
        joint: pl_joint.Joint | Coord,
        distance: float | None = None,
    ) -> None:
        """Set the first anchor for this Joint.

        :param joint: The joint to use as anchor.
        :param distance: Distance to keep constant from the anchor. The default is None.
        """
        self.joint0 = pl_joint.joint_syntax_parser(joint)
        self.set_constraints(distance0=distance)

    def set_anchor1(
        self,
        joint: pl_joint.Joint | Coord,
        distance: float | None = None,
    ) -> None:
        """Set the second anchor for this Joint.

        :param joint: The joint to use as anchor.
        :param distance: Distance to keep constant from the anchor. The default is None.
        """
        self.joint1 = pl_joint.joint_syntax_parser(joint)
        self.set_constraints(distance1=distance)


class Pivot(Revolute):
    """
    Revolute Joint definition.

    .. deprecated :: 0.6.0
        This class has been deprecated in favor of `Revolute` which has a standard name.
        It will be removed in PyLinkage 0.7.0.
    """

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        joint0: pl_joint.Joint | Coord | None = None,
        joint1: pl_joint.Joint | Coord | None = None,
        distance0: float | None = None,
        distance1: float | None = None,
        name: str | None = None,
    ) -> None:
        """
        Set point position, parents, and if it is fixed for this turn.

        :param x: Position on the horizontal axis.
            (Default value = 0).
        :param y: Position on vertical axis.
            (Default value = 0).
        :param joint0: Linked pivot joint 0 (geometric constraints).
            (Default value = None).
        :param joint1: Linked pivot joint 1 (geometric constraints).
            (Default value = None).
        :param distance0: Distance from joint0 to the current Joint.
            (Default value = None).
        :param distance1: Distance from joint1 to the current Joint.
            (Default value = None).
        :param name: Friendly name for human readability.
            (Default value = None).
        """
        warnings.warn(
            "The Pivot class is deprecated in favor of the Revolute class.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            x=x,
            y=y,
            joint0=joint0,
            joint1=joint1,
            distance0=distance0,
            distance1=distance1,
            name=name,
        )
