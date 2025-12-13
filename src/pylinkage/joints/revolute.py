"""
Definition of a revolute joint (RRR dyad).

A revolute joint is positioned at the intersection of two circles,
each centered at a parent joint. This corresponds to an RRR (three
revolute joints) dyad in Assur group theory.

This is a thin wrapper around the solver's solve_revolute function.
"""

import math
import warnings

from .. import exceptions as pl_exceptions
from .._types import Circle, Coord
from ..solver.joints import solve_revolute
from . import joint as pl_joint


class Revolute(pl_joint.Joint):
    """Revolute joint (RRR dyad) - circle-circle intersection.

    The position is computed as the intersection of two circles:
    - Circle 1: centered at joint0 with radius r0
    - Circle 2: centered at joint1 with radius r1

    When two solutions exist, the nearest to the current position
    is chosen (hysteresis for continuity during simulation).
    """

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
        """Compute position using solver (RRR dyad - circle-circle).

        :param dt: Unused, but preserves the object structure.
        """
        if self.joint0 is None:
            return

        # Validate both parents are available
        joints = self.__get_joints__()
        ref = tuple(x for x in joints if x is not None and None not in x.coord())

        if len(ref) == 0:
            return
        if len(ref) == 1:
            warnings.warn(
                f"Unable to set coordinates of revolute joint {self.name}: "
                "Only one constraint is set. Coordinates unchanged",
                stacklevel=2,
            )
            return

        # Validate constraints
        if self.r0 is None or self.r1 is None:
            warnings.warn(
                f"Revolute joint {self.name} missing distance constraints. "
                "Coordinates unchanged",
                stacklevel=2,
            )
            return

        if self.x is None or self.y is None:
            # Initialize to a reasonable position if not set
            self.x = self.joint0.x if self.joint0.x is not None else 0.0
            self.y = self.joint0.y if self.joint0.y is not None else 0.0

        # Check for coincident circles (same center, same radius) - edge case
        # This is a degenerate case where infinite solutions exist
        j0x = self.joint0.x if self.joint0 is not None else 0.0
        j0y = self.joint0.y if self.joint0 is not None else 0.0
        j1x = self.joint1.x if self.joint1 is not None else 0.0
        j1y = self.joint1.y if self.joint1 is not None else 0.0
        dx = (j1x or 0.0) - (j0x or 0.0)
        dy = (j1y or 0.0) - (j0y or 0.0)
        dist_sq = dx * dx + dy * dy
        if dist_sq < 1e-10 and abs(self.r0 - self.r1) < 1e-10:
            warnings.warn(
                f"Joint {self.name} has an infinite number of "
                "solutions, position will be arbitrary",
                stacklevel=2,
            )

        # Delegate to solver function (single source of truth)
        new_x, new_y = solve_revolute(
            self.x, self.y,
            j0x or 0.0, j0y or 0.0,
            self.r0,
            j1x or 0.0, j1y or 0.0,
            self.r1,
        )

        # Handle unbuildable case
        if math.isnan(new_x):
            raise pl_exceptions.UnbuildableError(self)

        self.x, self.y = new_x, new_y

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
