"""
Definition of a linear joint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import exceptions as pl_exceptions
from .. import geometry as geom
from . import joint as pl_joint

if TYPE_CHECKING:
    from .._types import Coord


class Linear(pl_joint.Joint):
    """Define a point constrained by a prismatic joint and a revolute joint."""

    __slots__ = "revolute_radius", "joint2"

    revolute_radius: float | None
    joint2: pl_joint.Joint | pl_joint.Static | None

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        joint0: pl_joint.Joint | Coord | None = None,
        joint1: pl_joint.Joint | Coord | None = None,
        joint2: pl_joint.Joint | Coord | None = None,
        revolute_radius: float | None = None,
        name: str | None = None,
    ) -> None:
        """
        Set point position, parents, and if it is fixed for this turn.

        :param x: Position on horizontal axis. The default is 0.
        :param y: Position on vertical axis. The default is 0.
        :param joint0: Linked revolute joint 1 (geometric constraints). The default is None.
        :param joint1: First joint or point defining the axis. The default is None.
        :param joint2: Second joint or point defining the axis. The default is None.
        :param revolute_radius: Distance from joint0 to the current Joint. The default is None.
        :param name: Friendly name for human readability. The default is None.
        """
        super().__init__(x, y, joint0, joint1, name)
        self.revolute_radius = revolute_radius
        self.joint2 = pl_joint.joint_syntax_parser(joint2)

    def reload(self) -> None:
        """Compute position of revolute joint, with the three linked joints."""
        if self.joint0 is None:
            raise pl_exceptions.NotCompletelyDefinedError(self, "joint0 is not defined")
        positions_circle = (*self.joint0.coord(), self.revolute_radius)
        if None in positions_circle:
            raise pl_exceptions.UnbuildableError(
                self,
                message="Joint has missing constraints. "
                        "Current constraints are " + str(positions_circle)
            )
        positions = geom.circle_line_from_points_intersection(
            positions_circle,
            self.joint1.coord(),
            self.joint2.coord()
        )
        if len(positions) == 0:
            raise pl_exceptions.UnbuildableError(self)
        elif len(positions) == 1:
            self.x, self.y = positions[0]
        else:
            # Got to the nearest point
            self.x, self.y = geom.get_nearest_point(self.coord(), positions[0], positions[1])

    def get_constraints(self) -> tuple[float | None]:
        """Return the only distance constraint for this joint."""
        return (self.revolute_radius,)

    def set_constraints(self, distance0: float | None = None) -> None:
        """Set the only distance constraint for this joint.

        :param distance0: Distance from joint0. (Default value = None).
        """
        self.revolute_radius = distance0 or self.revolute_radius
