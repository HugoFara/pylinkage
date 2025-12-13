"""
Definition of a prismatic joint (RRP dyad).

A prismatic joint is positioned at the intersection of a circle and a line.
The circle is centered at a revolute anchor, and the line is defined by
two other joints. This corresponds to an RRP (two revolute, one prismatic)
dyad in Assur group theory.

This is a thin wrapper around the solver's solve_linear function.
"""

import math
import warnings

from .. import exceptions as pl_exceptions
from .._types import Coord
from ..solver.joints import solve_linear
from . import joint as pl_joint


class Prismatic(pl_joint.Joint):
    """Prismatic joint (RRP dyad) - circle-line intersection.

    The position is computed as the intersection of:
    - Circle: centered at joint0 with radius revolute_radius
    - Line: passing through joint1 and joint2

    When two solutions exist, the nearest to the current position
    is chosen (hysteresis for continuity during simulation).
    """

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

    def reload(self, dt: float = 1) -> None:
        """Compute position using solver (RRP dyad - circle-line).

        :param dt: Unused, but preserves the object structure.
        """
        if self.joint0 is None:
            raise pl_exceptions.NotCompletelyDefinedError(self, "joint0 is not defined")
        if self.joint1 is None or self.joint2 is None:
            raise pl_exceptions.NotCompletelyDefinedError(self, "joint1 or joint2 is not defined")

        # Validate all required values
        if self.joint0.x is None or self.joint0.y is None:
            raise pl_exceptions.UnbuildableError(
                self, message="joint0 has None coordinates"
            )
        if self.revolute_radius is None:
            raise pl_exceptions.UnbuildableError(
                self, message="revolute_radius is not set"
            )
        if self.joint1.x is None or self.joint1.y is None:
            raise pl_exceptions.UnbuildableError(
                self, message="joint1 has None coordinates"
            )
        if self.joint2.x is None or self.joint2.y is None:
            raise pl_exceptions.UnbuildableError(
                self, message="joint2 has None coordinates"
            )

        if self.x is None or self.y is None:
            # Initialize to a reasonable position
            self.x = self.joint0.x
            self.y = self.joint0.y

        # Delegate to solver function (single source of truth)
        new_x, new_y = solve_linear(
            self.x, self.y,
            self.joint0.x, self.joint0.y,
            self.revolute_radius,
            self.joint1.x, self.joint1.y,
            self.joint2.x, self.joint2.y,
        )

        # Handle unbuildable case
        if math.isnan(new_x):
            raise pl_exceptions.UnbuildableError(self)

        self.x, self.y = new_x, new_y

    def get_constraints(self) -> tuple[float | None]:
        """Return the only distance constraint for this joint."""
        return (self.revolute_radius,)

    def set_constraints(self, distance0: float | None = None, *args: float | None) -> None:
        """Set the only distance constraint for this joint.

        :param distance0: Distance from joint0. (Default value = None).
        :param args: Unused, but preserves the object structure.
        """
        self.revolute_radius = distance0 or self.revolute_radius


class Linear(Prismatic):
    """Deprecated alias for Prismatic joint.

    .. deprecated::
        Use :class:`Prismatic` instead. The ``Linear`` name will be removed
        in a future version.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize with deprecation warning."""
        warnings.warn(
            "Linear is deprecated, use Prismatic instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
