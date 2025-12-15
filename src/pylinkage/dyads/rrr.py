"""RRRDyad - circle-circle intersection (two revolute joints meeting at one).

The most common Assur group, consisting of two links connected by
revolute joints, meeting at a computed internal revolute joint.
"""

from __future__ import annotations

import math

from .. import exceptions as pl_exceptions
from ._base import BinaryDyad, Dyad, _AnchorProxy


class RRRDyad(BinaryDyad):
    """RRR Dyad - circle-circle intersection.

    Positions a joint at the intersection of two circles centered
    at the anchor points. This is the most common Assur group.

    When two solutions exist, the nearest to current position is chosen
    (hysteresis for continuity during simulation).

    Attributes:
        anchor1: First connection point.
        anchor2: Second connection point.
        distance1: Distance from anchor1 to this joint.
        distance2: Distance from anchor2 to this joint.

    Example:
        >>> O1 = Ground(0.0, 0.0, name="O1")
        >>> O2 = Ground(2.0, 0.0, name="O2")
        >>> crank = Crank(anchor=O1, radius=1.0)
        >>> rocker = RRRDyad(
        ...     anchor1=crank.output,
        ...     anchor2=O2,
        ...     distance1=2.0,
        ...     distance2=1.5,
        ...     name="rocker"
        ... )
    """

    __slots__ = ("distance1", "distance2")

    distance1: float
    distance2: float

    def __init__(
        self,
        anchor1: Dyad | _AnchorProxy,
        anchor2: Dyad | _AnchorProxy,
        distance1: float,
        distance2: float,
        x: float | None = None,
        y: float | None = None,
        name: str | None = None,
    ) -> None:
        """Create an RRR dyad (circle-circle intersection).

        Args:
            anchor1: First anchor joint (must be positioned).
            anchor2: Second anchor joint (must be positioned).
            distance1: Distance from anchor1 to this joint.
            distance2: Distance from anchor2 to this joint.
            x: Initial x position hint (optional).
            y: Initial y position hint (optional).
            name: Human-readable identifier.
        """
        super().__init__(x, y, name)
        self.anchor1 = anchor1
        self.anchor2 = anchor2
        self.distance1 = distance1
        self.distance2 = distance2

        # Initialize position if not provided
        if self.x is None or self.y is None:
            self._initialize_position()

    def _initialize_position(self) -> None:
        """Compute initial position from anchor positions."""
        a1x, a1y = self._get_anchor_position(self.anchor1)
        a2x, a2y = self._get_anchor_position(self.anchor2)

        if a1x is None or a1y is None or a2x is None or a2y is None:
            # Can't compute yet, leave as None
            return

        # Use midpoint offset as initial guess
        self.x = (a1x + a2x) / 2
        self.y = (a1y + a2y) / 2 + 0.1  # Slight offset to avoid degenerate cases

        # Try to solve for actual position
        try:
            self.reload(0)  # dt=0 means no crank movement
        except pl_exceptions.UnbuildableError:
            # Reset to midpoint if unbuildable at init
            self.x = (a1x + a2x) / 2
            self.y = (a1y + a2y) / 2 + 0.1

    def get_constraints(self) -> tuple[float, float]:
        """Return the two distance constraints.

        Returns:
            Tuple of (distance1, distance2).
        """
        return (self.distance1, self.distance2)

    def set_constraints(
        self,
        distance1: float | None = None,
        distance2: float | None = None,
        *args: float | None,
    ) -> None:
        """Set the distance constraints.

        Args:
            distance1: New distance to anchor1.
            distance2: New distance to anchor2.
            *args: Ignored (for interface compatibility).
        """
        if distance1 is not None:
            self.distance1 = distance1
        if distance2 is not None:
            self.distance2 = distance2

    def reload(self, dt: float = 1) -> None:
        """Recompute position using circle-circle intersection.

        Args:
            dt: Time step (unused for RRR, but required for interface).

        Raises:
            UnbuildableError: If the circles don't intersect.
        """
        from ..solver.joints import solve_revolute

        a1x, a1y = self._get_anchor_position(self.anchor1)
        a2x, a2y = self._get_anchor_position(self.anchor2)

        if a1x is None or a1y is None:
            return  # Parent not positioned yet
        if a2x is None or a2y is None:
            return  # Parent not positioned yet

        # Use current position for hysteresis, or midpoint if unset
        curr_x = self.x if self.x is not None else (a1x + a2x) / 2
        curr_y = self.y if self.y is not None else (a1y + a2y) / 2

        new_x, new_y = solve_revolute(
            curr_x,
            curr_y,
            a1x,
            a1y,
            self.distance1,
            a2x,
            a2y,
            self.distance2,
        )

        if math.isnan(new_x):
            raise pl_exceptions.UnbuildableError(self)

        self.x, self.y = new_x, new_y
