"""RRPDyad - circle-line intersection (slider mechanism).

An RRP dyad consists of a revolute connection to an anchor plus
a prismatic (sliding) connection along a line.
"""

from __future__ import annotations

import math

from .. import exceptions as pl_exceptions
from ._base import ConnectedDyad, Dyad, _AnchorProxy


class RRPDyad(ConnectedDyad):
    """RRP Dyad - circle-line intersection (slider mechanism).

    Positions a joint at the intersection of:
    - A circle centered at the revolute anchor
    - A line defined by two line anchor points

    The joint slides along the line while maintaining a fixed distance
    from the revolute anchor.

    When two solutions exist, the nearest to current position is chosen
    (hysteresis for continuity during simulation).

    Attributes:
        revolute_anchor: Joint connected by revolute pair.
        line_anchor1: First joint defining the sliding line.
        line_anchor2: Second joint defining the sliding line.
        distance: Distance from revolute_anchor to this joint.

    Example:
        >>> O1 = Ground(0.0, 0.0, name="O1")
        >>> L1 = Ground(0.0, 1.0, name="L1")
        >>> L2 = Ground(2.0, 1.0, name="L2")
        >>> crank = Crank(anchor=O1, radius=1.0)
        >>> slider = RRPDyad(
        ...     revolute_anchor=crank.output,
        ...     line_anchor1=L1,
        ...     line_anchor2=L2,
        ...     distance=1.5,
        ...     name="slider"
        ... )
    """

    __slots__ = ("revolute_anchor", "line_anchor1", "line_anchor2", "distance")

    revolute_anchor: Dyad | _AnchorProxy
    line_anchor1: Dyad | _AnchorProxy
    line_anchor2: Dyad | _AnchorProxy
    distance: float

    def __init__(
        self,
        revolute_anchor: Dyad | _AnchorProxy,
        line_anchor1: Dyad | _AnchorProxy,
        line_anchor2: Dyad | _AnchorProxy,
        distance: float,
        x: float | None = None,
        y: float | None = None,
        name: str | None = None,
    ) -> None:
        """Create an RRP dyad (circle-line intersection).

        Args:
            revolute_anchor: Joint connected by revolute pair.
            line_anchor1: First joint defining the sliding line.
            line_anchor2: Second joint defining the sliding line.
            distance: Distance from revolute_anchor to this joint.
            x: Initial x position hint (optional).
            y: Initial y position hint (optional).
            name: Human-readable identifier.
        """
        super().__init__(x, y, name)
        self.revolute_anchor = revolute_anchor
        self.line_anchor1 = line_anchor1
        self.line_anchor2 = line_anchor2
        self.distance = distance

        # Initialize position if not provided
        if self.x is None or self.y is None:
            self._initialize_position()

    @property
    def anchors(self) -> tuple[Dyad, Dyad, Dyad]:
        """Return the parent dyads (revolute anchor, line anchors)."""
        ra = (
            self.revolute_anchor._parent
            if isinstance(self.revolute_anchor, _AnchorProxy)
            else self.revolute_anchor
        )
        la1 = (
            self.line_anchor1._parent
            if isinstance(self.line_anchor1, _AnchorProxy)
            else self.line_anchor1
        )
        la2 = (
            self.line_anchor2._parent
            if isinstance(self.line_anchor2, _AnchorProxy)
            else self.line_anchor2
        )
        return (ra, la1, la2)

    def _get_anchor_position(
        self, anchor: Dyad | _AnchorProxy
    ) -> tuple[float | None, float | None]:
        """Get the position of an anchor."""
        if isinstance(anchor, _AnchorProxy):
            return anchor.position
        return anchor.position

    def _initialize_position(self) -> None:
        """Compute initial position from anchor positions."""
        ra_pos = self._get_anchor_position(self.revolute_anchor)
        l1_pos = self.line_anchor1.position
        l2_pos = self.line_anchor2.position

        if (
            ra_pos[0] is None
            or ra_pos[1] is None
            or l1_pos[0] is None
            or l1_pos[1] is None
            or l2_pos[0] is None
            or l2_pos[1] is None
        ):
            return

        # Use revolute anchor position as initial guess
        self.x, self.y = ra_pos

        # Try to solve for actual position
        try:
            self.reload(0)
        except pl_exceptions.UnbuildableError:
            # Reset to revolute anchor if unbuildable at init
            self.x, self.y = ra_pos

    def get_constraints(self) -> tuple[float]:
        """Return the distance constraint.

        Returns:
            Tuple containing the distance to revolute anchor.
        """
        return (self.distance,)

    def set_constraints(
        self,
        distance: float | None = None,
        *args: float | None,
    ) -> None:
        """Set the distance constraint.

        Args:
            distance: New distance to revolute anchor.
            *args: Ignored (for interface compatibility).
        """
        if distance is not None:
            self.distance = distance

    def reload(self, dt: float = 1) -> None:
        """Recompute position using circle-line intersection.

        Args:
            dt: Time step (unused for RRP, but required for interface).

        Raises:
            UnbuildableError: If circle doesn't intersect line.
        """
        from ..solver.joints import solve_linear

        ra_pos = self._get_anchor_position(self.revolute_anchor)
        l1_pos = self.line_anchor1.position
        l2_pos = self.line_anchor2.position

        if ra_pos[0] is None or ra_pos[1] is None:
            return  # Parent not positioned yet
        if l1_pos[0] is None or l1_pos[1] is None:
            return
        if l2_pos[0] is None or l2_pos[1] is None:
            return

        # Use current position for hysteresis, or revolute anchor if unset
        curr_x = self.x if self.x is not None else ra_pos[0]
        curr_y = self.y if self.y is not None else ra_pos[1]

        new_x, new_y = solve_linear(
            curr_x,
            curr_y,
            ra_pos[0],
            ra_pos[1],
            self.distance,
            l1_pos[0],
            l1_pos[1],
            l2_pos[0],
            l2_pos[1],
        )

        if math.isnan(new_x):
            raise pl_exceptions.UnbuildableError(self)

        self.x, self.y = new_x, new_y
