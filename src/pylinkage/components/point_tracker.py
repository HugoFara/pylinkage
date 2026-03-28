"""PointTracker - sensor component for tracking positions on links.

A PointTracker observes a position at fixed distance/angle from two anchors.
Unlike FixedDyad, it is semantically a sensor that doesn't affect kinematics
and has no optimizable constraints.
"""

from __future__ import annotations

from ._base import Component, ConnectedComponent, _AnchorProxy


class PointTracker(ConnectedComponent):
    """A sensor component for tracking positions on a link.

    PointTracker computes its position at a fixed distance and angle from
    anchor1, with the angle measured relative to the line from anchor1 to anchor2.
    This is functionally identical to FixedDyad but is semantically a "sensor"
    that observes without contributing constraints to optimization.

    Use PointTracker when you want to:
    - Track a coupler point on a link
    - Observe a position on a mechanism for analysis
    - Add tracer points without affecting optimization

    Use FixedDyad when:
    - The distance/angle are parameters to optimize
    - The point is part of the mechanism structure

    Attributes:
        anchor1: First anchor (origin for polar coordinates).
        anchor2: Second anchor (defines reference direction).
        distance: Distance from anchor1 to this tracker.
        angle: Angle offset from anchor1->anchor2 direction (radians).

    Example:
        >>> O1 = Ground(0.0, 0.0, name="O1")
        >>> crank = Crank(anchor=O1, radius=1.0)
        >>> O2 = Ground(2.0, 0.0, name="O2")
        >>> # Track a point at 45 degrees from crank->O2 line
        >>> tracker = PointTracker(
        ...     anchor1=crank.output,
        ...     anchor2=O2,
        ...     distance=0.5,
        ...     angle=math.pi/4,
        ...     name="tracer_point"
        ... )
    """

    __slots__ = ("anchor1", "anchor2", "distance", "angle")

    anchor1: Component | _AnchorProxy
    anchor2: Component | _AnchorProxy
    distance: float
    angle: float

    def __init__(
        self,
        anchor1: Component | _AnchorProxy,
        anchor2: Component | _AnchorProxy,
        distance: float,
        angle: float,
        name: str | None = None,
    ) -> None:
        """Create a point tracker (sensor).

        Args:
            anchor1: First anchor (origin for polar coordinates).
            anchor2: Second anchor (defines reference direction).
            distance: Distance from anchor1 to this tracker.
            angle: Angle offset from anchor1->anchor2 direction (radians).
            name: Human-readable identifier.
        """
        super().__init__(None, None, name)
        self.anchor1 = anchor1
        self.anchor2 = anchor2
        self.distance = distance
        self.angle = angle

        # Initialize position
        self._initialize_position()

    def _initialize_position(self) -> None:
        """Compute position from anchor positions."""
        a1_pos = self._get_anchor_position(self.anchor1)
        a2_pos = self._get_anchor_position(self.anchor2)

        if a1_pos[0] is None or a1_pos[1] is None or a2_pos[0] is None or a2_pos[1] is None:
            return

        # Compute position deterministically
        self.reload(0)

    @property
    def anchors(self) -> tuple[Component, Component]:
        """Return the two parent anchors."""
        return (self._resolve_anchor(self.anchor1), self._resolve_anchor(self.anchor2))

    def _resolve_anchor(self, anchor: Component | _AnchorProxy) -> Component:
        """Resolve an anchor reference to its Component."""
        if isinstance(anchor, _AnchorProxy):
            return anchor._parent
        return anchor

    def _get_anchor_position(
        self, anchor: Component | _AnchorProxy
    ) -> tuple[float | None, float | None]:
        """Get the position of an anchor."""
        if isinstance(anchor, _AnchorProxy):
            return anchor.position
        return anchor.position

    def get_constraints(self) -> tuple[()]:
        """Return empty tuple - PointTracker has no optimizable constraints.

        PointTracker is a sensor/observer; its distance and angle are fixed
        and should not be included in optimization bounds.

        Returns:
            Empty tuple.
        """
        return ()

    def set_constraints(self, *args: float | None) -> None:
        """No-op - PointTracker has no optimizable constraints.

        Args:
            *args: Ignored.
        """
        pass

    def reload(self, dt: float = 1) -> None:
        """Recompute position using polar projection.

        The position is always deterministic - no ambiguity.

        Args:
            dt: Time step (unused for PointTracker, required for interface).
        """
        from ..solver.joints import solve_fixed

        a1_pos = self._get_anchor_position(self.anchor1)
        a2_pos = self._get_anchor_position(self.anchor2)

        if a1_pos[0] is None or a1_pos[1] is None:
            return  # Parent not positioned yet
        if a2_pos[0] is None or a2_pos[1] is None:
            return  # Parent not positioned yet

        self.x, self.y = solve_fixed(
            a1_pos[0],
            a1_pos[1],
            a2_pos[0],
            a2_pos[1],
            self.distance,
            self.angle,
        )
