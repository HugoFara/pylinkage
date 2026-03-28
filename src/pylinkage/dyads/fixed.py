"""FixedDyad - deterministic polar constraint.

A fixed dyad positions a joint at a fixed distance and angle
relative to two anchor joints. Unlike RRR dyad, the position
is deterministic (no ambiguity).
"""

from __future__ import annotations

from ._base import BinaryDyad, Dyad, _AnchorProxy


class FixedDyad(BinaryDyad):
    """Fixed Dyad - deterministic polar projection.

    Positions a joint at a fixed distance and angle from anchor1,
    with the angle measured relative to the line from anchor1 to anchor2.

    Unlike RRRDyad which has two possible solutions, FixedDyad always
    has exactly one deterministic solution.

    Attributes:
        anchor1: First anchor (origin for polar coordinates).
        anchor2: Second anchor (defines reference direction).
        distance: Distance from anchor1 to this joint.
        angle: Angle offset from anchor1->anchor2 direction (radians).

    Example:
        >>> O1 = Ground(0.0, 0.0, name="O1")
        >>> O2 = Ground(2.0, 0.0, name="O2")
        >>> crank = Crank(anchor=O1, radius=1.0)
        >>> # Create a point at 90 degrees from crank->O2 line
        >>> fixed = FixedDyad(
        ...     anchor1=crank.output,
        ...     anchor2=O2,
        ...     distance=1.0,
        ...     angle=math.pi/2,
        ...     name="coupler_point"
        ... )
    """

    __slots__ = ("distance", "angle")

    distance: float
    angle: float

    def __init__(
        self,
        anchor1: Dyad | _AnchorProxy,
        anchor2: Dyad | _AnchorProxy,
        distance: float,
        angle: float,
        name: str | None = None,
    ) -> None:
        """Create a fixed dyad (polar projection).

        Args:
            anchor1: First anchor (origin for polar coordinates).
            anchor2: Second anchor (defines reference direction).
            distance: Distance from anchor1 to this joint.
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

    def get_constraints(self) -> tuple[float, float]:
        """Return the distance and angle constraints.

        Returns:
            Tuple of (distance, angle).
        """
        return (self.distance, self.angle)

    def set_constraints(
        self,
        distance: float | None = None,
        angle: float | None = None,
        *args: float | None,
    ) -> None:
        """Set the distance and angle constraints.

        Args:
            distance: New distance from anchor1.
            angle: New angle offset (radians).
            *args: Ignored (for interface compatibility).
        """
        if distance is not None:
            self.distance = distance
        if angle is not None:
            self.angle = angle

    def reload(self, dt: float = 1) -> None:
        """Recompute position using polar projection.

        The position is always deterministic - no ambiguity.

        Args:
            dt: Time step (unused for Fixed, but required for interface).
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
