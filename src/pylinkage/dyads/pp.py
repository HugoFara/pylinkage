"""PPDyad - line-line intersection.

A PP dyad consists of two prismatic constraints, positioning a joint
at the intersection of two lines. This covers isomers like T_R_T, T_RT_, _TRT_.
"""

from __future__ import annotations

import math

from .. import exceptions as pl_exceptions
from ._base import ConnectedDyad, Dyad, _AnchorProxy


class PPDyad(ConnectedDyad):
    """PP Dyad - line-line intersection.

    Positions a joint at the intersection of two lines:
    - Line 1: defined by line1_anchor1 and line1_anchor2
    - Line 2: defined by line2_anchor1 and line2_anchor2

    This dyad has no distance constraints - its position is fully
    determined by the four line-defining anchor points.

    Attributes:
        line1_anchor1: First point defining line 1.
        line1_anchor2: Second point defining line 1.
        line2_anchor1: First point defining line 2.
        line2_anchor2: Second point defining line 2.

    Example:
        >>> A = Ground(0.0, 0.0, name="A")
        >>> B = Ground(2.0, 0.0, name="B")
        >>> C = Ground(0.0, 1.0, name="C")
        >>> D = Ground(2.0, 2.0, name="D")
        >>> joint = PPDyad(
        ...     line1_anchor1=A,
        ...     line1_anchor2=B,
        ...     line2_anchor1=C,
        ...     line2_anchor2=D,
        ...     name="intersection"
        ... )
    """

    __slots__ = ("line1_anchor1", "line1_anchor2", "line2_anchor1", "line2_anchor2")

    line1_anchor1: Dyad | _AnchorProxy
    line1_anchor2: Dyad | _AnchorProxy
    line2_anchor1: Dyad | _AnchorProxy
    line2_anchor2: Dyad | _AnchorProxy

    def __init__(
        self,
        line1_anchor1: Dyad | _AnchorProxy,
        line1_anchor2: Dyad | _AnchorProxy,
        line2_anchor1: Dyad | _AnchorProxy,
        line2_anchor2: Dyad | _AnchorProxy,
        x: float | None = None,
        y: float | None = None,
        name: str | None = None,
    ) -> None:
        """Create a PP dyad (line-line intersection).

        Args:
            line1_anchor1: First point defining line 1.
            line1_anchor2: Second point defining line 1.
            line2_anchor1: First point defining line 2.
            line2_anchor2: Second point defining line 2.
            x: Initial x position hint (optional).
            y: Initial y position hint (optional).
            name: Human-readable identifier.
        """
        super().__init__(x, y, name)
        self.line1_anchor1 = line1_anchor1
        self.line1_anchor2 = line1_anchor2
        self.line2_anchor1 = line2_anchor1
        self.line2_anchor2 = line2_anchor2

        # Initialize position if not provided
        if self.x is None or self.y is None:
            self._initialize_position()

    @property
    def anchors(self) -> tuple[Dyad, Dyad, Dyad, Dyad]:
        """Return the parent dyads (four line anchors)."""
        return (
            self._resolve_anchor(self.line1_anchor1),
            self._resolve_anchor(self.line1_anchor2),
            self._resolve_anchor(self.line2_anchor1),
            self._resolve_anchor(self.line2_anchor2),
        )

    def _resolve_anchor(self, anchor: Dyad | _AnchorProxy) -> Dyad:
        """Resolve an anchor reference to its Dyad."""
        if isinstance(anchor, _AnchorProxy):
            return anchor._parent
        return anchor

    def _get_anchor_position(
        self, anchor: Dyad | _AnchorProxy
    ) -> tuple[float | None, float | None]:
        """Get the position of an anchor."""
        if isinstance(anchor, _AnchorProxy):
            return anchor.position
        return anchor.position

    def _initialize_position(self) -> None:
        """Compute initial position from anchor positions."""
        l1_p1 = self._get_anchor_position(self.line1_anchor1)
        l1_p2 = self._get_anchor_position(self.line1_anchor2)
        l2_p1 = self._get_anchor_position(self.line2_anchor1)
        l2_p2 = self._get_anchor_position(self.line2_anchor2)

        if any(pos[0] is None or pos[1] is None for pos in [l1_p1, l1_p2, l2_p1, l2_p2]):
            return

        # Type assertions since we just checked for None above
        assert l1_p1[0] is not None and l1_p2[0] is not None
        assert l2_p1[0] is not None and l2_p2[0] is not None
        assert l1_p1[1] is not None and l1_p2[1] is not None
        assert l2_p1[1] is not None and l2_p2[1] is not None

        # Use centroid of line-defining points as initial guess
        self.x = (l1_p1[0] + l1_p2[0] + l2_p1[0] + l2_p2[0]) / 4.0
        self.y = (l1_p1[1] + l1_p2[1] + l2_p1[1] + l2_p2[1]) / 4.0

        # Try to solve for actual position
        try:
            self.reload(0)
        except pl_exceptions.UnbuildableError:
            # Reset to centroid if unbuildable at init
            # Type assertions are valid since we already checked for None above
            assert l1_p1[0] is not None and l1_p2[0] is not None
            assert l2_p1[0] is not None and l2_p2[0] is not None
            assert l1_p1[1] is not None and l1_p2[1] is not None
            assert l2_p1[1] is not None and l2_p2[1] is not None
            self.x = (l1_p1[0] + l1_p2[0] + l2_p1[0] + l2_p2[0]) / 4.0
            self.y = (l1_p1[1] + l1_p2[1] + l2_p1[1] + l2_p2[1]) / 4.0

    def get_constraints(self) -> tuple[()]:
        """Return the constraints (none for PP dyad).

        PP dyads have no distance constraints - position is fully
        determined by the four anchor points.

        Returns:
            Empty tuple (no constraints).
        """
        return ()

    def set_constraints(
        self,
        *args: float | None,
    ) -> None:
        """Set constraints (no-op for PP dyad).

        PP dyads have no constraints to set.

        Args:
            *args: Ignored (for interface compatibility).
        """
        pass

    def reload(self, dt: float = 1) -> None:
        """Recompute position using line-line intersection.

        Args:
            dt: Time step (unused for PP, but required for interface).

        Raises:
            UnbuildableError: If lines are parallel (no intersection).
        """
        from ..solver.joints import solve_line_line

        l1_p1 = self._get_anchor_position(self.line1_anchor1)
        l1_p2 = self._get_anchor_position(self.line1_anchor2)
        l2_p1 = self._get_anchor_position(self.line2_anchor1)
        l2_p2 = self._get_anchor_position(self.line2_anchor2)

        if any(pos[0] is None or pos[1] is None for pos in [l1_p1, l1_p2, l2_p1, l2_p2]):
            return  # Parents not positioned yet

        new_x, new_y = solve_line_line(
            l1_p1[0],
            l1_p1[1],
            l1_p2[0],
            l1_p2[1],
            l2_p1[0],
            l2_p1[1],
            l2_p2[0],
            l2_p2[1],
        )

        if math.isnan(new_x):
            raise pl_exceptions.UnbuildableError(self)

        self.x, self.y = new_x, new_y
