"""Ground dyad - fixed point on the frame.

A ground point is a stationary reference point in a planar mechanism.
It does not move during simulation.
"""

from __future__ import annotations

from ._base import Dyad


class Ground(Dyad):
    """A fixed point on the frame (ground link).

    Ground dyads define the stationary reference points of a mechanism.
    They don't move during simulation and serve as anchors for other
    kinematic elements.

    Example:
        >>> O1 = Ground(0.0, 0.0, name="O1")
        >>> O2 = Ground(2.0, 0.0, name="O2")
        >>> O1.position
        (0.0, 0.0)
    """

    __slots__ = ()

    def __init__(
        self,
        x: float,
        y: float,
        name: str | None = None,
    ) -> None:
        """Create a ground point.

        Args:
            x: Horizontal position coordinate.
            y: Vertical position coordinate.
            name: Human-readable identifier.
        """
        super().__init__(x, y, name)

    def get_constraints(self) -> tuple[()]:
        """Return empty tuple - ground has no constraints to optimize."""
        return ()

    def set_constraints(self, *args: float | None) -> None:
        """No-op - ground has no constraints."""
        pass

    def reload(self, dt: float = 1) -> None:
        """No-op - ground doesn't move."""
        pass
