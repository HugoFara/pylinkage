"""Base classes for Assur group dyads.

This module defines the base class for binary dyads (Assur groups with
two anchor points).
"""

from __future__ import annotations

from ..components import Component, ConnectedComponent, _AnchorProxy

# Re-export for backwards compatibility and use by dyad classes
__all__ = ["BinaryDyad", "Dyad", "ConnectedDyad", "_AnchorProxy", "Component", "ConnectedComponent"]

# Backwards compatibility aliases
Dyad = Component
ConnectedDyad = ConnectedComponent


class BinaryDyad(ConnectedComponent):
    """Base class for dyads connecting exactly two parents.

    Binary dyads have two anchor points and typically compute
    their position as an intersection (circle-circle, circle-line, etc.).
    These are the true Assur groups that add 0 degrees of freedom.
    """

    __slots__ = ("anchor1", "anchor2")

    anchor1: Component | _AnchorProxy
    anchor2: Component | _AnchorProxy

    @property
    def anchors(self) -> tuple[Component, Component]:
        """Return the two parent dyads."""
        return (self._resolve_anchor(self.anchor1), self._resolve_anchor(self.anchor2))

    def _resolve_anchor(self, anchor: Component | _AnchorProxy) -> Component:
        """Resolve an anchor reference to its Component.

        Args:
            anchor: Either a Component or an _AnchorProxy.

        Returns:
            The resolved Component object.
        """
        if isinstance(anchor, _AnchorProxy):
            return anchor._parent
        return anchor

    def _get_anchor_position(
        self, anchor: Component | _AnchorProxy
    ) -> tuple[float | None, float | None]:
        """Get the position of an anchor.

        Args:
            anchor: Either a Component or an _AnchorProxy.

        Returns:
            The (x, y) position of the anchor.
        """
        if isinstance(anchor, _AnchorProxy):
            return anchor.position
        return anchor.position
