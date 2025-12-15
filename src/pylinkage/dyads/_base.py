"""Base classes for Assur group building blocks (dyads).

This module defines the abstract base classes for all dyad types:
- Dyad: Base class for all dyads
- ConnectedDyad: Base for dyads with parent connections
- BinaryDyad: Base for dyads connecting two parents
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mechanism.joint import Joint
    from ..mechanism.link import Link


class Dyad(ABC):
    """Base class for all Assur group building blocks.

    A dyad represents a kinematic element in a planar linkage mechanism.
    Each dyad has a position (x, y) and can compute its constraints.

    Attributes:
        x: Horizontal position coordinate.
        y: Vertical position coordinate.
        name: Human-readable identifier.
    """

    __slots__ = ("x", "y", "name", "_mechanism_joint", "_mechanism_links")

    x: float | None
    y: float | None
    name: str
    _mechanism_joint: Joint | None
    _mechanism_links: list[Link]

    def __init__(
        self,
        x: float | None,
        y: float | None,
        name: str | None = None,
    ) -> None:
        """Initialize a dyad.

        Args:
            x: Horizontal position coordinate.
            y: Vertical position coordinate.
            name: Human-readable identifier. Defaults to object id.
        """
        self.x = x
        self.y = y
        self.name = name if name is not None else str(id(self))
        self._mechanism_joint = None
        self._mechanism_links = []

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, name={self.name!r})"

    @property
    def position(self) -> tuple[float | None, float | None]:
        """Return current (x, y) position."""
        return (self.x, self.y)

    def coord(self) -> tuple[float | None, float | None]:
        """Return cartesian coordinates (alias for position)."""
        return self.position

    def set_coord(self, x: float | None, y: float | None) -> None:
        """Set cartesian coordinates.

        Args:
            x: Horizontal position coordinate.
            y: Vertical position coordinate.
        """
        self.x = x
        self.y = y

    @abstractmethod
    def get_constraints(self) -> tuple[float | None, ...]:
        """Return constraint values for optimization.

        Returns:
            Tuple of constraint values (distances, angles, etc.).
        """
        ...

    @abstractmethod
    def set_constraints(self, *args: float | None) -> None:
        """Set constraint values from optimization.

        Args:
            *args: Constraint values to apply.
        """
        ...

    @abstractmethod
    def reload(self, dt: float = 1) -> None:
        """Recompute position based on parent positions.

        Args:
            dt: Time step or fraction of movement.
        """
        ...


class ConnectedDyad(Dyad):
    """Base class for dyads that connect to parent joints.

    Connected dyads have one or more anchor points that they
    reference for position computation.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def anchors(self) -> tuple[Dyad, ...]:
        """Return the parent dyads this connects to.

        Returns:
            Tuple of parent Dyad objects.
        """
        ...


class BinaryDyad(ConnectedDyad):
    """Base class for dyads connecting exactly two parents.

    Binary dyads have two anchor points and typically compute
    their position as an intersection (circle-circle, circle-line, etc.).
    """

    __slots__ = ("anchor1", "anchor2")

    anchor1: Dyad | _AnchorProxy
    anchor2: Dyad | _AnchorProxy

    @property
    def anchors(self) -> tuple[Dyad, Dyad]:
        """Return the two parent dyads."""
        return (self._resolve_anchor(self.anchor1), self._resolve_anchor(self.anchor2))

    def _resolve_anchor(self, anchor: Dyad | _AnchorProxy) -> Dyad:
        """Resolve an anchor reference to its Dyad.

        Args:
            anchor: Either a Dyad or an _AnchorProxy.

        Returns:
            The resolved Dyad object.
        """
        if isinstance(anchor, _AnchorProxy):
            return anchor._parent
        return anchor

    def _get_anchor_position(
        self, anchor: Dyad | _AnchorProxy
    ) -> tuple[float | None, float | None]:
        """Get the position of an anchor.

        Args:
            anchor: Either a Dyad or an _AnchorProxy.

        Returns:
            The (x, y) position of the anchor.
        """
        if isinstance(anchor, _AnchorProxy):
            return anchor.position
        return anchor.position


class _AnchorProxy:
    """Proxy object for accessing a dyad's output position.

    Used by Crank.output to provide a consistent interface
    for other dyads to connect to.
    """

    __slots__ = ("_parent",)

    def __init__(self, parent: Dyad) -> None:
        """Initialize proxy.

        Args:
            parent: The dyad this proxy represents.
        """
        self._parent = parent

    @property
    def position(self) -> tuple[float | None, float | None]:
        """Return the parent's position."""
        return self._parent.position

    @property
    def x(self) -> float | None:
        """Return the parent's x coordinate."""
        return self._parent.x

    @property
    def y(self) -> float | None:
        """Return the parent's y coordinate."""
        return self._parent.y

    @property
    def name(self) -> str:
        """Return the parent's name."""
        return self._parent.name
