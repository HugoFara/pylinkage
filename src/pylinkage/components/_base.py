"""Base classes for kinematic building blocks.

This module defines the abstract base classes for all kinematic components:
- Component: Base class for all kinematic elements
- ConnectedComponent: Base for components with parent connections
- _AnchorProxy: Proxy for output position access
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mechanism.joint import Joint
    from ..mechanism.link import Link


class Component(ABC):
    """Base class for all kinematic building blocks.

    A component represents a kinematic element in a planar linkage mechanism.
    Each component has a position (x, y) and can compute its constraints.

    Attributes:
        x: Horizontal position coordinate.
        y: Vertical position coordinate.
        name: Human-readable identifier.
        velocity: Linear velocity (vx, vy) in units/s. None if not computed.
        acceleration: Linear acceleration (ax, ay) in units/s². None if not computed.
    """

    __slots__ = (
        "x",
        "y",
        "name",
        "_mechanism_joint",
        "_mechanism_links",
        "_velocity",
        "_acceleration",
    )

    x: float | None
    y: float | None
    name: str
    _mechanism_joint: Joint | None
    _mechanism_links: list[Link]
    _velocity: tuple[float, float] | None
    _acceleration: tuple[float, float] | None

    def __init__(
        self,
        x: float | None,
        y: float | None,
        name: str | None = None,
    ) -> None:
        """Initialize a component.

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
        self._velocity = None
        self._acceleration = None

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

    @property
    def velocity(self) -> tuple[float, float] | None:
        """Return linear velocity (vx, vy) in units/s.

        Returns None if velocity has not been computed.
        """
        return self._velocity

    @velocity.setter
    def velocity(self, value: tuple[float, float] | None) -> None:
        """Set the velocity of this component."""
        self._velocity = value

    @property
    def acceleration(self) -> tuple[float, float] | None:
        """Return linear acceleration (ax, ay) in units/s².

        Returns None if acceleration has not been computed.
        """
        return self._acceleration

    @acceleration.setter
    def acceleration(self, value: tuple[float, float] | None) -> None:
        """Set the acceleration of this component."""
        self._acceleration = value

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


class ConnectedComponent(Component):
    """Base class for components that connect to parent elements.

    Connected components have one or more anchor points that they
    reference for position computation.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def anchors(self) -> tuple[Component, ...]:
        """Return the parent components this connects to.

        Returns:
            Tuple of parent Component objects.
        """
        ...


class _AnchorProxy:
    """Proxy object for accessing a component's output position.

    Used by actuators (e.g., Crank.output) to provide a consistent
    interface for other components to connect to.
    """

    __slots__ = ("_parent",)

    def __init__(self, parent: Component) -> None:
        """Initialize proxy.

        Args:
            parent: The component this proxy represents.
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

    @property
    def velocity(self) -> tuple[float, float] | None:
        """Return the parent's velocity."""
        return self._parent.velocity

    @property
    def acceleration(self) -> tuple[float, float] | None:
        """Return the parent's acceleration."""
        return self._parent.acceleration


# Backwards compatibility aliases
Dyad = Component
ConnectedDyad = ConnectedComponent
