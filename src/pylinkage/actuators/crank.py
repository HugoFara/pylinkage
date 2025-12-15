"""Crank actuator - motor-driven rotary input.

A crank is a driver link that rotates around a ground anchor
at a constant angular velocity.
"""

from __future__ import annotations

import math

from ..components import ConnectedComponent, Ground, _AnchorProxy


class Crank(ConnectedComponent):
    """A motor-driven rotary input (crank).

    A crank rotates around a ground anchor at constant angular velocity,
    producing an output point that traces a circle. This is the primary
    input driver for most linkage mechanisms.

    Attributes:
        anchor: The ground point this crank rotates around.
        radius: Distance from anchor to output.
        angular_velocity: Rotation rate in radians per step.
        initial_angle: Starting angle in radians.

    Example:
        >>> O1 = Ground(0.0, 0.0, name="O1")
        >>> crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1)
        >>> crank.position
        (1.0, 0.0)
        >>> crank.output.position  # Same as crank.position
        (1.0, 0.0)
    """

    __slots__ = ("anchor", "radius", "angular_velocity", "initial_angle", "_output")

    anchor: Ground
    radius: float
    angular_velocity: float
    initial_angle: float
    _output: _AnchorProxy

    def __init__(
        self,
        anchor: Ground,
        radius: float,
        angular_velocity: float = 0.1,
        initial_angle: float = 0.0,
        name: str | None = None,
    ) -> None:
        """Create a crank (rotary driver).

        Args:
            anchor: The ground point this crank rotates around.
            radius: Distance from anchor to output point.
            angular_velocity: Rotation rate in radians per step.
            initial_angle: Starting angle in radians (from +x axis).
            name: Human-readable identifier.
        """
        if anchor.x is None or anchor.y is None:
            raise ValueError(f"Ground anchor {anchor.name} must have defined position")

        # Compute initial output position
        out_x = anchor.x + radius * math.cos(initial_angle)
        out_y = anchor.y + radius * math.sin(initial_angle)
        super().__init__(out_x, out_y, name)

        self.anchor = anchor
        self.radius = radius
        self.angular_velocity = angular_velocity
        self.initial_angle = initial_angle
        self._output = _AnchorProxy(self)

    @property
    def output(self) -> _AnchorProxy:
        """Return the output joint (end of the crank).

        This proxy can be used as an anchor for other dyads.

        Returns:
            An anchor proxy representing the crank output.
        """
        return self._output

    @property
    def anchors(self) -> tuple[Ground]:
        """Return the parent dyads (just the ground anchor)."""
        return (self.anchor,)

    def get_constraints(self) -> tuple[float]:
        """Return the radius (optimizable constraint).

        Returns:
            Tuple containing the crank radius.
        """
        return (self.radius,)

    def set_constraints(
        self,
        distance: float | None = None,
        *args: float | None,
    ) -> None:
        """Set the radius constraint.

        Args:
            distance: New radius value.
            *args: Ignored (for interface compatibility).
        """
        if distance is not None:
            self.radius = distance

    def reload(self, dt: float = 1) -> None:
        """Advance the crank by one step.

        Rotates the crank position by angular_velocity * dt radians.

        Args:
            dt: Time step multiplier.

        Raises:
            ValueError: If anchor position is undefined.
        """
        from ..solver.joints import solve_crank

        if self.anchor.x is None or self.anchor.y is None:
            raise ValueError(f"Anchor {self.anchor.name} has undefined position")

        if self.x is None or self.y is None:
            raise ValueError(f"Crank {self.name} has undefined position")

        self.x, self.y = solve_crank(
            self.x,
            self.y,
            self.anchor.x,
            self.anchor.y,
            self.radius,
            self.angular_velocity,
            dt,
        )
