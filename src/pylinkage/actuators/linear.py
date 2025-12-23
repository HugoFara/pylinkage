"""Linear actuator - motor-driven linear input.

A linear actuator is a driver that moves along a straight line
at constant velocity, oscillating between 0 and stroke limits.
"""

from __future__ import annotations

import math

from ..components import ConnectedComponent, Ground, _AnchorProxy


class LinearActuator(ConnectedComponent):
    """A motor-driven linear input (linear actuator).

    A linear actuator moves along a line from its anchor at constant
    speed, producing an output point that oscillates between 0
    and the stroke limit. This provides linear reciprocating motion.

    Attributes:
        anchor: The ground point this actuator extends from.
        angle: Direction angle in radians (from +x axis).
        stroke: Maximum extension distance.
        speed: Linear speed magnitude (units per step).
        initial_extension: Starting extension from anchor.

    Example:
        >>> O1 = Ground(0.0, 0.0, name="O1")
        >>> actuator = LinearActuator(anchor=O1, angle=0.0, stroke=2.0, speed=0.1)
        >>> actuator.position
        (0.0, 0.0)
        >>> actuator.reload()
        >>> actuator.position  # Moved 0.1 units along x-axis
        (0.1, 0.0)
    """

    __slots__ = (
        "anchor",
        "angle",
        "stroke",
        "speed",
        "initial_extension",
        "_extension",
        "_direction",
        "_output",
    )

    anchor: Ground
    angle: float
    stroke: float
    speed: float
    initial_extension: float
    _extension: float
    _direction: float
    _output: _AnchorProxy

    def __init__(
        self,
        anchor: Ground,
        angle: float,
        stroke: float,
        speed: float = 0.1,
        initial_extension: float = 0.0,
        name: str | None = None,
    ) -> None:
        """Create a linear actuator (linear driver).

        Args:
            anchor: The ground point this actuator extends from.
            angle: Direction angle in radians (from +x axis).
            stroke: Maximum extension distance (must be positive).
            speed: Linear speed magnitude (units per step).
            initial_extension: Starting extension from anchor (0 to stroke).
            name: Human-readable identifier.

        Raises:
            ValueError: If anchor position is undefined, stroke <= 0,
                or initial_extension is out of range.
        """
        if anchor.x is None or anchor.y is None:
            raise ValueError(f"Ground anchor {anchor.name} must have defined position")

        if stroke <= 0:
            raise ValueError(f"Stroke must be positive, got {stroke}")

        if not 0 <= initial_extension <= stroke:
            raise ValueError(
                f"Initial extension must be between 0 and stroke ({stroke}), "
                f"got {initial_extension}"
            )

        # Compute initial output position
        out_x = anchor.x + initial_extension * math.cos(angle)
        out_y = anchor.y + initial_extension * math.sin(angle)
        super().__init__(out_x, out_y, name)

        self.anchor = anchor
        self.angle = angle
        self.stroke = stroke
        self.speed = speed
        self.initial_extension = initial_extension
        self._extension = initial_extension
        self._direction = 1.0  # Start moving in positive direction
        self._output = _AnchorProxy(self)

    @property
    def output(self) -> _AnchorProxy:
        """Return the output joint (end of the actuator).

        This proxy can be used as an anchor for other dyads.

        Returns:
            An anchor proxy representing the actuator output.
        """
        return self._output

    @property
    def extension(self) -> float:
        """Return current extension from anchor."""
        return self._extension

    @property
    def anchors(self) -> tuple[Ground]:
        """Return the parent dyads (just the ground anchor)."""
        return (self.anchor,)

    def get_constraints(self) -> tuple[float, float]:
        """Return the stroke and speed (optimizable constraints).

        Returns:
            Tuple containing (stroke, speed).
        """
        return (self.stroke, self.speed)

    def set_constraints(
        self,
        stroke: float | None = None,
        speed: float | None = None,
        *args: float | None,
    ) -> None:
        """Set the constraints.

        Args:
            stroke: New stroke value (must be positive).
            speed: New speed value.
            *args: Ignored (for interface compatibility).
        """
        if stroke is not None:
            if stroke <= 0:
                raise ValueError(f"Stroke must be positive, got {stroke}")
            self.stroke = stroke
            # Clamp current extension if needed
            if self._extension > stroke:
                self._extension = stroke

        if speed is not None:
            self.speed = speed

    def reload(self, dt: float = 1) -> None:
        """Advance the actuator by one step.

        Moves the actuator position by speed * dt, reversing
        direction when hitting stroke limits.

        Args:
            dt: Time step multiplier.

        Raises:
            ValueError: If anchor position is undefined.
        """
        from ..solver.joints import solve_linear_actuator

        if self.anchor.x is None or self.anchor.y is None:
            raise ValueError(f"Anchor {self.anchor.name} has undefined position")

        self.x, self.y, self._extension, self._direction = solve_linear_actuator(
            self._extension,
            self._direction,
            self.anchor.x,
            self.anchor.y,
            self.angle,
            self.stroke,
            self.speed,
            dt,
        )
