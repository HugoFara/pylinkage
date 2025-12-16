"""Arc crank actuator - motor-driven oscillating rotary input.

An arc crank is a driver link that oscillates between angle limits
at a constant angular velocity, reversing direction at boundaries.
"""

from __future__ import annotations

import math

from ..components import ConnectedComponent, Ground, _AnchorProxy


class ArcCrank(ConnectedComponent):
    """A motor-driven oscillating rotary input (arc crank).

    An arc crank oscillates around a ground anchor between two angle limits
    at constant angular velocity, producing an output point that traces an arc.
    Direction reverses ("bounces") when reaching angle limits, similar to
    LinearActuator behavior at stroke limits.

    Attributes:
        anchor: The ground point this arc crank rotates around.
        radius: Distance from anchor to output.
        angular_velocity: Rotation rate magnitude in radians per step.
        arc_start: Minimum angle limit in radians.
        arc_end: Maximum angle limit in radians.
        initial_angle: Starting angle in radians (must be between arc_start and arc_end).

    Example:
        >>> O1 = Ground(0.0, 0.0, name="O1")
        >>> arc_crank = ArcCrank(
        ...     anchor=O1,
        ...     radius=1.0,
        ...     angular_velocity=0.1,
        ...     arc_start=0.0,
        ...     arc_end=math.pi/2,
        ... )
        >>> arc_crank.position
        (1.0, 0.0)
    """

    __slots__ = (
        "anchor",
        "radius",
        "angular_velocity",
        "arc_start",
        "arc_end",
        "initial_angle",
        "_angle",
        "_direction",
        "_output",
    )

    anchor: Ground
    radius: float
    angular_velocity: float
    arc_start: float
    arc_end: float
    initial_angle: float
    _angle: float
    _direction: float
    _output: _AnchorProxy

    def __init__(
        self,
        anchor: Ground,
        radius: float,
        angular_velocity: float = 0.1,
        arc_start: float = 0.0,
        arc_end: float = math.pi,
        initial_angle: float | None = None,
        name: str | None = None,
    ) -> None:
        """Create an arc crank (oscillating rotary driver).

        Args:
            anchor: The ground point this arc crank rotates around.
            radius: Distance from anchor to output point.
            angular_velocity: Rotation rate magnitude in radians per step.
            arc_start: Minimum angle limit in radians.
            arc_end: Maximum angle limit in radians (must be > arc_start).
            initial_angle: Starting angle (defaults to arc_start).
            name: Human-readable identifier.

        Raises:
            ValueError: If anchor position is undefined, arc_end <= arc_start,
                or initial_angle is out of range.
        """
        if anchor.x is None or anchor.y is None:
            raise ValueError(f"Ground anchor {anchor.name} must have defined position")

        if arc_end <= arc_start:
            raise ValueError(
                f"arc_end must be greater than arc_start, "
                f"got arc_start={arc_start}, arc_end={arc_end}"
            )

        # Handle initial_angle default and validation
        if initial_angle is None:
            initial_angle = arc_start

        if not arc_start <= initial_angle <= arc_end:
            raise ValueError(
                f"initial_angle must be between arc_start ({arc_start}) "
                f"and arc_end ({arc_end}), got {initial_angle}"
            )

        # Compute initial output position
        out_x = anchor.x + radius * math.cos(initial_angle)
        out_y = anchor.y + radius * math.sin(initial_angle)
        super().__init__(out_x, out_y, name)

        self.anchor = anchor
        self.radius = radius
        self.angular_velocity = angular_velocity
        self.arc_start = arc_start
        self.arc_end = arc_end
        self.initial_angle = initial_angle
        self._angle = initial_angle
        self._direction = 1.0  # Start moving in positive direction
        self._output = _AnchorProxy(self)

    @property
    def output(self) -> _AnchorProxy:
        """Return the output joint (end of the arc crank).

        This proxy can be used as an anchor for other dyads.

        Returns:
            An anchor proxy representing the arc crank output.
        """
        return self._output

    @property
    def angle(self) -> float:
        """Return current angle."""
        return self._angle

    @property
    def anchors(self) -> tuple[Ground]:
        """Return the parent dyads (just the ground anchor)."""
        return (self.anchor,)

    def get_constraints(self) -> tuple[float, float, float]:
        """Return optimizable constraints.

        Returns:
            Tuple containing (radius, arc_start, arc_end).
        """
        return (self.radius, self.arc_start, self.arc_end)

    def set_constraints(
        self,
        radius: float | None = None,
        arc_start: float | None = None,
        arc_end: float | None = None,
        *args: float | None,
    ) -> None:
        """Set the constraints.

        Args:
            radius: New radius value.
            arc_start: New arc_start value.
            arc_end: New arc_end value.
            *args: Ignored (for interface compatibility).
        """
        if radius is not None:
            self.radius = radius

        if arc_start is not None:
            self.arc_start = arc_start
            # Clamp current angle if needed
            if self._angle < arc_start:
                self._angle = arc_start

        if arc_end is not None:
            self.arc_end = arc_end
            # Clamp current angle if needed
            if self._angle > arc_end:
                self._angle = arc_end

    def reload(self, dt: float = 1) -> None:
        """Advance the arc crank by one step.

        Rotates the arc crank position by angular_velocity * dt radians,
        reversing direction when hitting angle limits.

        Args:
            dt: Time step multiplier.

        Raises:
            ValueError: If anchor position is undefined.
        """
        from ..solver.joints import solve_arc_crank

        if self.anchor.x is None or self.anchor.y is None:
            raise ValueError(f"Anchor {self.anchor.name} has undefined position")

        self.x, self.y, self._angle, self._direction = solve_arc_crank(
            self._angle,
            self._direction,
            self.anchor.x,
            self.anchor.y,
            self.radius,
            self.angular_velocity,
            self.arc_start,
            self.arc_end,
            dt,
        )
