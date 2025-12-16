"""Linkage container for simulating planar mechanisms.

The Linkage class orchestrates a collection of components (Ground, actuators,
dyads) to simulate a planar mechanism.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable
from math import gcd, tau
from typing import TYPE_CHECKING

from ..components import Component, ConnectedComponent, _AnchorProxy
from ..exceptions import UnderconstrainedError

if TYPE_CHECKING:
    from ..actuators import ArcCrank, Crank, LinearActuator


class Linkage:
    """A planar linkage mechanism built from components.

    The Linkage class orchestrates a collection of components (Ground points,
    actuators, and dyads) to simulate a planar mechanism. It handles solve
    order computation, stepping, and constraint management.

    Example:
        >>> from pylinkage.components import Ground
        >>> from pylinkage.actuators import Crank
        >>> from pylinkage.dyads import RRRDyad
        >>> from pylinkage.simulation import Linkage
        >>>
        >>> O1 = Ground(0.0, 0.0, name="O1")
        >>> O2 = Ground(2.0, 0.0, name="O2")
        >>> crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1)
        >>> rocker = RRRDyad(crank.output, O2, distance1=2.0, distance2=1.5)
        >>> linkage = Linkage([O1, O2, crank, rocker], name="Four-Bar")
        >>> for positions in linkage.step():
        ...     print(positions)
    """

    __slots__ = (
        "name",
        "components",
        "_cranks",
        "_arc_cranks",
        "_linear_actuators",
        "_solve_order",
    )

    name: str
    components: tuple[Component, ...]
    _cranks: tuple[Crank, ...]
    _arc_cranks: tuple[ArcCrank, ...]
    _linear_actuators: tuple[LinearActuator, ...]
    _solve_order: tuple[Component, ...]

    def __init__(
        self,
        components: Iterable[Component],
        order: Iterable[Component] | None = None,
        name: str | None = None,
    ) -> None:
        """Create a linkage from components.

        Args:
            components: Collection of components forming the linkage.
            order: Manual solve order. If None, computed automatically.
            name: Human-readable identifier.
        """
        from ..actuators import ArcCrank, Crank, LinearActuator

        self.name = name if name is not None else str(id(self))
        self.components = tuple(components)
        self._cranks = tuple(d for d in self.components if isinstance(d, Crank))
        self._arc_cranks = tuple(
            d for d in self.components if isinstance(d, ArcCrank)
        )
        self._linear_actuators = tuple(
            d for d in self.components if isinstance(d, LinearActuator)
        )

        if order is not None:
            self._solve_order = tuple(order)

    @property
    def dyads(self) -> tuple[Component, ...]:
        """Return components (backwards compatibility alias)."""
        return self.components

    def _find_solve_order(self) -> tuple[Component, ...]:
        """Compute automatic solve order.

        Ground points are solved first (trivial), then actuators
        (cranks and linear actuators), then dependent dyads whose
        anchors are already solved.

        Returns:
            Tuple of components in solvable order.

        Raises:
            UnderconstrainedError: If order cannot be determined.
        """
        from ..actuators import ArcCrank, Crank, LinearActuator
        from ..components import Ground

        # Start with ground points (always solvable)
        solved: list[Component] = [
            d for d in self.components if isinstance(d, Ground)
        ]
        solved_set: set[Component] = set(solved)

        # Track progress
        progress = True
        while len(solved) < len(self.components) and progress:
            progress = False
            for d in self.components:
                if d in solved_set:
                    continue

                # Check if this component can be solved
                can_solve = False

                if isinstance(d, (Crank, ArcCrank, LinearActuator)):
                    # Actuators can solve if their anchor is solved
                    can_solve = d.anchor in solved_set
                elif isinstance(d, ConnectedComponent):
                    # Connected components can solve if all anchors are solved
                    anchors = d.anchors
                    can_solve = all(
                        a in solved_set or self._is_anchor_solved(a, solved_set)
                        for a in anchors
                    )

                if can_solve:
                    solved.append(d)
                    solved_set.add(d)
                    progress = True

        if len(solved) < len(self.components):
            unsolved = [d for d in self.components if d not in solved_set]
            raise UnderconstrainedError(
                f"Cannot determine solve order. Unsolved components: "
                f"{', '.join(d.name for d in unsolved)}"
            )

        self._solve_order = tuple(solved)
        return self._solve_order

    def _is_anchor_solved(
        self, anchor: Component, solved_set: set[Component]
    ) -> bool:
        """Check if an anchor (possibly a proxy) is solved.

        Args:
            anchor: The anchor to check.
            solved_set: Set of solved components.

        Returns:
            True if the anchor's parent component is solved.
        """
        if isinstance(anchor, _AnchorProxy):
            return anchor._parent in solved_set
        return anchor in solved_set

    def rebuild(self, positions: list[tuple[float, float]] | None = None) -> None:
        """Rebuild the linkage, optionally setting initial positions.

        Args:
            positions: Initial positions for each component. If None, uses
                current positions.
        """
        if not hasattr(self, "_solve_order"):
            self._find_solve_order()

        if positions is not None:
            for component, pos in zip(self.components, positions, strict=True):
                component.set_coord(pos[0], pos[1])

    def step(
        self,
        iterations: int | None = None,
        dt: float = 1,
    ) -> Generator[tuple[tuple[float | None, float | None], ...], None, None]:
        """Simulate the linkage.

        Yields positions for all components at each step.

        Args:
            iterations: Number of steps. If None, uses get_rotation_period().
            dt: Time step multiplier for actuators (cranks and linear actuators).

        Yields:
            Tuple of (x, y) positions for each component.
        """
        from ..actuators import ArcCrank, Crank, LinearActuator

        if not hasattr(self, "_solve_order"):
            self._find_solve_order()

        if iterations is None:
            iterations = self.get_rotation_period()

        for _ in range(iterations):
            for component in self._solve_order:
                if isinstance(component, (Crank, ArcCrank, LinearActuator)):
                    component.reload(dt)
                else:
                    component.reload()
            yield tuple(d.position for d in self.components)

    def get_rotation_period(self) -> int:
        """Return number of steps for one full cycle.

        Computes the LCM of all actuator periods (cranks, arc cranks, and
        linear actuators).
        For cranks, period is 2*pi / angular_velocity.
        For arc cranks, period is 2 * (arc_end - arc_start) / angular_velocity.
        For linear actuators, period is 2 * stroke / velocity.

        Returns:
            Number of iterations with dt=1.
        """
        periods = 1

        # Consider crank periods
        for crank in self._cranks:
            if crank.angular_velocity != 0:
                freq = round(tau / abs(crank.angular_velocity))
                periods = periods * freq // gcd(periods, freq)

        # Consider arc crank periods
        for arc_crank in self._arc_cranks:
            if arc_crank.angular_velocity != 0:
                # Full cycle is 2 * arc_range / angular_velocity (one round trip)
                arc_range = arc_crank.arc_end - arc_crank.arc_start
                freq = round(2 * arc_range / abs(arc_crank.angular_velocity))
                periods = periods * freq // gcd(periods, freq)

        # Consider linear actuator periods
        for actuator in self._linear_actuators:
            if actuator.velocity != 0:
                # Full cycle is 2 * stroke / velocity (one round trip)
                freq = round(2 * actuator.stroke / abs(actuator.velocity))
                periods = periods * freq // gcd(periods, freq)

        return periods

    def get_coords(self) -> list[tuple[float | None, float | None]]:
        """Return positions of all components.

        Returns:
            List of (x, y) positions.
        """
        return [d.position for d in self.components]

    def set_coords(self, coords: list[tuple[float, float]]) -> None:
        """Set positions for all components.

        Args:
            coords: List of (x, y) positions.
        """
        for component, coord in zip(self.components, coords, strict=True):
            component.set_coord(coord[0], coord[1])

    def get_num_constraints(self) -> list[float]:
        """Return all constraints as a flat list.

        Used for optimization.

        Returns:
            Flat list of all constraint values.
        """
        constraints: list[float] = []
        for component in self.components:
            for c in component.get_constraints():
                if c is not None:
                    constraints.append(c)
        return constraints

    def set_num_constraints(self, values: list[float]) -> None:
        """Set constraints from a flat list.

        Used to apply optimization results.

        Args:
            values: Flat list of constraint values.
        """
        idx = 0
        for component in self.components:
            n_constraints = len(
                [c for c in component.get_constraints() if c is not None]
            )
            if n_constraints > 0:
                component.set_constraints(*values[idx : idx + n_constraints])
                idx += n_constraints
