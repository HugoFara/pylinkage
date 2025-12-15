"""Linkage container for dyads.

The Linkage class orchestrates a collection of dyads (Ground, Crank,
RRRDyad, etc.) to simulate a planar mechanism.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable
from math import gcd, tau
from typing import TYPE_CHECKING

from ..exceptions import UnderconstrainedError
from ._base import ConnectedDyad, Dyad

if TYPE_CHECKING:
    from .crank import Crank


class Linkage:
    """A planar linkage mechanism built from dyads.

    The Linkage class orchestrates a collection of dyads to simulate
    a planar mechanism. It handles solve order computation, stepping,
    and constraint management.

    Example:
        >>> from pylinkage.dyads import Ground, Crank, RRRDyad, Linkage
        >>> O1 = Ground(0.0, 0.0, name="O1")
        >>> O2 = Ground(2.0, 0.0, name="O2")
        >>> crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1)
        >>> rocker = RRRDyad(crank.output, O2, distance1=2.0, distance2=1.5)
        >>> linkage = Linkage([O1, O2, crank, rocker], name="Four-Bar")
        >>> for positions in linkage.step():
        ...     print(positions)
    """

    __slots__ = ("name", "dyads", "_cranks", "_solve_order")

    name: str
    dyads: tuple[Dyad, ...]
    _cranks: tuple[Crank, ...]
    _solve_order: tuple[Dyad, ...]

    def __init__(
        self,
        dyads: Iterable[Dyad],
        order: Iterable[Dyad] | None = None,
        name: str | None = None,
    ) -> None:
        """Create a linkage from dyads.

        Args:
            dyads: Collection of dyads forming the linkage.
            order: Manual solve order. If None, computed automatically.
            name: Human-readable identifier.
        """
        from .crank import Crank

        self.name = name if name is not None else str(id(self))
        self.dyads = tuple(dyads)
        self._cranks = tuple(d for d in self.dyads if isinstance(d, Crank))

        if order is not None:
            self._solve_order = tuple(order)

    def _find_solve_order(self) -> tuple[Dyad, ...]:
        """Compute automatic solve order.

        Ground points are solved first (trivial), then cranks,
        then dependent dyads whose anchors are already solved.

        Returns:
            Tuple of dyads in solvable order.

        Raises:
            UnderconstrainedError: If order cannot be determined.
        """
        from .crank import Crank
        from .ground import Ground

        # Start with ground points (always solvable)
        solved: list[Dyad] = [d for d in self.dyads if isinstance(d, Ground)]
        solved_set: set[Dyad] = set(solved)

        # Track progress
        progress = True
        while len(solved) < len(self.dyads) and progress:
            progress = False
            for d in self.dyads:
                if d in solved_set:
                    continue

                # Check if this dyad can be solved
                can_solve = False

                if isinstance(d, Crank):
                    # Crank can solve if its anchor is solved
                    can_solve = d.anchor in solved_set
                elif isinstance(d, ConnectedDyad):
                    # Connected dyads can solve if all anchors are solved
                    anchors = d.anchors
                    can_solve = all(
                        a in solved_set or self._is_anchor_solved(a, solved_set)
                        for a in anchors
                    )

                if can_solve:
                    solved.append(d)
                    solved_set.add(d)
                    progress = True

        if len(solved) < len(self.dyads):
            unsolved = [d for d in self.dyads if d not in solved_set]
            raise UnderconstrainedError(
                f"Cannot determine solve order. Unsolved dyads: "
                f"{', '.join(d.name for d in unsolved)}"
            )

        self._solve_order = tuple(solved)
        return self._solve_order

    def _is_anchor_solved(self, anchor: Dyad, solved_set: set[Dyad]) -> bool:
        """Check if an anchor (possibly a proxy) is solved.

        Args:
            anchor: The anchor to check.
            solved_set: Set of solved dyads.

        Returns:
            True if the anchor's parent dyad is solved.
        """
        from ._base import _AnchorProxy

        if isinstance(anchor, _AnchorProxy):
            return anchor._parent in solved_set
        return anchor in solved_set

    def rebuild(self, positions: list[tuple[float, float]] | None = None) -> None:
        """Rebuild the linkage, optionally setting initial positions.

        Args:
            positions: Initial positions for each dyad. If None, uses
                current positions.
        """
        if not hasattr(self, "_solve_order"):
            self._find_solve_order()

        if positions is not None:
            for dyad, pos in zip(self.dyads, positions, strict=True):
                dyad.set_coord(pos[0], pos[1])

    def step(
        self,
        iterations: int | None = None,
        dt: float = 1,
    ) -> Generator[tuple[tuple[float | None, float | None], ...], None, None]:
        """Simulate the linkage.

        Yields positions for all dyads at each step.

        Args:
            iterations: Number of steps. If None, uses get_rotation_period().
            dt: Time step multiplier for crank rotation.

        Yields:
            Tuple of (x, y) positions for each dyad.
        """
        from .crank import Crank

        if not hasattr(self, "_solve_order"):
            self._find_solve_order()

        if iterations is None:
            iterations = self.get_rotation_period()

        for _ in range(iterations):
            for dyad in self._solve_order:
                if isinstance(dyad, Crank):
                    dyad.reload(dt)
                else:
                    dyad.reload()
            yield tuple(d.position for d in self.dyads)

    def get_rotation_period(self) -> int:
        """Return number of steps for one full rotation.

        Computes the LCM of all crank periods.

        Returns:
            Number of iterations with dt=1.
        """
        periods = 1
        for crank in self._cranks:
            if crank.angular_velocity != 0:
                freq = round(tau / abs(crank.angular_velocity))
                periods = periods * freq // gcd(periods, freq)
        return periods

    def get_coords(self) -> list[tuple[float | None, float | None]]:
        """Return positions of all dyads.

        Returns:
            List of (x, y) positions.
        """
        return [d.position for d in self.dyads]

    def set_coords(self, coords: list[tuple[float, float]]) -> None:
        """Set positions for all dyads.

        Args:
            coords: List of (x, y) positions.
        """
        for dyad, coord in zip(self.dyads, coords, strict=True):
            dyad.set_coord(coord[0], coord[1])

    def get_num_constraints(self) -> list[float]:
        """Return all constraints as a flat list.

        Used for optimization.

        Returns:
            Flat list of all constraint values.
        """
        constraints: list[float] = []
        for dyad in self.dyads:
            for c in dyad.get_constraints():
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
        for dyad in self.dyads:
            n_constraints = len([c for c in dyad.get_constraints() if c is not None])
            if n_constraints > 0:
                dyad.set_constraints(*values[idx : idx + n_constraints])
                idx += n_constraints
