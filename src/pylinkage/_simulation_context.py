"""Container-agnostic simulation context manager.

A small helper used by both ``simulation.Linkage`` and
``mechanism.Mechanism`` to provide the legacy ``with linkage.simulation(...)``
ergonomics. Saves the joint coordinates on entry and restores them on
exit, and yields ``(step_index, positions)`` pairs while iterating.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any


class Simulation:
    """Context-managed wrapper around a linkage's ``step()`` generator.

    Works with any container that exposes ``get_coords()``,
    ``set_coords(positions)``, and ``step(iterations=..., dt=...)``.

    Example:
        >>> with linkage.simulation(iterations=100) as sim:
        ...     for step, coords in sim:
        ...         print(step, coords)
    """

    __slots__ = ("_linkage", "_iterations", "_dt", "_initial_coords")

    def __init__(
        self,
        linkage: Any,
        iterations: int | None = None,
        dt: float = 1.0,
    ) -> None:
        self._linkage = linkage
        self._iterations = iterations
        self._dt = dt
        self._initial_coords: list[tuple[float | None, float | None]] | None = None

    def __enter__(self) -> Simulation:
        self._initial_coords = self._linkage.get_coords()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._initial_coords is not None:
            self._linkage.set_coords(self._initial_coords)

    def __iter__(
        self,
    ) -> Generator[tuple[int, tuple[tuple[float | None, float | None], ...]], None, None]:
        yield from enumerate(self._linkage.step(iterations=self._iterations, dt=self._dt))

    @property
    def linkage(self) -> Any:
        """The linkage being simulated."""
        return self._linkage

    @property
    def iterations(self) -> int:
        """Number of iterations for this simulation."""
        if self._iterations is None:
            return int(self._linkage.get_rotation_period())
        return self._iterations
