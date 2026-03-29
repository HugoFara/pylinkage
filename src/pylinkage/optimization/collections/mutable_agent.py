"""MutableAgent class for optimization results."""

from collections.abc import Iterator, Sequence
from typing import Any


class MutableAgent:
    """A custom class that is mutable, subscriptable, and supports index assignment.

    You should only use it as a dictionary of 3 elements.
    No backward compatibility guaranty on other use cases.
    """

    score: float | None
    dimensions: Sequence[float] | None
    initial_positions: Sequence[tuple[float | None, float | None]] | None

    def __init__(
        self,
        score: float | None = None,
        dimensions: Sequence[float] | None = None,
        initial_positions: Sequence[tuple[float | None, float | None]] | None = None,
        # Backwards-compatible keyword
        init_position: Sequence[tuple[float | None, float | None]] | None = None,
    ) -> None:
        self.score = score
        self.dimensions = dimensions
        self.initial_positions = (
            initial_positions if initial_positions is not None else init_position
        )

    @property
    def init_positions(self) -> Sequence[tuple[float | None, float | None]] | None:
        """Backwards-compatible alias for ``initial_positions``."""
        return self.initial_positions

    @init_positions.setter
    def init_positions(self, value: Sequence[tuple[float | None, float | None]] | None) -> None:
        self.initial_positions = value

    def __iter__(self) -> Iterator[Any]:
        yield self.score
        yield self.dimensions
        yield self.initial_positions

    def __setitem__(self, key: int | slice, value: Any) -> None:
        """Allow index assignment."""
        # If the key is an integer, treat it as an index.
        if key == 0:
            self.score = value
        elif key == 1:
            self.dimensions = value
        elif key == 2:
            self.initial_positions = value
        elif isinstance(key, slice):
            for i, val in zip([0, 1, 2][key], value, strict=False):
                self[i] = val
        else:
            raise IndexError()

    def __getitem__(self, key: int | slice) -> Any:
        """Allow subscripting."""
        # If the key is an integer, treat it as an index.
        if key == 0:
            return self.score
        if key == 1:
            return self.dimensions
        if key == 2:
            return self.initial_positions
        if isinstance(key, slice):
            return list(self)[key]
        raise IndexError()

    def __repr__(self) -> str:
        return (
            f"Agent(score={self.score}, dimensions={self.dimensions}, "
            f"initial_positions={self.initial_positions})"
        )
