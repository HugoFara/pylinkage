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
    init_positions: Sequence[tuple[float | None, float | None]] | None

    def __init__(
        self,
        score: float | None = None,
        dimensions: Sequence[float] | None = None,
        init_position: Sequence[tuple[float | None, float | None]] | None = None,
    ) -> None:
        self.score = score
        self.dimensions = dimensions
        self.init_positions = init_position

    def __iter__(self) -> Iterator[Any]:
        yield self.score
        yield self.dimensions
        yield self.init_positions

    def __setitem__(self, key: int | slice, value: Any) -> None:
        """Allow index assignment."""
        # If the key is an integer, treat it as an index.
        if key == 0:
            self.score = value
        elif key == 1:
            self.dimensions = value
        elif key == 2:
            self.init_positions = value
        elif isinstance(key, slice):
            for i, val in zip([0, 1, 2][key], value):
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
            return self.init_positions
        if isinstance(key, slice):
            return list(self)[key]
        raise IndexError()

    def __repr__(self) -> str:
        return (
            f"Agent(score={self.score}, dimensions={self.dimensions}, "
            f"init_positions={self.init_positions})"
        )
