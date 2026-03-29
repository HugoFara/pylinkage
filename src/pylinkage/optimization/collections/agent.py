"""Agent class for optimization results."""

from collections.abc import Sequence
from typing import Any, NamedTuple

_SENTINEL = object()


class _AgentBase(NamedTuple):
    """Internal base for Agent fields."""

    score: float
    dimensions: Any  # NDArray or sequence of floats
    initial_positions: Sequence[tuple[float | None, float | None]]


class Agent(_AgentBase):
    """A class that uniformizes a linkage optimization.

    It is roughly a namedtuple with preassigned fields.

    The ``initial_positions`` field can also be accessed as ``init_positions``
    for backwards compatibility.
    """

    def __new__(
        cls,
        score: float,
        dimensions: Any = None,
        initial_positions: Sequence[tuple[float | None, float | None]] = (),
        *,
        init_positions: Any = _SENTINEL,
    ) -> "Agent":
        # Accept the old keyword ``init_positions`` as an alias
        if init_positions is not _SENTINEL:
            initial_positions = init_positions
        return super().__new__(cls, score, dimensions, initial_positions)

    @property
    def init_positions(self) -> Sequence[tuple[float | None, float | None]]:
        """Backwards-compatible alias for ``initial_positions``."""
        return self.initial_positions
