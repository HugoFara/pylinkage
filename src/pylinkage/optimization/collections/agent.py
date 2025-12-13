"""Agent class for optimization results."""


from collections.abc import Sequence
from typing import Any, NamedTuple


class Agent(NamedTuple):
    """A class that uniformizes a linkage optimization.

    It is roughly a namedtuple with preassigned fields.
    """

    score: float
    dimensions: Any  # NDArray or sequence of floats
    init_positions: Sequence[tuple[float | None, float | None]]
