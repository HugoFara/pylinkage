"""Member dataclass — a single mechanism in a population."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..optimization.collections.agent import Agent
from ..optimization.collections.pareto import ParetoSolution


@dataclass(slots=True)
class Member:
    """One mechanism variant in a population.

    Universal record type for a single mechanism, replacing the various
    result types (Agent, ParetoSolution) with a unified representation.

    Attributes:
        dimensions: Flat constraint vector, shape (n_constraints,).
        initial_positions: Joint positions, shape (n_joints, 2).
        scores: Named scores (e.g. {"stride": 4.2, "smoothness": 0.8}).
        trajectory: Cached simulation result, shape (steps, n_joints, 2).
            None until simulate() is called.
        metadata: Arbitrary extra data (topology name, catalog entry, etc.).
    """

    dimensions: NDArray[np.float64]
    initial_positions: NDArray[np.float64]
    scores: dict[str, float] = field(default_factory=dict)
    trajectory: NDArray[np.float64] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Primary score (first entry, or NaN if empty)."""
        if not self.scores:
            return float("nan")
        return next(iter(self.scores.values()))

    @classmethod
    def from_agent(cls, agent: Agent, n_joints: int) -> Member:
        """Convert a legacy Agent to a Member.

        Args:
            agent: Optimization result agent.
            n_joints: Number of joints in the linkage (needed to reshape
                initial_positions into (n_joints, 2)).
        """
        dims = np.asarray(agent.dimensions, dtype=np.float64)

        # Convert initial_positions from sequence of tuples to (n_joints, 2)
        if len(agent.initial_positions) > 0:
            pos = np.array(
                [(x if x is not None else 0.0, y if y is not None else 0.0)
                 for x, y in agent.initial_positions],
                dtype=np.float64,
            )
        else:
            pos = np.zeros((n_joints, 2), dtype=np.float64)

        return cls(
            dimensions=dims,
            initial_positions=pos,
            scores={"score": agent.score},
        )

    @classmethod
    def from_pareto_solution(cls, sol: ParetoSolution, n_joints: int,
                             objective_names: tuple[str, ...] = ()) -> Member:
        """Convert a ParetoSolution to a Member.

        Args:
            sol: Multi-objective optimization result.
            n_joints: Number of joints in the linkage.
            objective_names: Names for each objective score.
        """
        dims = np.asarray(sol.dimensions, dtype=np.float64)

        if len(sol.initial_positions) > 0:
            pos = np.array(
                [(x if x is not None else 0.0, y if y is not None else 0.0)
                 for x, y in sol.initial_positions],
                dtype=np.float64,
            )
        else:
            pos = np.zeros((n_joints, 2), dtype=np.float64)

        if objective_names and len(objective_names) == len(sol.scores):
            scores = dict(zip(objective_names, sol.scores, strict=True))
        else:
            scores = {f"obj_{i}": s for i, s in enumerate(sol.scores)}

        return cls(dimensions=dims, initial_positions=pos, scores=scores)

    def to_loci(self) -> tuple[tuple[tuple[float, float], ...], ...]:
        """Convert trajectory to the loci format used by the visualizer.

        Returns:
            Nested tuples ``loci[frame][joint] = (x, y)`` compatible
            with :func:`~pylinkage.visualizer.show_linkage` and friends.

        Raises:
            ValueError: If no trajectory has been computed yet.
        """
        if self.trajectory is None:
            raise ValueError(
                "No trajectory available. Call Ensemble.simulate() first."
            )
        traj = self.trajectory
        return tuple(
            tuple((float(traj[f, j, 0]), float(traj[f, j, 1]))
                  for j in range(traj.shape[1]))
            for f in range(traj.shape[0])
        )

    def to_agent(self) -> Agent:
        """Convert back to a legacy Agent for backwards compatibility."""
        pos_tuples = [
            (float(row[0]), float(row[1]))
            for row in self.initial_positions
        ]
        return Agent(
            score=self.score,
            dimensions=self.dimensions,
            initial_positions=pos_tuples,
        )
