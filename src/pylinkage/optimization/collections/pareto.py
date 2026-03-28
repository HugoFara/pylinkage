"""Pareto front result containers for multi-objective optimization."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..._types import JointPositions


@dataclass
class ParetoSolution:
    """A single solution on the Pareto front.

    Attributes:
        scores: Objective values, one per objective (all minimized).
        dimensions: Constraint values that produced this solution.
        init_positions: Initial joint positions used during optimization.
    """

    scores: tuple[float, ...]
    dimensions: NDArray[np.floating[Any]]
    init_positions: JointPositions

    def dominates(self, other: ParetoSolution) -> bool:
        """Check if this solution dominates another.

        A solution dominates another if it is at least as good in all
        objectives and strictly better in at least one.

        Args:
            other: Another Pareto solution to compare against.

        Returns:
            True if this solution dominates the other.
        """
        dominated = False
        for s1, s2 in zip(self.scores, other.scores, strict=True):
            if s1 > s2:
                return False
            if s1 < s2:
                dominated = True
        return dominated


@dataclass
class ParetoFront:
    """Collection of non-dominated solutions from multi-objective optimization.

    Attributes:
        solutions: List of Pareto-optimal solutions.
        objective_names: Names for each objective (for plotting).
    """

    solutions: list[ParetoSolution]
    objective_names: tuple[str, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        """Return the number of solutions in the Pareto front."""
        return len(self.solutions)

    def __iter__(self) -> Iterator[ParetoSolution]:
        """Iterate over solutions."""
        return iter(self.solutions)

    def __getitem__(self, index: int) -> ParetoSolution:
        """Get a solution by index."""
        return self.solutions[index]

    @property
    def n_objectives(self) -> int:
        """Return the number of objectives."""
        if not self.solutions:
            return 0
        return len(self.solutions[0].scores)

    def scores_array(self) -> NDArray[np.floating[Any]]:
        """Return all scores as a 2D numpy array.

        Returns:
            Array of shape (n_solutions, n_objectives).
        """
        if not self.solutions:
            return np.array([])
        return np.array([sol.scores for sol in self.solutions])

    def hypervolume(self, reference_point: Sequence[float]) -> float:
        """Compute the hypervolume indicator.

        The hypervolume is the volume of the objective space dominated by
        the Pareto front and bounded by a reference point. Higher is better.

        Args:
            reference_point: Upper bound for each objective. Should be
                worse than any solution in the front.

        Returns:
            The hypervolume indicator value.

        Raises:
            ImportError: If pymoo is not installed.
        """
        try:
            from pymoo.indicators.hv import HV
        except ImportError as e:
            raise ImportError(
                "pymoo is required for hypervolume calculation. "
                "Install with: pip install pylinkage[moo]"
            ) from e

        if not self.solutions:
            return 0.0

        scores = self.scores_array()
        ref = np.array(reference_point)
        indicator = HV(ref_point=ref)
        return float(indicator(scores))

    def filter(self, max_solutions: int) -> ParetoFront:
        """Filter to a subset of well-distributed solutions.

        Uses crowding distance to select diverse solutions.

        Args:
            max_solutions: Maximum number of solutions to keep.

        Returns:
            New ParetoFront with at most max_solutions solutions.
        """
        if len(self.solutions) <= max_solutions:
            return ParetoFront(
                solutions=list(self.solutions),
                objective_names=self.objective_names,
            )

        scores = self.scores_array()
        n = len(scores)

        # Compute crowding distance for each solution
        crowding = np.zeros(n)
        for obj_idx in range(self.n_objectives):
            sorted_indices = np.argsort(scores[:, obj_idx])
            obj_range = scores[sorted_indices[-1], obj_idx] - scores[sorted_indices[0], obj_idx]
            if obj_range == 0:
                continue
            # Boundary points get infinite crowding distance
            crowding[sorted_indices[0]] = np.inf
            crowding[sorted_indices[-1]] = np.inf
            # Interior points
            for i in range(1, n - 1):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i - 1]
                next_idx = sorted_indices[i + 1]
                crowding[idx] += (scores[next_idx, obj_idx] - scores[prev_idx, obj_idx]) / obj_range

        # Select solutions with highest crowding distance
        selected_indices = np.argsort(crowding)[-max_solutions:]
        selected = [self.solutions[i] for i in sorted(selected_indices)]

        return ParetoFront(
            solutions=selected,
            objective_names=self.objective_names,
        )

    def best_compromise(self, weights: Sequence[float] | None = None) -> ParetoSolution:
        """Select the best compromise solution.

        Uses weighted sum of normalized objectives to find a balanced solution.

        Args:
            weights: Weight for each objective. If None, uses equal weights.

        Returns:
            The solution with the lowest weighted sum.

        Raises:
            ValueError: If the front is empty.
        """
        if not self.solutions:
            raise ValueError("Cannot find best compromise in empty Pareto front")

        scores = self.scores_array()
        n_obj = self.n_objectives

        if weights is None:
            weights_list: list[float] = [1.0 / n_obj] * n_obj
        else:
            weights_list = list(weights)
        weights_arr = np.array(weights_list)

        # Normalize scores to [0, 1] range
        min_scores = scores.min(axis=0)
        max_scores = scores.max(axis=0)
        ranges = max_scores - min_scores
        ranges[ranges == 0] = 1.0  # Avoid division by zero

        normalized = (scores - min_scores) / ranges

        # Compute weighted sum
        weighted_sums = (normalized * weights_arr).sum(axis=1)
        best_idx = int(np.argmin(weighted_sums))

        return self.solutions[best_idx]

    def plot(
        self,
        ax: Axes | None = None,
        objective_indices: tuple[int, int] | tuple[int, int, int] = (0, 1),
        **kwargs: Any,
    ) -> Figure:
        """Plot the Pareto front.

        For 2 objectives: Creates a 2D scatter plot.
        For 3 objectives: Creates a 3D scatter plot.

        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure.
            objective_indices: Which objectives to plot (indices).
            **kwargs: Additional arguments passed to scatter().

        Returns:
            The matplotlib Figure containing the plot.

        Raises:
            ValueError: If the front is empty or indices are invalid.
        """
        import matplotlib.pyplot as plt

        if not self.solutions:
            raise ValueError("Cannot plot empty Pareto front")

        scores = self.scores_array()
        n_dims = len(objective_indices)

        if n_dims not in (2, 3):
            raise ValueError(f"Can only plot 2 or 3 objectives, got {n_dims}")

        for idx in objective_indices:
            if idx >= self.n_objectives:
                raise ValueError(
                    f"Objective index {idx} out of range (only {self.n_objectives} objectives)"
                )

        # Default plot settings
        plot_kwargs = {"s": 50, "alpha": 0.7, "edgecolors": "black", "linewidths": 0.5}
        plot_kwargs.update(kwargs)

        if n_dims == 2:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6))
            else:
                fig = ax.figure  # type: ignore[assignment]

            ax.scatter(
                scores[:, objective_indices[0]],
                scores[:, objective_indices[1]],
                **plot_kwargs,  # type: ignore[arg-type]
            )

            # Labels
            if self.objective_names:
                ax.set_xlabel(self.objective_names[objective_indices[0]])
                ax.set_ylabel(self.objective_names[objective_indices[1]])
            else:
                ax.set_xlabel(f"Objective {objective_indices[0]}")
                ax.set_ylabel(f"Objective {objective_indices[1]}")

            ax.set_title("Pareto Front")
            ax.grid(True, alpha=0.3)

        else:  # 3D plot
            if ax is None:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig = ax.figure  # type: ignore[assignment]

            # For 3D we need the third index (we know n_dims==3 so it's a 3-tuple)
            idx0, idx1 = objective_indices[0], objective_indices[1]
            idx2 = objective_indices[2]  # type: ignore[misc]
            ax.scatter(
                scores[:, idx0],
                scores[:, idx1],
                scores[:, idx2],
                **plot_kwargs,  # type: ignore[arg-type]
            )

            # Labels
            if self.objective_names:
                ax.set_xlabel(self.objective_names[idx0])
                ax.set_ylabel(self.objective_names[idx1])
                ax.set_zlabel(self.objective_names[idx2])  # type: ignore[union-attr]
            else:
                ax.set_xlabel(f"Objective {idx0}")
                ax.set_ylabel(f"Objective {idx1}")
                ax.set_zlabel(f"Objective {idx2}")  # type: ignore[union-attr]

            ax.set_title("Pareto Front")

        return fig
