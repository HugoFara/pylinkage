"""Ensemble — topology-bound population of mechanisms.

An Ensemble holds one linkage topology and N parameter variants.
Because all members share the same structure (joint types, parent
wiring, solve order), simulation can be batched through the numba
solver for maximum performance.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from numpy.typing import NDArray

from ..bridge.solver_conversion import linkage_to_solver_data
from ..optimization.collections.agent import Agent
from ..optimization.collections.pareto import ParetoFront
from ..solver.types import SolverData
from ._batch_sim import simulate_batch
from ._member import Member

if TYPE_CHECKING:
    from ..linkage import Linkage


class Ensemble:
    """N parameter variants of one linkage topology.

    All members share the same SolverData structure (joint_types,
    parent_indices, constraint_offsets, solve_order). Only constraints
    and initial_positions vary per member.

    Supports vectorized batch simulation via numba.

    Integer indexing returns a :class:`Member`; slice or array indexing
    returns a new :class:`Ensemble` (numpy convention).

    Args:
        linkage: Template Linkage defining the topology. Used to compile
            the shared SolverData; not mutated.
        dimensions: Constraint vectors, shape ``(n_members, n_constraints)``.
        initial_positions: Joint positions, shape ``(n_members, n_joints, 2)``.
        scores: Named score arrays, each of shape ``(n_members,)``.
            Stored columnar for efficient ranking/filtering.
    """

    def __init__(
        self,
        linkage: Linkage,
        dimensions: NDArray[np.float64],
        initial_positions: NDArray[np.float64],
        scores: dict[str, NDArray[np.float64]] | None = None,
    ) -> None:
        self._linkage = linkage
        self._template: SolverData = linkage_to_solver_data(linkage)
        self._dimensions = np.ascontiguousarray(dimensions, dtype=np.float64)
        self._initial_positions = np.ascontiguousarray(
            initial_positions, dtype=np.float64,
        )
        self._scores: dict[str, NDArray[np.float64]] = (
            {k: np.asarray(v, dtype=np.float64) for k, v in scores.items()}
            if scores is not None
            else {}
        )
        # Lazy: (n_members, iterations, n_joints, 2) or None
        self._trajectories: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_members(self) -> int:
        """Number of parameter variants."""
        return int(self._dimensions.shape[0])

    @property
    def n_constraints(self) -> int:
        """Number of constraints per member."""
        return int(self._dimensions.shape[1])

    @property
    def n_joints(self) -> int:
        """Number of joints in the shared topology."""
        return self._template.n_joints

    @property
    def dimensions(self) -> NDArray[np.float64]:
        """Constraint vectors, shape (n_members, n_constraints)."""
        return self._dimensions

    @property
    def initial_positions(self) -> NDArray[np.float64]:
        """Initial positions, shape (n_members, n_joints, 2)."""
        return self._initial_positions

    @property
    def scores(self) -> dict[str, NDArray[np.float64]]:
        """Named score arrays, each of shape (n_members,)."""
        return self._scores

    @property
    def trajectories(self) -> NDArray[np.float64] | None:
        """Cached trajectories, shape (n_members, iterations, n_joints, 2)."""
        return self._trajectories

    @property
    def linkage(self) -> Linkage:
        """Template linkage defining the shared topology."""
        return self._linkage

    @property
    def topology_key(self) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], tuple[int, ...]]:
        """Hashable topology identity for grouping.

        Two linkages with the same topology_key are structurally identical
        and can be merged into one Ensemble.
        """
        t = self._template
        return (
            tuple(int(x) for x in t.joint_types),
            tuple(tuple(int(x) for x in row) for row in t.parent_indices),
            tuple(int(x) for x in t.solve_order),
        )

    # ------------------------------------------------------------------
    # Collection protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_members

    @overload
    def __getitem__(self, idx: int) -> Member: ...
    @overload
    def __getitem__(self, idx: slice | NDArray[np.intp] | NDArray[np.bool_]) -> Ensemble: ...

    def __getitem__(
        self, idx: int | slice | NDArray[np.intp] | NDArray[np.bool_],
    ) -> Member | Ensemble:
        if isinstance(idx, (int, np.integer)):
            return self._member_at(int(idx))
        # Slice or array index → new Ensemble
        return self._slice(idx)

    def __iter__(self) -> Iterator[Member]:
        for i in range(self.n_members):
            yield self._member_at(i)

    def __repr__(self) -> str:
        score_keys = ", ".join(self._scores.keys()) if self._scores else "none"
        return (
            f"Ensemble(n_members={self.n_members}, n_joints={self.n_joints}, "
            f"n_constraints={self.n_constraints}, scores=[{score_keys}])"
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        iterations: int | None = None,
        dt: float = 1.0,
        *,
        store: bool = True,
    ) -> NDArray[np.float64]:
        """Batch-simulate all members.

        Args:
            iterations: Steps per member. If None, uses the linkage's
                rotation period.
            dt: Time step for crank rotation.
            store: If True, cache trajectories so they are available on
                individual Members and via :attr:`trajectories`.

        Returns:
            Array of shape ``(n_members, iterations, n_joints, 2)``.
        """
        if iterations is None:
            iterations = self._linkage.get_rotation_period()

        result = simulate_batch(
            self._template,
            self._dimensions,
            self._initial_positions,
            iterations,
            dt,
        )
        if store:
            self._trajectories = result
        return result

    def simulate_member(
        self, idx: int, iterations: int | None = None, dt: float = 1.0,
    ) -> NDArray[np.float64]:
        """Simulate a single member (lazy per-member access).

        The result is cached in :attr:`_trajectories` if the full batch
        has already been computed; otherwise a one-off simulation is run
        and the result is returned (not cached in the batch array).
        """
        if self._trajectories is not None:
            return np.asarray(self._trajectories[idx])

        if iterations is None:
            iterations = self._linkage.get_rotation_period()

        # One-off simulation for a single member
        traj = simulate_batch(
            self._template,
            self._dimensions[idx : idx + 1],
            self._initial_positions[idx : idx + 1],
            iterations,
            dt,
        )
        return np.asarray(traj[0])

    # ------------------------------------------------------------------
    # Filtering / ranking
    # ------------------------------------------------------------------

    def filter(self, predicate: Callable[[Member], bool]) -> Ensemble:
        """Return a new Ensemble with members matching *predicate*."""
        mask = np.array(
            [predicate(self._member_at(i)) for i in range(self.n_members)],
            dtype=np.bool_,
        )
        return self._slice(mask)

    def filter_by_score(
        self,
        name: str = "score",
        *,
        min_val: float = -np.inf,
        max_val: float = np.inf,
    ) -> Ensemble:
        """Return a new Ensemble with members whose score is in range."""
        arr = self._resolve_score(name)
        mask = (arr >= min_val) & (arr <= max_val)
        return self._slice(mask)

    def rank(self, key: str = "score", *, ascending: bool = True) -> Ensemble:
        """Return a new Ensemble sorted by *key*."""
        arr = self._resolve_score(key)
        order = np.argsort(arr)
        if not ascending:
            order = order[::-1]
        return self._slice(order)

    def top(self, n: int, key: str = "score", *, ascending: bool = True) -> Ensemble:
        """Return the top *n* members by *key*."""
        return self.rank(key, ascending=ascending)[:n]

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_agents(cls, linkage: Linkage, agents: list[Agent]) -> Ensemble:
        """Create an Ensemble from a legacy optimization result.

        Args:
            linkage: The linkage that was optimized.
            agents: Results from PSO, grid search, etc.
        """
        template = linkage_to_solver_data(linkage)
        n_joints = template.n_joints

        dims = np.array([np.asarray(a.dimensions) for a in agents], dtype=np.float64)
        scores_arr = np.array([a.score for a in agents], dtype=np.float64)

        positions = np.zeros((len(agents), n_joints, 2), dtype=np.float64)
        for i, a in enumerate(agents):
            if len(a.initial_positions) > 0:
                for j, (x, y) in enumerate(a.initial_positions):
                    if j >= n_joints:
                        break
                    positions[i, j, 0] = x if x is not None else 0.0
                    positions[i, j, 1] = y if y is not None else 0.0

        return cls(
            linkage=linkage,
            dimensions=dims,
            initial_positions=positions,
            scores={"score": scores_arr},
        )

    @classmethod
    def from_pareto_front(
        cls, linkage: Linkage, front: ParetoFront,
    ) -> Ensemble:
        """Create an Ensemble from a multi-objective Pareto front.

        Args:
            linkage: The linkage that was optimized.
            front: Pareto front result.
        """
        template = linkage_to_solver_data(linkage)
        n_joints = template.n_joints

        dims = np.array(
            [np.asarray(s.dimensions) for s in front.solutions],
            dtype=np.float64,
        )
        positions = np.zeros((len(front), n_joints, 2), dtype=np.float64)
        for i, s in enumerate(front.solutions):
            if len(s.initial_positions) > 0:
                for j, (x, y) in enumerate(s.initial_positions):
                    if j >= n_joints:
                        break
                    positions[i, j, 0] = x if x is not None else 0.0
                    positions[i, j, 1] = y if y is not None else 0.0

        # Build columnar scores from objective names
        n_obj = front.n_objectives
        names = front.objective_names or tuple(f"obj_{k}" for k in range(n_obj))
        scores_dict: dict[str, NDArray[np.float64]] = {}
        for k, name in enumerate(names):
            scores_dict[name] = np.array(
                [s.scores[k] for s in front.solutions], dtype=np.float64,
            )

        return cls(
            linkage=linkage,
            dimensions=dims,
            initial_positions=positions,
            scores=scores_dict,
        )

    def to_agents(self) -> list[Agent]:
        """Convert all members back to legacy Agents."""
        return [self._member_at(i).to_agent() for i in range(self.n_members)]

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def show(
        self,
        idx: int = 0,
        iterations: int | None = None,
        **kwargs: Any,
    ) -> object:
        """Animate a member with :func:`~pylinkage.visualizer.show_linkage`.

        Simulates the member if no trajectory is cached.

        Args:
            idx: Member index to visualize.
            iterations: Simulation steps (if trajectory not yet computed).
            **kwargs: Forwarded to ``show_linkage()``.

        Returns:
            The matplotlib FuncAnimation object.
        """
        from ..visualizer.animated import show_linkage

        member = self._member_at(idx)
        if member.trajectory is None:
            self.simulate_member(idx, iterations=iterations)
            member = self._member_at(idx)

        return show_linkage(self._linkage, loci=member.to_loci(), **kwargs)

    def plot_plotly(
        self,
        idx: int = 0,
        iterations: int | None = None,
        **kwargs: Any,
    ) -> object:
        """Plot a member with :func:`~pylinkage.visualizer.plot_linkage_plotly`.

        Simulates the member if no trajectory is cached.

        Args:
            idx: Member index to visualize.
            iterations: Simulation steps (if trajectory not yet computed).
            **kwargs: Forwarded to ``plot_linkage_plotly()``.

        Returns:
            A plotly Figure object.
        """
        from ..visualizer.plotly_viz import plot_linkage_plotly

        member = self._member_at(idx)
        if member.trajectory is None:
            self.simulate_member(idx, iterations=iterations)
            member = self._member_at(idx)

        return plot_linkage_plotly(
            self._linkage, loci=member.to_loci(), **kwargs,
        )

    def save_svg(
        self,
        path: str,
        idx: int = 0,
        iterations: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Save a member as SVG via :func:`~pylinkage.visualizer.save_linkage_svg`.

        Simulates the member if no trajectory is cached.

        Args:
            path: Output file path.
            idx: Member index to visualize.
            iterations: Simulation steps (if trajectory not yet computed).
            **kwargs: Forwarded to ``save_linkage_svg()``.
        """
        from ..visualizer.drawsvg_viz import save_linkage_svg

        member = self._member_at(idx)
        if member.trajectory is None:
            self.simulate_member(idx, iterations=iterations)
            member = self._member_at(idx)

        save_linkage_svg(self._linkage, path, loci=member.to_loci(), **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_score(self, key: str) -> NDArray[np.float64]:
        """Look up a score array by name.

        If *key* is not found but there is exactly one score, use it.
        This lets ``top()`` and ``rank()`` work without specifying the
        key when the Ensemble has a single named score.
        """
        arr = self._scores.get(key)
        if arr is not None:
            return arr
        if len(self._scores) == 1:
            return next(iter(self._scores.values()))
        raise KeyError(
            f"No score named {key!r}; available: {list(self._scores)}"
        )

    def _member_at(self, idx: int) -> Member:
        """Build a Member for the given index."""
        scores = {k: float(v[idx]) for k, v in self._scores.items()}
        traj = self._trajectories[idx] if self._trajectories is not None else None
        return Member(
            dimensions=self._dimensions[idx],
            initial_positions=self._initial_positions[idx],
            scores=scores,
            trajectory=traj,
        )

    def _slice(
        self, idx: slice | NDArray[np.intp] | NDArray[np.bool_],
    ) -> Ensemble:
        """Create a sub-Ensemble from a slice or fancy index."""
        new_dims = self._dimensions[idx]
        new_pos = self._initial_positions[idx]
        new_scores = {k: v[idx] for k, v in self._scores.items()}

        ens = Ensemble.__new__(Ensemble)
        ens._linkage = self._linkage
        ens._template = self._template  # shared, read-only
        ens._dimensions = np.ascontiguousarray(new_dims)
        ens._initial_positions = np.ascontiguousarray(new_pos)
        ens._scores = new_scores
        ens._trajectories = (
            self._trajectories[idx] if self._trajectories is not None else None
        )
        return ens
