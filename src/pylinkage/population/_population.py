"""Population — heterogeneous collection of mechanisms.

A Population holds mechanisms with potentially different topologies,
organized internally as a dict of topology_key → Ensemble. Each
Ensemble can be batch-simulated independently.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from ._ensemble import Ensemble
from ._member import Member

if TYPE_CHECKING:
    from ..linkage import Linkage
    from ..synthesis.topology_types import TopologySolution


class Population:
    """Collection of mechanisms with potentially different topologies.

    Internally organized as ``topology_key → Ensemble``. Supports the
    same iteration/filtering/ranking interface as :class:`Ensemble` but
    cannot batch-simulate across topologies.

    Args:
        ensembles: Pre-built ensembles keyed by a user-chosen label.
            If None, starts empty.
    """

    def __init__(
        self,
        ensembles: dict[str, Ensemble] | None = None,
    ) -> None:
        self._ensembles: dict[str, Ensemble] = dict(ensembles) if ensembles else {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def topologies(self) -> dict[str, Ensemble]:
        """Topology label → Ensemble mapping (read-only view)."""
        return dict(self._ensembles)

    @property
    def n_topologies(self) -> int:
        """Number of distinct topologies."""
        return len(self._ensembles)

    # ------------------------------------------------------------------
    # Collection protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of members across all ensembles."""
        return sum(len(e) for e in self._ensembles.values())

    def __iter__(self) -> Iterator[Member]:
        """Iterate over all members across all ensembles."""
        for ens in self._ensembles.values():
            yield from ens

    def __repr__(self) -> str:
        parts = [f"{k}: {len(e)} members" for k, e in self._ensembles.items()]
        return f"Population({', '.join(parts)})"

    # ------------------------------------------------------------------
    # Ensemble access
    # ------------------------------------------------------------------

    def ensemble(self, label: str) -> Ensemble:
        """Get an ensemble by its label.

        Raises:
            KeyError: If no ensemble with that label exists.
        """
        return self._ensembles[label]

    def add_ensemble(self, label: str, ensemble: Ensemble) -> None:
        """Add or replace an ensemble.

        Args:
            label: Human-readable label for this topology group.
            ensemble: The Ensemble to add.
        """
        self._ensembles[label] = ensemble

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_all(
        self,
        iterations: int | None = None,
        dt: float = 1.0,
    ) -> None:
        """Simulate all ensembles (vectorized within each topology).

        Args:
            iterations: Steps per member. If None, each ensemble uses
                its linkage's rotation period.
            dt: Time step for crank rotation.
        """
        for ens in self._ensembles.values():
            ens.simulate(iterations=iterations, dt=dt, store=True)

    # ------------------------------------------------------------------
    # Filtering / ranking
    # ------------------------------------------------------------------

    def flatten(self) -> list[Member]:
        """All members as a flat list, regardless of topology."""
        return list(self)

    def rank(self, key: str = "score", *, ascending: bool = True) -> list[Member]:
        """All members ranked by *key* across topologies.

        Returns a flat list because Members from different topologies
        cannot form a single Ensemble.  If *key* is not found on a
        member but the member has exactly one score, that score is used.
        """
        def _sort_key(m: Member) -> float:
            if key in m.scores:
                return m.scores[key]
            if len(m.scores) == 1:
                return next(iter(m.scores.values()))
            return float("inf")

        members = self.flatten()
        members.sort(key=_sort_key, reverse=not ascending)
        return members

    def top(self, n: int, key: str = "score", *, ascending: bool = True) -> list[Member]:
        """Top *n* members across all topologies."""
        return self.rank(key, ascending=ascending)[:n]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_ensembles(cls, ensembles: list[Ensemble]) -> Population:
        """Build a Population from a list of Ensembles.

        Labels are auto-generated as ``"topology_0"``, ``"topology_1"``, etc.
        """
        labeled = {f"topology_{i}": e for i, e in enumerate(ensembles)}
        return cls(ensembles=labeled)

    @classmethod
    def from_members(
        cls,
        members: list[tuple[Linkage, Member]],
    ) -> Population:
        """Build a Population from (linkage, member) pairs.

        Members are grouped by topology key into Ensembles automatically.

        Args:
            members: Pairs of (template linkage, member data).
        """
        from ..bridge.solver_conversion import linkage_to_solver_data

        # Group by topology key
        groups: dict[tuple[Any, ...], list[tuple[Linkage, Member]]] = defaultdict(list)
        key_cache: dict[int, tuple[Any, ...]] = {}

        for linkage, member in members:
            lid = id(linkage)
            if lid not in key_cache:
                sd = linkage_to_solver_data(linkage)
                key_cache[lid] = (
                    tuple(int(x) for x in sd.joint_types),
                    tuple(tuple(int(x) for x in row) for row in sd.parent_indices),
                    tuple(int(x) for x in sd.solve_order),
                )
            groups[key_cache[lid]].append((linkage, member))

        ensembles: dict[str, Ensemble] = {}
        for i, (_, group) in enumerate(groups.items()):
            linkage_template = group[0][0]
            dims = np.stack([m.dimensions for _, m in group])
            positions = np.stack([m.initial_positions for _, m in group])

            # Merge score names from all members
            all_score_names: set[str] = set()
            for _, m in group:
                all_score_names.update(m.scores.keys())

            scores_dict: dict[str, np.ndarray] = {}
            for name in sorted(all_score_names):
                scores_dict[name] = np.array(
                    [m.scores.get(name, float("nan")) for _, m in group],
                    dtype=np.float64,
                )

            ensembles[f"topology_{i}"] = Ensemble(
                linkage=linkage_template,
                dimensions=dims,
                initial_positions=positions,
                scores=scores_dict,
            )

        return cls(ensembles=ensembles)

    @classmethod
    def from_topology_solutions(
        cls,
        solutions: list[TopologySolution],
    ) -> Population:
        """Build a Population from multi-topology synthesis results.

        Groups solutions by ``topology_id`` into Ensembles. Each
        solution's :class:`~pylinkage.synthesis.topology_types.QualityMetrics`
        are carried as scores on the Ensemble members.

        Args:
            solutions: Results from ``multi_topology_synthesize()`` or
                ``generalized_synthesis()``.
        """
        # Group by topology_id
        groups: dict[str, list[TopologySolution]] = defaultdict(list)
        for sol in solutions:
            groups[sol.solution.topology_id].append(sol)

        ensembles: dict[str, Ensemble] = {}
        for topo_id, group in groups.items():
            template = group[0].linkage
            n = len(group)
            n_constraints = len(template.get_num_constraints())
            from .._compat import get_parts

            n_joints = len(get_parts(template))

            dims = np.empty((n, n_constraints), dtype=np.float64)
            positions = np.empty((n, n_joints, 2), dtype=np.float64)

            for i, tsol in enumerate(group):
                lk = tsol.linkage
                constraints = lk.get_num_constraints()
                dims[i] = [
                    c if c is not None else 0.0 for c in constraints
                ]
                for j, (x, y) in enumerate(lk.get_coords()):
                    positions[i, j, 0] = x if x is not None else 0.0
                    positions[i, j, 1] = y if y is not None else 0.0

            # Carry QualityMetrics as score columns
            scores: dict[str, np.ndarray] = {
                "path_accuracy": np.array(
                    [s.metrics.path_accuracy for s in group], dtype=np.float64,
                ),
                "min_transmission_angle": np.array(
                    [s.metrics.min_transmission_angle for s in group],
                    dtype=np.float64,
                ),
                "link_ratio": np.array(
                    [s.metrics.link_ratio for s in group], dtype=np.float64,
                ),
                "compactness": np.array(
                    [s.metrics.compactness for s in group], dtype=np.float64,
                ),
                "overall_score": np.array(
                    [s.metrics.overall_score for s in group], dtype=np.float64,
                ),
            }

            ensembles[topo_id] = Ensemble(
                linkage=template,
                dimensions=dims,
                initial_positions=positions,
                scores=scores,
            )

        return cls(ensembles=ensembles)
