"""
Multi-objective optimization using NSGA-II/NSGA-III.

This module provides Pareto-optimal solutions for linkage optimization
with multiple competing objectives.

Created on 2025.

@author: HugoFara
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..exceptions import OptimizationError
from ..population import Ensemble
from .utils import generate_bounds

if TYPE_CHECKING:
    from typing import Any as Linkage  # accepts legacy/sim Linkage and Mechanism

    from .._types import JointPositions


def _check_pymoo_available() -> None:
    """Check if pymoo is installed and raise informative error if not."""
    try:
        import pymoo  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pymoo is required for multi-objective optimization. "
            "Install with: pip install pylinkage[moo]"
        ) from e


def _evaluate_candidate(
    objectives: Sequence[Callable[..., float]],
    linkage: Linkage,
    constraints: list[float],
    joint_pos: JointPositions,
) -> list[float]:
    """Top-level per-candidate evaluator.

    Kept as a module-level function so it is picklable for
    ``ProcessPoolExecutor`` workers.
    """
    return [obj(linkage, constraints, joint_pos) for obj in objectives]


def _evaluate_candidate_factory(
    objectives: Sequence[Callable[..., float]],
    linkage_factory: Callable[[], Linkage],
    constraints: list[float],
    joint_pos: JointPositions,
) -> list[float]:
    """Top-level per-candidate evaluator that builds a fresh linkage.

    Use this path when the linkage (or its cached numba ``SolverData``)
    is not picklable: each worker calls ``linkage_factory()`` once per
    candidate.
    """
    linkage = linkage_factory()
    return [obj(linkage, constraints, joint_pos) for obj in objectives]


class LinkageProblem:
    """Pymoo Problem wrapper for linkage optimization.

    This class adapts the linkage constraint optimization problem to
    pymoo's Problem interface.
    """

    def __init__(
        self,
        linkage: Linkage,
        objectives: Sequence[Callable[..., float]],
        bounds: tuple[Sequence[float], Sequence[float]],
        joint_pos: JointPositions,
        n_workers: int = 1,
        linkage_factory: Callable[[], Linkage] | None = None,
    ) -> None:
        """Initialize the linkage optimization problem.

        Args:
            linkage: The linkage to optimize.
            objectives: List of objective functions. Each takes
                (linkage, constraints, joint_positions) and returns a float.
            bounds: Tuple of (lower_bounds, upper_bounds) for constraints.
            joint_pos: Initial joint positions.
            n_workers: Number of worker processes. ``1`` (default) runs
                serially; anything higher spawns a ``ProcessPoolExecutor``.
            linkage_factory: Optional zero-arg callable building a fresh
                linkage. Used instead of shipping ``linkage`` to workers
                when the linkage is not picklable.
        """
        from pymoo.core.problem import Problem

        self.linkage = linkage
        self.objectives = objectives
        self.joint_pos = joint_pos
        self._n_workers = max(1, n_workers)
        self._linkage_factory = linkage_factory
        # Lazy-created on first parallel batch and reused across
        # generations. Pool startup forks N worker processes — doing
        # that per generation wastes ~50–500 ms on top of every batch.
        # ``close()`` shuts it down.
        self._pool: Any = None

        n_var = len(bounds[0])
        n_obj = len(objectives)

        outer = self

        class _Problem(Problem):  # type: ignore[misc]
            def __init__(inner_self) -> None:
                super().__init__(
                    n_var=n_var,
                    n_obj=n_obj,
                    n_ieq_constr=0,
                    xl=np.array(bounds[0]),
                    xu=np.array(bounds[1]),
                )

            def _evaluate(
                inner_self,
                X: NDArray[np.floating[Any]],
                out: dict[str, Any],
                *args: Any,
                **kwargs: Any,
            ) -> None:
                out["F"] = outer._evaluate_batch(X)

        self._problem = _Problem()

    def _evaluate_batch(self, X: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        n_pop = len(X)
        n_obj = len(self.objectives)
        F = np.full((n_pop, n_obj), np.inf, dtype=np.float64)

        if self._n_workers <= 1:
            for i, x in enumerate(X):
                F[i] = _evaluate_candidate(
                    self.objectives, self.linkage, x.tolist(), self.joint_pos
                )
            return F

        import contextlib
        from concurrent.futures import as_completed

        pool = self._get_pool()
        candidates = [X[i].tolist() for i in range(n_pop)]
        use_factory = self._linkage_factory is not None
        futures = {}
        for idx, candidate in enumerate(candidates):
            if use_factory:
                assert self._linkage_factory is not None  # for mypy
                fut = pool.submit(
                    _evaluate_candidate_factory,
                    self.objectives,
                    self._linkage_factory,
                    candidate,
                    self.joint_pos,
                )
            else:
                fut = pool.submit(
                    _evaluate_candidate,
                    self.objectives,
                    self.linkage,
                    candidate,
                    self.joint_pos,
                )
            futures[fut] = idx

        for future in as_completed(futures):
            idx = futures[future]
            # Leave +inf on failure so the candidate is dominated and the
            # caller can filter non-finite scores afterwards.
            with contextlib.suppress(Exception):
                F[idx] = future.result()
        return F

    def _get_pool(self) -> Any:
        """Return the shared process pool, creating it on first use.

        Uses the ``spawn`` start method explicitly. The default ``fork``
        emits a ``DeprecationWarning`` under Python 3.12+ when the
        parent has live threads (pytest, matplotlib, Jupyter all spawn
        helper threads) and risks deadlocks in the child. Spawn pays
        a one-time module-import cost per worker — already amortized
        because the pool is reused across generations.
        """
        if self._pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            self._pool = ProcessPoolExecutor(
                max_workers=self._n_workers,
                mp_context=mp.get_context("spawn"),
            )
        return self._pool

    def close(self) -> None:
        """Shut down the shared process pool, if one was created.

        Safe to call repeatedly. :func:`multi_objective_optimization`
        invokes this in a ``finally`` block so workers don't outlive
        the optimization call.
        """
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __del__(self) -> None:  # pragma: no cover — best-effort cleanup
        # Defensive cleanup if the caller forgot ``close()``. Pool
        # workers otherwise linger until the parent process exits.
        import contextlib

        with contextlib.suppress(Exception):
            self.close()

    @property
    def problem(self) -> Any:
        """Return the pymoo Problem instance."""
        return self._problem


def multi_objective_optimization(
    objectives: Sequence[Callable[..., float]],
    linkage: Linkage,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    objective_names: Sequence[str] | None = None,
    algorithm: Literal["nsga2", "nsga3"] = "nsga2",
    n_generations: int = 100,
    pop_size: int = 100,
    seed: int | None = None,
    verbose: bool = True,
    n_workers: int = 1,
    linkage_factory: Callable[[], Linkage] | None = None,
    **kwargs: Any,
) -> Ensemble:
    """Multi-objective optimization using NSGA-II or NSGA-III.

    Finds Pareto-optimal solutions that trade off between multiple objectives.
    All objectives are MINIMIZED. For maximization, negate the objective
    (e.g., use ``lambda loci, **kw: -original_func(loci, **kw)``).

    Args:
        objectives: List of objective functions. Each should be decorated
            with ``@kinematic_minimization`` or take the signature
            ``(linkage, constraints, joint_positions) -> float``.
        linkage: The linkage to optimize.
        bounds: Tuple of (lower_bounds, upper_bounds) for constraints.
            If None, bounds are auto-generated from current constraints.
        objective_names: Names for each objective (used in plotting).
            If None, names are auto-generated as "Objective 0", etc.
        algorithm: Optimization algorithm. Options:
            - "nsga2": NSGA-II (default), good for 2-3 objectives
            - "nsga3": NSGA-III, better for many objectives (>3)
        n_generations: Number of generations to run. Default is 100.
        pop_size: Population size. Default is 100.
        seed: Random seed for reproducibility.
        verbose: Print progress if True. Default is True.
        n_workers: Parallel evaluation worker count. ``1`` (default)
            evaluates candidates serially in the calling process; anything
            higher spawns a ``concurrent.futures.ProcessPoolExecutor`` and
            evaluates one candidate per worker. The ``linkage`` and the
            objective functions must be picklable when ``n_workers > 1``.
        linkage_factory: Optional zero-arg callable that returns a fresh
            linkage. When supplied *and* ``n_workers > 1``, each worker
            builds its own linkage via this callable instead of receiving
            a pickled copy. Use this escape hatch when the linkage itself
            is not picklable (e.g. holds cached numba ``SolverData``).
        **kwargs: Additional arguments passed to the algorithm.

    Returns:
        Ensemble containing all non-dominated solutions, with one
        score column per objective.

    Raises:
        ImportError: If pymoo is not installed.
        OptimizationError: If parameters are invalid.

    Example::

        from pylinkage.optimization import (
            multi_objective_optimization,
            kinematic_minimization,
        )

        @kinematic_minimization
        def path_error(loci, **kwargs):
            # Compute error from target path
            return compute_path_error(loci)

        @kinematic_minimization
        def transmission_penalty(loci, linkage, **kwargs):
            # Penalize poor transmission angles
            analysis = linkage.analyze_transmission()
            return abs(90 - analysis.mean_angle)

        pareto = multi_objective_optimization(
            objectives=[path_error, transmission_penalty],
            linkage=my_linkage,
            objective_names=["Path Error", "Transmission Penalty"],
            n_generations=100,
        )

        # Visualize trade-offs
        pareto.plot()

        # Get best compromise solution
        best = pareto.best_compromise()
        my_linkage.set_constraints(best.dimensions)
    """
    _check_pymoo_available()

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions

    # Validate inputs
    if len(objectives) < 1:
        raise OptimizationError("At least one objective is required")

    if n_generations <= 0:
        raise OptimizationError(f"n_generations must be positive, got {n_generations}")

    if pop_size <= 0:
        raise OptimizationError(f"pop_size must be positive, got {pop_size}")

    # Get constraints and generate bounds if needed
    raw_constraints = tuple(linkage.get_constraints())
    constraints = tuple(c for c in raw_constraints if c is not None and isinstance(c, (int, float)))
    dimensions = len(constraints)

    if dimensions <= 0:
        raise OptimizationError(f"Dimensions must be positive, got {dimensions}")

    if bounds is None:
        generated = generate_bounds(constraints)
        bounds = (list(generated[0]), list(generated[1]))

    if len(bounds) != 2:
        raise OptimizationError(
            f"Bounds must be a tuple of (lower, upper), got {len(bounds)} elements"
        )

    if len(bounds[0]) != dimensions or len(bounds[1]) != dimensions:
        raise OptimizationError(
            f"Bounds dimensions ({len(bounds[0])}, {len(bounds[1])}) "
            f"must match number of dimensions ({dimensions})"
        )

    # Store initial joint positions
    joint_pos: tuple[tuple[float | None, float | None], ...] = tuple(linkage.get_coords())

    # Create the optimization problem
    problem = LinkageProblem(
        linkage,
        objectives,
        bounds,
        joint_pos,
        n_workers=n_workers,
        linkage_factory=linkage_factory,
    )

    # Select and configure the algorithm
    n_obj = len(objectives)
    algorithm_kwargs = dict(kwargs)

    if algorithm == "nsga2":
        algo = NSGA2(pop_size=pop_size, **algorithm_kwargs)
    elif algorithm == "nsga3":
        # NSGA-III requires reference directions
        if "ref_dirs" not in algorithm_kwargs:
            # Auto-generate reference directions
            n_partitions = max(4, 12 - n_obj)  # Fewer partitions for more objectives
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
            algorithm_kwargs["ref_dirs"] = ref_dirs
        algo = NSGA3(pop_size=pop_size, **algorithm_kwargs)
    else:
        raise OptimizationError(f"Unknown algorithm: {algorithm}. Use 'nsga2' or 'nsga3'.")

    # Run optimization
    try:
        result = minimize(
            problem.problem,
            algo,
            ("n_gen", n_generations),
            seed=seed,
            verbose=verbose,
        )
    finally:
        problem.close()

    # Generate objective names if not provided
    if objective_names is None:
        objective_names = tuple(f"Objective {i}" for i in range(n_obj))

    # Extract Pareto front
    if result.F is None or result.X is None:
        # No feasible solutions found — return empty Ensemble
        return Ensemble(
            linkage=linkage,
            dimensions=np.empty((0, dimensions), dtype=np.float64),
            initial_positions=np.empty((0, len(joint_pos), 2), dtype=np.float64),
            scores={name: np.empty(0, dtype=np.float64) for name in objective_names},
        )

    # Single-objective pymoo runs return the single best solution with
    # F shape (n_obj,) and X shape (n_var,); multi-objective runs return
    # the Pareto front with F shape (n_pop, n_obj) and X shape (n_pop, n_var).
    # Normalize both to 2-D so the same Ensemble-building code path works.
    F = np.atleast_2d(np.asarray(result.F))
    X = np.atleast_2d(np.asarray(result.X))
    if F.shape[1] != n_obj and F.shape[0] == n_obj:
        F = F.T

    n_solutions = len(F)
    joint_pos_arr = np.array(
        [(x if x is not None else 0.0, y if y is not None else 0.0) for x, y in joint_pos],
        dtype=np.float64,
    )
    all_positions = np.tile(joint_pos_arr, (n_solutions, 1, 1))

    scores_dict: dict[str, NDArray[np.float64]] = {}
    for k, name in enumerate(objective_names):
        scores_dict[name] = F[:, k].astype(np.float64)

    return Ensemble(
        linkage=linkage,
        dimensions=X.astype(np.float64),
        initial_positions=all_positions,
        scores=scores_dict,
    )
