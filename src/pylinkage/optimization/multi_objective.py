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
from .collections.pareto import ParetoFront, ParetoSolution
from .utils import generate_bounds

if TYPE_CHECKING:
    from .._types import JointPositions
    from ..linkage.linkage import Linkage


def _check_pymoo_available() -> None:
    """Check if pymoo is installed and raise informative error if not."""
    try:
        import pymoo  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pymoo is required for multi-objective optimization. "
            "Install with: pip install pylinkage[moo]"
        ) from e


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
    ) -> None:
        """Initialize the linkage optimization problem.

        Args:
            linkage: The linkage to optimize.
            objectives: List of objective functions. Each takes
                (linkage, constraints, joint_positions) and returns a float.
            bounds: Tuple of (lower_bounds, upper_bounds) for constraints.
            joint_pos: Initial joint positions.
        """
        from pymoo.core.problem import Problem

        self.linkage = linkage
        self.objectives = objectives
        self.joint_pos = joint_pos

        n_var = len(bounds[0])
        n_obj = len(objectives)

        # Create the problem class dynamically
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
                F = np.zeros((len(X), n_obj))
                for i, x in enumerate(X):
                    for j, obj in enumerate(self.objectives):
                        F[i, j] = obj(self.linkage, x.tolist(), self.joint_pos)
                out["F"] = F

        self._problem = _Problem()

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
    **kwargs: Any,
) -> ParetoFront:
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
        **kwargs: Additional arguments passed to the algorithm.

    Returns:
        ParetoFront containing all non-dominated solutions.

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
        my_linkage.set_num_constraints(best.dimensions)
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
    raw_constraints = tuple(linkage.get_num_constraints())
    constraints = tuple(
        c for c in raw_constraints if c is not None and isinstance(c, (int, float))
    )
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
    joint_pos: tuple[tuple[float | None, float | None], ...] = tuple(
        j.coord() for j in linkage.joints
    )

    # Create the optimization problem
    problem = LinkageProblem(linkage, objectives, bounds, joint_pos)

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
    result = minimize(
        problem.problem,
        algo,
        ("n_gen", n_generations),
        seed=seed,
        verbose=verbose,
    )

    # Extract Pareto front
    if result.F is None or result.X is None:
        # No feasible solutions found
        return ParetoFront(
            solutions=[],
            objective_names=tuple(objective_names) if objective_names else tuple(),
        )

    solutions = []
    for i in range(len(result.F)):
        sol = ParetoSolution(
            scores=tuple(result.F[i]),
            dimensions=result.X[i],
            init_positions=joint_pos,
        )
        solutions.append(sol)

    # Generate objective names if not provided
    if objective_names is None:
        objective_names = tuple(f"Objective {i}" for i in range(n_obj))

    return ParetoFront(
        solutions=solutions,
        objective_names=tuple(objective_names),
    )
