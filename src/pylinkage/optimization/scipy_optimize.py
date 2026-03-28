"""
SciPy-based optimization algorithms for linkage optimization.

This module provides wrappers around SciPy's optimization functions:
- differential_evolution_optimization: Global optimization using differential evolution
- minimize_linkage: Local optimization using various gradient-free methods

Created on 2025.

@author: HugoFara
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from scipy.optimize import differential_evolution, minimize

from ..exceptions import OptimizationError
from .collections import Agent
from .utils import generate_bounds

if TYPE_CHECKING:
    from .._types import JointPositions
    from ..linkage.linkage import Linkage


def differential_evolution_optimization(
    eval_func: "Callable[[Linkage, Sequence[float], JointPositions], float]",
    linkage: "Linkage",
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    order_relation: Callable[[float, float], float] = max,
    strategy: str = "best1bin",
    maxiter: int = 1000,
    popsize: int = 15,
    tol: float = 0.01,
    mutation: tuple[float, float] | float = (0.5, 1.0),
    recombination: float = 0.7,
    seed: int | None = None,
    workers: int = 1,
    verbose: bool = True,
    **kwargs: Any,
) -> list[Agent]:
    """Differential Evolution optimization wrapper for scipy.

    This function is a wrapper to optimize a linkage using Differential
    Evolution, a global optimization algorithm that does not require
    gradient information. It is generally faster than grid search and
    can handle multimodal objective functions.

    :param eval_func: The evaluation function.
        Input: (linkage, num_constraints, initial_coordinates).
        Output: score (float).
        The optimizer will look for the score based on order_relation.
    :param linkage: Linkage to be optimized.
    :param bounds: Bounds to the space, in format (lower_bound, upper_bound).
        If None, bounds will be generated from linkage constraints using
        generate_bounds().
    :param order_relation: How to compare scores (max or min). Default is max.
    :param strategy: Differential evolution strategy. Options include:
        'best1bin', 'best1exp', 'rand1exp', 'randtobest1exp', 'best2exp',
        'rand2exp', 'randtobest1bin', 'best2bin', 'rand2bin', 'rand1bin'.
        Default is "best1bin".
    :param maxiter: Maximum number of generations. Default is 1000.
    :param popsize: Population size multiplier. The total population will be
        popsize * dimensions. Default is 15.
    :param tol: Relative tolerance for convergence. Default is 0.01.
    :param mutation: Mutation constant (dithering). Can be a float in [0, 2]
        or a tuple (min, max). Default is (0.5, 1.0).
    :param recombination: Recombination constant in [0, 1]. Default is 0.7.
    :param seed: Random seed for reproducibility.
    :param workers: Number of parallel workers. Use -1 for all CPUs. Default is 1.
    :param verbose: Print progress if True. Default is True.
    :param kwargs: Additional keyword arguments passed to differential_evolution.

    :returns: List containing single Agent with best score, dimensions, and positions.

    :raises OptimizationError: If parameters are invalid or optimization fails.

    Example::

        from pylinkage.optimization import differential_evolution_optimization
        from pylinkage.optimization.utils import kinematic_minimization

        @kinematic_minimization
        def fitness(loci, **kwargs):
            return some_metric(loci)

        results = differential_evolution_optimization(
            eval_func=fitness,
            linkage=my_linkage,
            maxiter=500,
            order_relation=min,
        )
        best_score, best_dims, init_pos = results[0]
    """
    raw_constraints = tuple(linkage.get_num_constraints())
    # Filter to get only float values for bounds generation
    constraints = cast(
        tuple[float, ...],
        tuple(c for c in raw_constraints if c is not None and isinstance(c, (int, float))),
    )
    dimensions = len(constraints)

    if dimensions <= 0:
        raise OptimizationError(f"Dimensions must be positive, got {dimensions}")
    if maxiter <= 0:
        raise OptimizationError(f"Maximum iterations must be positive, got {maxiter}")
    if popsize <= 0:
        raise OptimizationError(f"Population size must be positive, got {popsize}")

    # Generate bounds if not provided
    if bounds is None:
        generated = generate_bounds(constraints)
        bounds = (list(generated[0]), list(generated[1]))

    # Validate bounds
    if len(bounds) != 2:
        raise OptimizationError(
            f"Bounds must be a tuple of (lower, upper), got {len(bounds)} elements"
        )
    if len(bounds[0]) != dimensions or len(bounds[1]) != dimensions:
        raise OptimizationError(
            f"Bounds dimensions ({len(bounds[0])}, {len(bounds[1])}) "
            f"must match number of dimensions ({dimensions})"
        )

    # Convert bounds to scipy format: list of (min, max) tuples
    scipy_bounds = list(zip(bounds[0], bounds[1], strict=True))

    # Store initial joint positions
    joint_pos: tuple[tuple[float | None, float | None], ...] = tuple(
        j.coord() for j in linkage.joints
    )

    def objective(x: np.ndarray) -> float:
        """Objective function wrapper for scipy."""
        score = eval_func(linkage, x.tolist(), joint_pos)
        # scipy minimizes, so negate for maximization
        if order_relation is max:
            return -score
        return score

    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds=scipy_bounds,
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        seed=seed,
        workers=workers,
        disp=verbose,
        **kwargs,
    )

    # Convert result
    best_score = -result.fun if order_relation is max else result.fun
    best_dims = result.x

    return [Agent(best_score, best_dims, joint_pos)]


def minimize_linkage(
    eval_func: "Callable[[Linkage, Sequence[float], JointPositions], float]",
    linkage: "Linkage",
    x0: Sequence[float] | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    order_relation: Callable[[float, float], float] = max,
    method: Literal["Nelder-Mead", "Powell", "COBYLA", "L-BFGS-B", "SLSQP", "TNC"] = "Nelder-Mead",
    maxiter: int | None = None,
    tol: float | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> list[Agent]:
    """Local optimization using scipy.optimize.minimize.

    This function is a wrapper for local optimization of a linkage using
    scipy's minimize function. It supports various gradient-free methods
    suitable for linkage optimization.

    :param eval_func: The evaluation function.
        Input: (linkage, num_constraints, initial_coordinates).
        Output: score (float).
        The optimizer will look for the score based on order_relation.
    :param linkage: Linkage to be optimized.
    :param x0: Initial guess for the parameters. If None, uses current
        linkage constraints.
    :param bounds: Bounds to the space, in format (lower_bound, upper_bound).
        If None and method supports bounds, bounds will be generated from
        linkage constraints.
    :param order_relation: How to compare scores (max or min). Default is max.
    :param method: Optimization method. Options:
        - "Nelder-Mead": Simplex method, no bounds (default)
        - "Powell": Powell's method, no bounds
        - "COBYLA": Constrained optimization by linear approximation
        - "L-BFGS-B": Limited-memory BFGS with bounds
        - "SLSQP": Sequential Least Squares Programming
        - "TNC": Truncated Newton with bounds
    :param maxiter: Maximum number of iterations. If None, uses scipy default.
    :param tol: Tolerance for termination. If None, uses scipy default.
    :param verbose: Print progress if True. Default is True.
    :param kwargs: Additional keyword arguments passed to minimize.

    :returns: List containing single Agent with best score, dimensions, and positions.

    :raises OptimizationError: If parameters are invalid or optimization fails.

    Example::

        from pylinkage.optimization import minimize_linkage
        from pylinkage.optimization.utils import kinematic_minimization

        @kinematic_minimization
        def fitness(loci, **kwargs):
            return some_metric(loci)

        # Refine a solution from global optimization
        results = minimize_linkage(
            eval_func=fitness,
            linkage=my_linkage,
            x0=initial_guess,
            method="Nelder-Mead",
            order_relation=min,
        )
        best_score, best_dims, init_pos = results[0]
    """
    raw_constraints = tuple(linkage.get_num_constraints())
    # Filter to get only float values
    constraints = cast(
        tuple[float, ...],
        tuple(c for c in raw_constraints if c is not None and isinstance(c, (int, float))),
    )
    dimensions = len(constraints)

    if dimensions <= 0:
        raise OptimizationError(f"Dimensions must be positive, got {dimensions}")

    # Use current constraints as initial guess if not provided
    if x0 is None:
        x0 = list(constraints)

    if len(x0) != dimensions:
        raise OptimizationError(
            f"Initial guess length ({len(x0)}) must match dimensions ({dimensions})"
        )

    # Methods that support bounds
    bounded_methods = {"L-BFGS-B", "SLSQP", "TNC"}

    # Generate bounds for bounded methods if not provided
    scipy_bounds = None
    if method in bounded_methods:
        if bounds is None:
            generated = generate_bounds(constraints)
            bounds = (list(generated[0]), list(generated[1]))
        # Convert to scipy format
        scipy_bounds = list(zip(bounds[0], bounds[1], strict=True))

    # Store initial joint positions
    joint_pos: tuple[tuple[float | None, float | None], ...] = tuple(
        j.coord() for j in linkage.joints
    )

    def objective(x: np.ndarray) -> float:
        """Objective function wrapper for scipy."""
        score = eval_func(linkage, x.tolist(), joint_pos)
        # scipy minimizes, so negate for maximization
        if order_relation is max:
            return -score
        return score

    # Build options dict
    options: dict[str, Any] = {}
    if maxiter is not None:
        options["maxiter"] = maxiter
    if verbose:
        options["disp"] = True

    # Run minimize
    result = minimize(
        objective,
        x0=np.array(x0),
        method=method,
        bounds=scipy_bounds,
        tol=tol,
        options=options if options else None,
        **kwargs,
    )

    # Convert result
    best_score = -result.fun if order_relation is max else result.fun
    best_dims = result.x

    return [Agent(best_score, best_dims, joint_pos)]
