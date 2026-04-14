"""
SciPy-based optimization algorithms for linkage optimization.

This module provides wrappers around SciPy's optimization functions:
- differential_evolution_optimization: Global optimization using differential evolution
- dual_annealing_optimization: Global optimization using generalized simulated annealing
- minimize_linkage: Local optimization using various gradient-free methods

Created on 2025.

@author: HugoFara
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from ..exceptions import OptimizationError
from ..population import Ensemble
from .utils import generate_bounds


def _make_ensemble(
    linkage: "Linkage",
    best_score: float,
    best_dims: np.ndarray,
    joint_pos: tuple[tuple[float | None, float | None], ...],
) -> Ensemble:
    """Build a single-member Ensemble from an optimizer result."""
    joint_pos_arr = np.array(
        [(x if x is not None else 0.0, y if y is not None else 0.0) for x, y in joint_pos],
        dtype=np.float64,
    )
    return Ensemble(
        linkage=linkage,
        dimensions=np.asarray(best_dims, dtype=np.float64).reshape(1, -1),
        initial_positions=joint_pos_arr.reshape(1, -1, 2),
        scores={"score": np.array([best_score])},
    )


if TYPE_CHECKING:
    from typing import Any as Linkage  # accepts legacy/sim Linkage and Mechanism

    from .._types import JointPositions


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
) -> Ensemble:
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

    :returns: Ensemble with the best result (single member).

    :raises OptimizationError: If parameters are invalid or optimization fails.

    Example::

        from pylinkage.optimization import differential_evolution_optimization
        from pylinkage.optimization.utils import kinematic_minimization

        @kinematic_minimization
        def fitness(loci, **kwargs):
            return some_metric(loci)

        result = differential_evolution_optimization(
            eval_func=fitness,
            linkage=my_linkage,
            maxiter=500,
            order_relation=min,
        )
        best = result[0]  # Member with .score, .dimensions, .initial_positions
    """
    raw_constraints = tuple(linkage.get_constraints())
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
    joint_pos: tuple[tuple[float | None, float | None], ...] = tuple(linkage.get_coords())

    def objective(x: np.ndarray) -> float:
        """Objective function wrapper for scipy."""
        score = eval_func(linkage, x.tolist(), joint_pos)
        # scipy minimizes, so negate for maximization
        if order_relation is max:
            return -score
        return score

    from scipy.optimize import differential_evolution

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
    return _make_ensemble(linkage, best_score, result.x, joint_pos)


def dual_annealing_optimization(
    eval_func: "Callable[[Linkage, Sequence[float], JointPositions], float]",
    linkage: "Linkage",
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    order_relation: Callable[[float, float], float] = max,
    maxiter: int = 1000,
    initial_temp: float = 5230.0,
    restart_temp_ratio: float = 2e-5,
    visit: float = 2.62,
    accept: float = -5.0,
    seed: int | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> Ensemble:
    """Dual Annealing optimization wrapper for scipy.

    This function wraps scipy's generalized simulated annealing optimizer.
    Unlike population-based methods (PSO, DE), it follows a single trajectory
    with controlled random jumps, making it effective for problems with many
    local minima and expensive evaluations.

    :param eval_func: The evaluation function.
        Input: (linkage, num_constraints, initial_coordinates).
        Output: score (float).
    :param linkage: Linkage to be optimized.
    :param bounds: Bounds to the space, in format (lower_bound, upper_bound).
        If None, bounds will be generated from linkage constraints.
    :param order_relation: How to compare scores (max or min). Default is max.
    :param maxiter: Maximum number of global iterations. Default is 1000.
    :param initial_temp: Initial temperature for the annealing schedule.
        Higher values allow more exploration. Default is 5230.0.
    :param restart_temp_ratio: Fraction of initial_temp at which the
        temperature is restarted. Default is 2e-5.
    :param visit: Parameter for the visiting distribution. Higher values
        give heavier tails (more long-range jumps). Default is 2.62.
    :param accept: Parameter for the acceptance distribution. Lower values
        make acceptance more restrictive. Default is -5.0.
    :param seed: Random seed for reproducibility.
    :param verbose: Print progress if True. Default is True.
    :param kwargs: Additional keyword arguments passed to dual_annealing.

    :returns: Ensemble with the best result (single member).

    :raises OptimizationError: If parameters are invalid or optimization fails.

    Example::

        from pylinkage.optimization import dual_annealing_optimization
        from pylinkage.optimization.utils import kinematic_minimization

        @kinematic_minimization
        def fitness(loci, **kwargs):
            return some_metric(loci)

        result = dual_annealing_optimization(
            eval_func=fitness,
            linkage=my_linkage,
            maxiter=500,
            order_relation=min,
        )
        best = result[0]  # Member with .score, .dimensions, .initial_positions
    """
    raw_constraints = tuple(linkage.get_constraints())
    constraints = cast(
        tuple[float, ...],
        tuple(c for c in raw_constraints if c is not None and isinstance(c, (int, float))),
    )
    dimensions = len(constraints)

    if dimensions <= 0:
        raise OptimizationError(f"Dimensions must be positive, got {dimensions}")
    if maxiter <= 0:
        raise OptimizationError(f"Maximum iterations must be positive, got {maxiter}")

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

    scipy_bounds = list(zip(bounds[0], bounds[1], strict=True))

    joint_pos: tuple[tuple[float | None, float | None], ...] = tuple(linkage.get_coords())

    def objective(x: np.ndarray) -> float:
        score = eval_func(linkage, x.tolist(), joint_pos)
        if order_relation is max:
            return -score
        return score

    from scipy.optimize import dual_annealing

    callback = None
    if verbose:
        _iter_count = [0]

        def callback(x: np.ndarray, f: float, context: int) -> bool:
            _iter_count[0] += 1
            score = -f if order_relation is max else f
            print(
                f"DA iter {_iter_count[0]}  score: {score:.6f}  context: {context}",
                end="\r",
            )
            return False

    result = dual_annealing(
        objective,
        bounds=scipy_bounds,
        maxiter=maxiter,
        initial_temp=initial_temp,
        restart_temp_ratio=restart_temp_ratio,
        visit=visit,
        accept=accept,
        seed=seed,
        callback=callback,
        **kwargs,
    )

    if verbose:
        print()

    best_score = -result.fun if order_relation is max else result.fun
    return _make_ensemble(linkage, best_score, result.x, joint_pos)


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
) -> Ensemble:
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

    :returns: Ensemble with the best result (single member).

    :raises OptimizationError: If parameters are invalid or optimization fails.

    Example::

        from pylinkage.optimization import minimize_linkage
        from pylinkage.optimization.utils import kinematic_minimization

        @kinematic_minimization
        def fitness(loci, **kwargs):
            return some_metric(loci)

        # Refine a solution from global optimization
        result = minimize_linkage(
            eval_func=fitness,
            linkage=my_linkage,
            x0=initial_guess,
            method="Nelder-Mead",
            order_relation=min,
        )
        best = result[0]  # Member with .score, .dimensions, .initial_positions
    """
    raw_constraints = tuple(linkage.get_constraints())
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
    joint_pos: tuple[tuple[float | None, float | None], ...] = tuple(linkage.get_coords())

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

    from scipy.optimize import minimize

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
    return _make_ensemble(linkage, best_score, result.x, joint_pos)


def chain_optimizers(
    eval_func: "Callable[[Linkage, Sequence[float], JointPositions], float]",
    linkage: "Linkage",
    stages: Sequence[
        tuple[
            "Callable[..., Ensemble]",
            dict[str, Any],
        ]
    ],
    order_relation: Callable[[float, float], float] = max,
    verbose: bool = True,
) -> Ensemble:
    """Run multiple optimizers in sequence, feeding each result to the next.

    A common pattern is global search followed by local refinement, e.g.
    DE or PSO for exploration, then Nelder-Mead to polish the best solution.

    Each stage receives the best solution from the previous stage as its
    starting point (via ``center`` for population-based methods, or ``x0``
    for local methods).

    :param eval_func: The evaluation function, shared across all stages.
    :param linkage: Linkage to be optimized.
    :param stages: Sequence of ``(optimizer_func, kwargs)`` tuples. Each
        ``optimizer_func`` must accept ``eval_func`` and ``linkage`` as
        its first two positional arguments. ``kwargs`` are passed through.
        Do **not** include ``eval_func`` or ``linkage`` in ``kwargs``.
    :param order_relation: How to compare scores (max or min). Default is max.
    :param verbose: Print stage headers and progress. Default is True.

    :returns: Ensemble from the final optimization stage.

    :raises OptimizationError: If no stages are provided.

    Example::

        from pylinkage.optimization import (
            chain_optimizers,
            differential_evolution_optimization,
            minimize_linkage,
        )

        result = chain_optimizers(
            eval_func=fitness,
            linkage=my_linkage,
            stages=[
                (differential_evolution_optimization, {"maxiter": 300}),
                (minimize_linkage, {"method": "Nelder-Mead", "maxiter": 500}),
            ],
            order_relation=min,
        )
        best = result[0]
    """
    if not stages:
        raise OptimizationError("At least one optimization stage is required")

    best_result: Ensemble | None = None

    for i, (optimizer, kwargs) in enumerate(stages):
        if verbose:
            name = getattr(optimizer, "__name__", str(optimizer))
            print(f"\n=== Stage {i + 1}/{len(stages)}: {name} ===")

        stage_kwargs: dict[str, Any] = {
            "eval_func": eval_func,
            "linkage": linkage,
            "order_relation": order_relation,
            "verbose": verbose,
            **kwargs,
        }

        # Feed previous result as starting point
        if best_result is not None:
            best_dims = best_result[0].dimensions
            # Local methods use x0, population-based use center
            if "x0" not in stage_kwargs and "center" not in stage_kwargs:
                import inspect

                sig = inspect.signature(optimizer)
                if "x0" in sig.parameters:
                    stage_kwargs["x0"] = list(best_dims)
                elif "center" in sig.parameters:
                    stage_kwargs["center"] = list(best_dims)

        result = optimizer(**stage_kwargs)
        best_result = result

    assert best_result is not None
    return best_result
