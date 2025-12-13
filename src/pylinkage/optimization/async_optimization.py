"""
Async wrappers for optimization functions.

This module provides async versions of the optimization functions that allow:
- Progress callbacks without blocking
- Cancellation support via asyncio.CancelledError
- Integration with async frameworks

Created on 2025.

@author: HugoFara
"""


import asyncio
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .collections import Agent, MutableAgent
from .grid_search import trials_and_errors_optimization
from .particle_swarm import particle_swarm_optimization

if TYPE_CHECKING:
    from .._types import JointPositions
    from ..linkage.linkage import Linkage


@dataclass
class OptimizationProgress:
    """Progress information for async optimization.

    Attributes:
        current_iteration: Current iteration number.
        total_iterations: Total number of iterations.
        best_score: Best score found so far (may be None if not yet available).
        is_complete: Whether the optimization has completed.
    """

    current_iteration: int
    total_iterations: int
    best_score: float | None = None
    is_complete: bool = False

    @property
    def progress_fraction(self) -> float:
        """Return progress as a fraction between 0.0 and 1.0."""
        if self.total_iterations == 0:
            return 0.0
        return self.current_iteration / self.total_iterations


ProgressCallback = Callable[[OptimizationProgress], None]
"""Type alias for progress callback functions."""


async def particle_swarm_optimization_async(
    eval_func: "Callable[[Linkage, Sequence[float], JointPositions], float]",
    linkage: "Linkage",
    center: Sequence[float] | float | None = None,
    dimensions: int | None = None,
    n_particles: int = 100,
    leader: float = 3.0,
    follower: float = 0.1,
    inertia: float = 0.6,
    neighbors: int = 17,
    iters: int = 200,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    order_relation: Callable[[float, float], float] = max,
    on_progress: ProgressCallback | None = None,
    executor: ThreadPoolExecutor | None = None,
    **kwargs: Any,
) -> list[Agent]:
    """Async version of particle_swarm_optimization.

    This function runs the PSO optimization in a thread pool executor to avoid
    blocking the event loop, while providing progress callbacks and cancellation
    support.

    :param eval_func: The evaluation function.
        Input: (linkage, num_constraints, initial_coordinates).
        Output: score (float).
        The swarm will look for the HIGHEST score.
    :param linkage: Linkage to be optimized.
    :param center: A list of initial dimensions. If None, dimensions will be
        generated randomly between bounds. The default is None.
    :param dimensions: Number of dimensions of the swarm space.
        If None, it takes len(tuple(linkage.get_num_constraints())).
    :param n_particles: Number of particles in the swarm. The default is 100.
    :param inertia: Inertia of each particle. The default is 0.6.
    :param leader: Learning coefficient of each particle. The default is 3.0.
    :param follower: Social coefficient. The default is 0.1.
    :param neighbors: Number of neighbors to consider. The default is 17.
    :param iters: Number of iterations. The default is 200.
    :param bounds: Bounds to the space, in format (lower_bound, upper_bound).
    :param order_relation: How to compare scores (max or min). Default is max.
    :param on_progress: Optional callback function called with progress updates.
        The callback receives an OptimizationProgress object.
    :param executor: Optional ThreadPoolExecutor to use. If None, a default
        executor will be created.
    :param kwargs: Additional keyword arguments passed to LocalBestPSO.

    :returns: List of Agents: best score, best dimensions and initial positions.

    :raises asyncio.CancelledError: If the optimization is cancelled.
    :raises OptimizationError: If parameters are invalid or optimization fails.

    Example::

        async def my_optimization():
            def progress_handler(progress):
                print(f"Progress: {progress.progress_fraction:.1%}")

            results = await particle_swarm_optimization_async(
                eval_func=my_fitness,
                linkage=my_linkage,
                iters=100,
                on_progress=progress_handler,
            )
            return results
    """
    loop = asyncio.get_running_loop()

    # Signal progress at start
    if on_progress is not None:
        on_progress(OptimizationProgress(
            current_iteration=0,
            total_iterations=iters,
            best_score=None,
            is_complete=False,
        ))

    def run_optimization() -> list[Agent]:
        return particle_swarm_optimization(
            eval_func=eval_func,
            linkage=linkage,
            center=center,
            dimensions=dimensions,
            n_particles=n_particles,
            leader=leader,
            follower=follower,
            inertia=inertia,
            neighbors=neighbors,
            iters=iters,
            bounds=bounds,
            order_relation=order_relation,
            verbose=False,  # Disable verbose output in async mode
            **kwargs,
        )

    # Run optimization in thread pool
    if executor is not None:
        result = await loop.run_in_executor(executor, run_optimization)
    else:
        result = await loop.run_in_executor(None, run_optimization)

    # Signal completion
    if on_progress is not None:
        best_score = result[0].score if result else None
        on_progress(OptimizationProgress(
            current_iteration=iters,
            total_iterations=iters,
            best_score=best_score,
            is_complete=True,
        ))

    return result


async def trials_and_errors_optimization_async(
    eval_func: "Callable[[Linkage, Sequence[float], JointPositions], float]",
    linkage: "Linkage",
    parameters: Sequence[float] | None = None,
    n_results: int = 10,
    divisions: int = 5,
    on_progress: ProgressCallback | None = None,
    executor: ThreadPoolExecutor | None = None,
    **kwargs: Any,
) -> list[MutableAgent]:
    """Async version of trials_and_errors_optimization.

    This function runs the grid search optimization in a thread pool executor
    to avoid blocking the event loop, while providing progress callbacks and
    cancellation support.

    :param eval_func: Evaluation function.
        Input: (linkage, num_constraints, initial_coordinates).
        Output: score (float).
    :param linkage: Linkage to evaluate.
    :param parameters: Parameters that will be modified. If None, uses
        tuple(linkage.get_num_constraints()).
    :param n_results: Number of best candidates to return. The default is 10.
    :param divisions: Number of subdivisions between bounds. The default is 5.
    :param on_progress: Optional callback function called with progress updates.
    :param executor: Optional ThreadPoolExecutor to use. If None, a default
        executor will be created.
    :param kwargs: Additional arguments for the optimization:
        - bounds: A 2-tuple containing minimal and maximal bounds.
        - order_relation: Function to compare scores (max, min, abs).
        - sequential: If True, consecutive linkages have small variation.

    :returns: List of (score, dimensions, initial_position) tuples.

    :raises asyncio.CancelledError: If the optimization is cancelled.
    :raises OptimizationError: If parameters are invalid or no valid solution found.

    Example::

        async def my_optimization():
            def progress_handler(progress):
                print(f"Progress: {progress.progress_fraction:.1%}")

            results = await trials_and_errors_optimization_async(
                eval_func=my_fitness,
                linkage=my_linkage,
                divisions=10,
                on_progress=progress_handler,
            )
            return results
    """
    loop = asyncio.get_running_loop()

    # Calculate total iterations for progress
    if parameters is None:
        num_params = len(tuple(linkage.get_num_constraints()))
    else:
        num_params = len(parameters)
    total_iterations = divisions ** num_params

    # Signal progress at start
    if on_progress is not None:
        on_progress(OptimizationProgress(
            current_iteration=0,
            total_iterations=total_iterations,
            best_score=None,
            is_complete=False,
        ))

    def run_optimization() -> list[MutableAgent]:
        return trials_and_errors_optimization(
            eval_func=eval_func,
            linkage=linkage,
            parameters=parameters,
            n_results=n_results,
            divisions=divisions,
            verbose=False,  # Disable verbose output in async mode
            **kwargs,
        )

    # Run optimization in thread pool
    if executor is not None:
        result = await loop.run_in_executor(executor, run_optimization)
    else:
        result = await loop.run_in_executor(None, run_optimization)

    # Signal completion
    if on_progress is not None:
        best_score = result[0].score if result else None
        on_progress(OptimizationProgress(
            current_iteration=total_iterations,
            total_iterations=total_iterations,
            best_score=best_score,
            is_complete=True,
        ))

    return result
