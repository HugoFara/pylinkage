"""
Implementation of a particle swarm optimization.

Pure NumPy local-best PSO — no external dependencies beyond numpy.

Created on Fri Mar 8, 13:51:45 2019.

@author: HugoFara
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..exceptions import OptimizationError
from .collections import Agent

if TYPE_CHECKING:
    from .._types import JointPositions
    from ..linkage.linkage import Linkage


def _local_best_pso(
    objective: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    dimensions: int,
    bounds: tuple[NDArray[np.floating], NDArray[np.floating]],
    n_particles: int = 100,
    inertia: float = 0.6,
    c1: float = 3.0,
    c2: float = 0.1,
    neighbors: int = 17,
    iterations: int = 200,
    center: NDArray[np.floating] | float | None = None,
    verbose: bool = True,
) -> tuple[float, NDArray[np.floating]]:
    """Local-best Particle Swarm Optimization in pure NumPy.

    Each particle's social component is informed only by the best among its
    *k* nearest ring-topology neighbors, preventing premature convergence
    to a single basin.

    :param objective: Function mapping (n_particles, dimensions) array to
        (n_particles,) array of **costs to minimize**.
    :param dimensions: Number of parameters.
    :param bounds: ``(lower, upper)`` arrays each of shape ``(dimensions,)``.
    :param n_particles: Swarm size.
    :param inertia: Velocity damping factor *w*.
    :param c1: Cognitive (personal-best) acceleration coefficient.
    :param c2: Social (neighborhood-best) acceleration coefficient.
    :param neighbors: Ring-topology neighborhood size *k*.
    :param iterations: Number of iterations.
    :param center: Center for initial position sampling. If ``None``, particles
        are sampled uniformly within *bounds*; if a float, positions are sampled
        from ``U(lower, upper) * center``.
    :param verbose: Print iteration progress.
    :returns: ``(best_cost, best_position)``.
    """
    lb, ub = bounds
    rng = np.random.default_rng()

    # Initialize positions
    if center is None:
        positions = rng.uniform(lb, ub, size=(n_particles, dimensions))
    else:
        positions = rng.uniform(lb, ub, size=(n_particles, dimensions)) * float(center)
        positions = np.clip(positions, lb, ub)

    # Initialize velocities as small random values
    span = ub - lb
    velocities = rng.uniform(-span, span, size=(n_particles, dimensions)) * 0.1

    # Evaluate initial costs
    costs = objective(positions)
    personal_best_pos = positions.copy()
    personal_best_cost = costs.copy()

    global_best_idx = int(np.argmin(costs))
    global_best_cost = costs[global_best_idx]
    global_best_pos = positions[global_best_idx].copy()

    # Precompute ring-topology neighbor indices
    half_k = neighbors // 2
    neighbor_indices = np.array(
        [
            [(i + offset) % n_particles for offset in range(-half_k, half_k + 1)]
            for i in range(n_particles)
        ]
    )

    for iteration in range(iterations):
        # Find neighborhood best for each particle
        nbr_costs = personal_best_cost[neighbor_indices]  # (n_particles, k)
        best_nbr_local = np.argmin(nbr_costs, axis=1)  # index within neighborhood
        best_nbr_idx = neighbor_indices[np.arange(n_particles), best_nbr_local]
        nbr_best_pos = personal_best_pos[best_nbr_idx]

        # PSO velocity update
        r1 = rng.uniform(0, 1, size=(n_particles, dimensions))
        r2 = rng.uniform(0, 1, size=(n_particles, dimensions))
        velocities = (
            inertia * velocities
            + c1 * r1 * (personal_best_pos - positions)
            + c2 * r2 * (nbr_best_pos - positions)
        )

        # Update positions and clamp to bounds
        positions = np.clip(positions + velocities, lb, ub)

        # Evaluate
        costs = objective(positions)

        # Update personal bests
        improved = costs < personal_best_cost
        personal_best_pos[improved] = positions[improved]
        personal_best_cost[improved] = costs[improved]

        # Update global best
        iter_best_idx = int(np.argmin(personal_best_cost))
        if personal_best_cost[iter_best_idx] < global_best_cost:
            global_best_cost = personal_best_cost[iter_best_idx]
            global_best_pos = personal_best_pos[iter_best_idx].copy()

        if verbose:
            print(
                f"PSO iter {iteration + 1}/{iterations}  best cost: {global_best_cost:.6f}",
                end="\r",
            )

    if verbose:
        print()  # newline after carriage returns

    return float(global_best_cost), global_best_pos


def particle_swarm_optimization(
    eval_func: "Callable[[Linkage, Sequence[float], JointPositions], float]",
    linkage: "Linkage",
    center: Sequence[float] | float | None = None,
    dimensions: int | None = None,
    n_particles: int = 100,
    leader: float = 3.0,
    follower: float = 0.1,
    inertia: float = 0.6,
    neighbors: int = 17,
    iterations: int = 200,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    order_relation: Callable[[float, float], float] = max,
    verbose: bool = True,
    **kwargs: int,
) -> list[Agent]:
    """Particle Swarm Optimization for linkage parameters.

    Uses a local-best ring-topology PSO implemented in pure NumPy.

    :param eval_func: The evaluation function.
        Input: (linkage, num_constraints, initial_coordinates).
        Output: score (float).
        The swarm will look for the HIGHEST score.
    :param linkage: Linkage to be optimized. Make sure to give an optimized linkage for
        better results.
    :param center: A list of initial dimensions. If None, dimensions will be generated
        randomly between bounds. The default is None.
    :param dimensions: Number of dimensions of the swarm space, number of parameters.
        If None, it takes the value len(tuple(linkage.get_num_constraints())).
        The default is None.
    :param n_particles: Number of particles in the swarm. The default is 100.
    :param inertia: Inertia of each particle. The default is 0.6.
    :param leader: Cognitive acceleration coefficient (c1). The default is 3.0.
    :param follower: Social acceleration coefficient (c2). The default is 0.1.
    :param neighbors: Number of neighbors in ring topology. The default is 17.
    :param iterations: Number of iterations. The default is 200.
    :param bounds: Bounds to the space, in format (lower_bound, upper_bound).
        (Default value = None).
    :param order_relation: How to compare scores.
        There should not be anything else than the built-in
        max and min functions.
        The default is max.
    :param verbose: The optimization state will be printed in the console if True.
        (Default value = True).

    :returns: List of Agents: best score, best dimensions and initial positions.

    :raises OptimizationError: If parameters are invalid or optimization fails.
    """
    # Backwards-compatible alias
    if "iters" in kwargs:
        iterations = kwargs.pop("iters")
    if dimensions is None:
        dimensions = len(tuple(linkage.get_num_constraints()))
    if dimensions <= 0:
        raise OptimizationError(f"Dimensions must be positive, got {dimensions}")
    if n_particles <= 0:
        raise OptimizationError(f"Number of particles must be positive, got {n_particles}")
    if iterations <= 0:
        raise OptimizationError(f"Number of iterations must be positive, got {iterations}")
    if bounds is not None:
        if len(bounds) != 2:
            raise OptimizationError(
                f"Bounds must be a tuple of (lower, upper), got {len(bounds)} elements"
            )
        if len(bounds[0]) != dimensions or len(bounds[1]) != dimensions:
            raise OptimizationError(
                f"Bounds dimensions ({len(bounds[0])}, {len(bounds[1])}) "
                f"must match number of dimensions ({dimensions})"
            )
    else:
        # Default bounds: [-10, 10] per dimension
        bounds = ([-10.0] * dimensions, [10.0] * dimensions)

    joint_pos: tuple[tuple[float | None, float | None], ...] = tuple(
        linkage.get_coords()
    )

    np_bounds = (np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float))

    def objective(dims: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate all particles; return costs to minimize."""
        costs = np.empty(dims.shape[0])
        for i, d in enumerate(dims):
            score = eval_func(linkage, d.tolist(), joint_pos)
            costs[i] = -score if order_relation is max else score
        return costs

    center_val: NDArray[np.floating] | float | None
    if center is not None and not isinstance(center, (int, float)):
        center_val = np.asarray(center, dtype=float)
    else:
        center_val = center

    best_cost, best_pos = _local_best_pso(
        objective=objective,
        dimensions=dimensions,
        bounds=np_bounds,
        n_particles=n_particles,
        inertia=inertia,
        c1=leader,
        c2=follower,
        neighbors=neighbors,
        iterations=iterations,
        center=center_val,
        verbose=verbose,
    )

    # Un-negate for maximization
    best_score = -best_cost if order_relation is max else best_cost
    return [Agent(best_score, best_pos, joint_pos)]
