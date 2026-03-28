"""Service layer for optimization endpoints.

Converts mechanism dicts to Linkage objects, builds objective functions
from frontend specifications, runs optimization, and returns results
as mechanism dicts.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from pylinkage.exceptions import UnbuildableError
from pylinkage.linkage.analysis import bounding_box
from pylinkage.mechanism.conversion import mechanism_from_linkage, mechanism_to_linkage
from pylinkage.mechanism.serialization import mechanism_from_dict, mechanism_to_dict
from pylinkage.optimization import (
    differential_evolution_optimization,
    generate_bounds,
    particle_swarm_optimization,
    trials_and_errors_optimization,
)
from pylinkage.optimization.scipy_optimize import minimize_linkage
from pylinkage.optimization.collections import Agent, MutableAgent

from ..models.optimization_schemas import (
    AlgorithmParams,
    DifferentialEvolutionParams,
    GridSearchParams,
    NelderMeadParams,
    ObjectiveSpec,
    OptimizationRequest,
    OptimizationResponse,
    OptimizationResultDTO,
    PSOParams,
)

logger = logging.getLogger(__name__)


# --- Objective function builders ---


def _path_length(loci: tuple[tuple[Any, ...], ...], joint_index: int) -> float:
    """Total path length of a joint."""
    total = 0.0
    for i in range(1, len(loci)):
        prev = loci[i - 1][joint_index]
        curr = loci[i][joint_index]
        if prev is None or curr is None:
            continue
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def _bounding_box_area(loci: tuple[tuple[Any, ...], ...], joint_index: int) -> float:
    """Bounding box area of a joint's path."""
    points = [step[joint_index] for step in loci if step[joint_index] is not None]
    if len(points) < 2:
        return 0.0
    y_min, x_max, y_max, x_min = bounding_box(points)
    return (x_max - x_min) * (y_max - y_min)


def _x_extent(loci: tuple[tuple[Any, ...], ...], joint_index: int) -> float:
    """Horizontal extent of a joint's path."""
    points = [step[joint_index] for step in loci if step[joint_index] is not None]
    if len(points) < 2:
        return 0.0
    xs = [p[0] for p in points]
    return max(xs) - min(xs)


def _y_extent(loci: tuple[tuple[Any, ...], ...], joint_index: int) -> float:
    """Vertical extent of a joint's path."""
    points = [step[joint_index] for step in loci if step[joint_index] is not None]
    if len(points) < 2:
        return 0.0
    ys = [p[1] for p in points]
    return max(ys) - min(ys)


def _target_path_distance(
    loci: tuple[tuple[Any, ...], ...],
    joint_index: int,
    target_points: list[list[float]],
) -> float:
    """Sum of minimum distances from each target point to the joint's path."""
    path = [step[joint_index] for step in loci if step[joint_index] is not None]
    if not path:
        return float("inf")
    total = 0.0
    for tx, ty in target_points:
        min_dist = float("inf")
        for px, py in path:
            d = math.sqrt((px - tx) ** 2 + (py - ty) ** 2)
            if d < min_dist:
                min_dist = d
        total += min_dist
    return total


def _build_eval_func(
    objective: ObjectiveSpec, minimize: bool
) -> Any:
    """Build an evaluation function from an objective specification.

    Returns a raw fitness function (not yet wrapped with kinematic_default_test).
    The function signature is: (linkage, params, init_pos) -> float.
    """
    error_penalty = float("inf") if minimize else -float("inf")

    def eval_func(linkage: Any, params: Any, init_pos: Any) -> float:
        from pylinkage.linkage.linkage import Linkage

        if init_pos is not None:
            linkage.set_coords(init_pos)
        linkage.set_num_constraints(params)
        try:
            points = 12
            n = linkage.get_rotation_period()
            # Quick buildability check
            tuple(
                tuple(i) for i in linkage.step(iterations=points + 1, dt=n / points)
            )
            # Full simulation for fitness evaluation
            n_steps = 96
            factor = int(points / n_steps) + 1
            loci = tuple(
                tuple(i)
                for i in linkage.step(iterations=n_steps * factor, dt=1 / factor)
            )
        except UnbuildableError:
            return error_penalty

        joint_index = objective.joint_index

        if objective.type == "path_length":
            score = _path_length(loci, joint_index)
        elif objective.type == "bounding_box_area":
            score = _bounding_box_area(loci, joint_index)
        elif objective.type == "x_extent":
            score = _x_extent(loci, joint_index)
        elif objective.type == "y_extent":
            score = _y_extent(loci, joint_index)
        elif objective.type == "target_path":
            score = _target_path_distance(loci, joint_index, objective.target_points)
        else:
            return error_penalty

        return score

    return eval_func


def _agent_to_result(
    agent: Agent | MutableAgent, linkage: Any, warnings: list[str]
) -> OptimizationResultDTO:
    """Convert an Agent to a result DTO, including the updated mechanism dict."""
    score = agent.score if agent.score is not None else 0.0
    dims = list(np.asarray(agent.dimensions).flat) if agent.dimensions is not None else []

    mechanism_dict = None
    try:
        linkage.set_num_constraints(dims)
        if agent.init_positions is not None:
            linkage.set_coords(agent.init_positions)
        # Rebuild to get valid positions
        n = linkage.get_rotation_period()
        for _ in linkage.step(iterations=1, dt=n):
            break
        mechanism = mechanism_from_linkage(linkage)
        mechanism_dict = mechanism_to_dict(mechanism)
    except Exception as e:
        logger.warning("Failed to convert optimized result to mechanism: %s", e)
        warnings.append(f"Could not convert result to mechanism: {e}")

    return OptimizationResultDTO(
        score=score,
        constraints=dims,
        mechanism_dict=mechanism_dict,
    )


def run_optimization(request: OptimizationRequest) -> OptimizationResponse:
    """Run optimization on a mechanism."""
    warnings: list[str] = []

    # Build Mechanism from dict, then convert to Linkage
    mechanism = mechanism_from_dict(request.mechanism)
    linkage = mechanism_to_linkage(mechanism)

    # Build evaluation function
    eval_func = _build_eval_func(request.objective, request.minimize)

    # Determine order relation
    order_relation = min if request.minimize else max

    # Get current constraints for bounds generation
    constraints = list(linkage.get_num_constraints())
    constraint_names = [
        f"constraint_{i}" for i in range(len(constraints))
    ]

    # Generate bounds
    np_constraints = np.array(constraints, dtype=float)
    bounds = (
        np_constraints / request.bounds_factor,
        np_constraints * request.bounds_factor,
    )

    algo = request.algorithm

    # PSO returns the raw pyswarms cost (negated when maximizing),
    # so we need to un-negate the score for the response.
    pso_negate_score = False

    if isinstance(algo, PSOParams):
        pso_negate_score = not request.minimize
        agents = particle_swarm_optimization(
            eval_func=eval_func,
            linkage=linkage,
            center=constraints,
            n_particles=algo.n_particles,
            iters=algo.iterations,
            inertia=algo.inertia,
            leader=algo.leader,
            follower=algo.follower,
            neighbors=min(algo.neighbors, algo.n_particles),
            bounds=bounds,
            order_relation=order_relation,
            verbose=False,
        )
    elif isinstance(algo, DifferentialEvolutionParams):
        mutation: tuple[float, float] | float
        if len(algo.mutation) == 2:
            mutation = (algo.mutation[0], algo.mutation[1])
        else:
            mutation = algo.mutation[0]
        agents = differential_evolution_optimization(
            eval_func=eval_func,
            linkage=linkage,
            bounds=bounds,
            order_relation=order_relation,
            strategy=algo.strategy,
            maxiter=algo.max_iterations,
            popsize=algo.population_size,
            tol=algo.tolerance,
            mutation=mutation,
            recombination=algo.recombination,
            seed=algo.seed,
            verbose=False,
        )
    elif isinstance(algo, NelderMeadParams):
        agents = minimize_linkage(
            eval_func=eval_func,
            linkage=linkage,
            x0=constraints,
            bounds=bounds,
            order_relation=order_relation,
            method="Nelder-Mead",
            maxiter=algo.max_iterations,
            tol=algo.tolerance,
            verbose=False,
        )
    elif isinstance(algo, GridSearchParams):
        agents = trials_and_errors_optimization(
            eval_func=eval_func,
            linkage=linkage,
            parameters=constraints,
            n_results=algo.n_results,
            divisions=algo.divisions,
            bounds=bounds,
            order_relation=order_relation,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Convert results
    results: list[OptimizationResultDTO] = []
    for agent in agents:
        if agent.score is None:
            continue
        result = _agent_to_result(agent, linkage, warnings)
        # Un-negate PSO scores (pyswarms returns negated cost when maximizing)
        if pso_negate_score:
            result.score = -result.score
        results.append(result)

    best_score = results[0].score if results else None

    return OptimizationResponse(
        results=results,
        best_score=best_score,
        constraint_names=constraint_names,
        warnings=warnings,
    )
