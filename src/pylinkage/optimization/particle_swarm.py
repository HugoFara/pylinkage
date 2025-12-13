"""
Implementation of a particle swarm optimization.

Created on Fri Mar 8, 13:51:45 2019.

@author: HugoFara
"""


from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from pyswarms.single.local_best import LocalBestPSO

from ..exceptions import OptimizationError
from .collections import Agent

if TYPE_CHECKING:
    from .._types import JointPositions
    from ..linkage.linkage import Linkage


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
    iters: int = 200,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    order_relation: Callable[[float, float], float] = max,
    verbose: bool = True,
    **kwargs: Any,
) -> list[Agent]:
    """Particle Swarm Optimization wrapper for pyswarms.

    This function is a simple wrapper to optimize a linkage using PSO. It will
    mainly call the LocalBestPSO function from pyswarms.single.

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
    :param inertia: Inertia of each particle, w in pyswarms. The default is .3.
    :param leader: Learning coefficient of each particle, c1 in pyswarms.
        The default is .2.
    :param follower: Social coefficient, c2 in pyswarms. The default is .5.
    :param neighbors: Number of neighbors to consider. The default is 17.
    :param iters: Number of iterations to describe. The default is 200.
    :param bounds: Bounds to the space, in format (lower_bound, upper_bound).
        (Default value = None).
    :param order_relation: How to compare scores.
        There should not be anything else than the built-in
        max and min functions.
        The default is max.
    :param verbose: The optimization state will be printed in the console if True.
        (Default value = True).
    :param kwargs: keyword arguments to pass to pyswarm.local.single.LocalBestPSO.

    :returns: List of Agents: best score, best dimensions and initial positions.

    :raises OptimizationError: If parameters are invalid or optimization fails.
    """
    if dimensions is None:
        dimensions = len(tuple(linkage.get_num_constraints()))
    if dimensions <= 0:
        raise OptimizationError(f"Dimensions must be positive, got {dimensions}")
    if n_particles <= 0:
        raise OptimizationError(f"Number of particles must be positive, got {n_particles}")
    if iters <= 0:
        raise OptimizationError(f"Number of iterations must be positive, got {iters}")
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
    options: dict[str, float | int] = {
        'c1': leader,
        'c2': follower,
        'w': inertia,
        'k': neighbors,
        'p': 1,
    }
    joint_pos: tuple[tuple[float | None, float | None], ...] = tuple(
        j.coord() for j in linkage.joints
    )
    optimizer = LocalBestPSO(
        n_particles=n_particles,
        dimensions=dimensions,
        options=options,
        bounds=bounds,
        center=center if center is not None else 1.0,
        **kwargs,
    )

    def eval_wrapper(dims: NDArray[np.floating]) -> list[float]:
        """Wrapper for the evaluation function since PySwarms is too rigid.

        :param dims: Dimensions to evaluate.
        """
        if order_relation is max:
            return [-eval_func(linkage, d, joint_pos) for d in dims]
        elif order_relation is min:
            return [eval_func(linkage, d, joint_pos) for d in dims]
        return [eval_func(linkage, d, joint_pos) for d in dims]

    score, constraints = optimizer.optimize(
        eval_wrapper,
        iters,
        verbose=verbose,
    )
    return [Agent(score, constraints, joint_pos)]
