# -*- coding: utf-8 -*-
"""
The optimizer module proposing different optimization algorithms.

The output of these functions is generally dimensions of linkages.

Created on Fri Mar 8, 13:51:45 2019.

@author: HugoFara
"""
# Particle swarm optimization
from pyswarms.single.local_best import LocalBestPSO


def particle_swarm_optimization(
        eval_func,
        linkage,
        center=None,
        dimensions=None,
        n_particles=100,
        leader=3.0, follower=.1, inertia=0.6, neighbors=17,
        iters=200,
        bounds=None,
        order_relation=max,
        verbose=True,
        **kwargs
):
    """
    Particle Swarm Optimization wrapper for pyswarms.

    This function is a simple wrapper to optimize a linkage using PSO. It will
    mainly call the LocalBestPSO function from pyswarms.single.

    Parameters
    ----------
    eval_func : callable -> float
        The evaluation function.
        Input: (linkage, num_constraints, initial_coordinates).
        Output: score (float).
        The swarm will look for the HIGHEST score.
    linkage : pylinkage.linkage.Linkage
        Linkage to be optimized. Make sure to give an optimized linkage for
        better results
    center : list, optional
        A list of initial dimensions. If None, dimensions will be generated
        randomly between bounds. The default is None.
    dimensions : int, optional
        Number of dimensions of the swarm space, number of parameters.
        If None, it takes the value len(tuple(linkage.get_num_constraints())).
        The default is None.
    n_particles : float, optional
        Number of particles in the swarm. The default is 100.
    inertia : float, optional
        Inertia of each particle, w in pyswarms. The default is .3.
    leader : float, optional
        Learning coefficient of each particle, c1 in pyswarms.
        The default is .2.
    follower : float, optional
        Social coefficient, c2 in pyswarms. The default is .5.
    neighbors : int, optional
        Number of neighbors to consider. The default is 17.
    iters : int, optional
        Number of iterations to describe. The default is 200.
    bounds : sequence of two list of float
        Bounds to the space, in format (lower_bound, upper_bound).
    order_relation : callable(float, float) -> float, optional
        How to compare scores.
        There should not be anything else than the built-in
        max and min functions.
        The default is max.
    verbose : bool, optional
        The optimization state will be printed in the console if True.
        The default is True.
    **kwargs : dict
        keyword arguments to pass to pyswarm.local.single.LocalBestPSO.

    Returns
    -------
    list[float, list[float], list[list[float]]]
        List of 3 elements: best score, best dimensions and initial positions.

    """
    if dimensions is None:
        dimensions = len(tuple(linkage.get_num_constraints()))
    options = {
        'c1': leader,
        'c2': follower,
        'w': inertia,
        'k': neighbors,
        'p': 1,
    }
    joint_pos = tuple(j.coord() for j in linkage.joints)
    optimizer = LocalBestPSO(
        n_particles=n_particles,
        dimensions=dimensions,
        options=options,
        bounds=bounds,
        center=center if center is not None else 1.0,
        **kwargs
    )
    # vectorized_eval_func=np.vectorize(eval_func, signature='(n),(m,k)->()')

    def eval_wrapper(dims):
        """Wrapper for the evaluation function since PySwarms is too rigid."""
        if order_relation is max:
            return [-eval_func(linkage, d, joint_pos) for d in dims]
        elif order_relation is min:
            return [eval_func(linkage, d, joint_pos) for d in dims]

    score, constraints = optimizer.optimize(
        eval_wrapper,
        iters,
        verbose=verbose
    )
    return [(score, constraints, joint_pos)]
