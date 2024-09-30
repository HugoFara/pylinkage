"""
Implementation of a particle swarm optimization.

Created on Fri Mar 8, 13:51:45 2019.

@author: HugoFara
"""
from pyswarms.single.local_best import LocalBestPSO
from ..collections import Agent


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
    """Particle Swarm Optimization wrapper for pyswarms.
    
    This function is a simple wrapper to optimize a linkage using PSO. It will
    mainly call the LocalBestPSO function from pyswarms.single.

    :param eval_func: The evaluation function.
        Input: (linkage, num_constraints, initial_coordinates).
        Output: score (float).
        The swarm will look for the HIGHEST score.
    :type eval_func: Callable -> float
    :param linkage: Linkage to be optimized. Make sure to give an optimized linkage for
        better results
    :type linkage: pylinkage.linkage.Linkage
    :param center: A list of initial dimensions. If None, dimensions will be generated
        randomly between bounds. The default is None.
    :type center: list
    :param dimensions: Number of dimensions of the swarm space, number of parameters.
        If None, it takes the value len(tuple(linkage.get_num_constraints())).
        The default is None.
    :type dimensions: int
    :param n_particles: Number of particles in the swarm. The default is 100.
    :type n_particles: float
    :param inertia: Inertia of each particle, w in pyswarms. The default is .3.
    :type inertia: float
    :param leader: Learning coefficient of each particle, c1 in pyswarms.
        The default is .2.
    :type leader: float
    :param follower: Social coefficient, c2 in pyswarms. The default is .5.
    :type follower: float
    :param neighbors: Number of neighbors to consider. The default is 17.
    :type neighbors: int
    :param iters: Number of iterations to describe. The default is 200.
    :type iters: int
    :param bounds: Bounds to the space, in format (lower_bound, upper_bound). (Default value = None)
    :type bounds: sequence of two list of float
    :param order_relation: How to compare scores.
        There should not be anything else than the built-in
        max and min functions.
        The default is max.
    :type order_relation: Callable(float, float) -> float
    :param verbose: The optimization state will be printed in the console if True.
        (Default value = True).
    :type verbose: bool
    :param kwargs: keyword arguments to pass to pyswarm.local.single.LocalBestPSO.
    :type kwargs: dict

    :returns: List of Agents: best score, best dimensions and initial positions.
    :rtype: List[Agent]
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
        """Wrapper for the evaluation function since PySwarms is too rigid.

        :param dims: 

        """
        if order_relation is max:
            return [-eval_func(linkage, d, joint_pos) for d in dims]
        elif order_relation is min:
            return [eval_func(linkage, d, joint_pos) for d in dims]

    score, constraints = optimizer.optimize(
        eval_wrapper,
        iters,
        verbose=verbose
    )
    return [Agent(score, constraints, joint_pos)]
