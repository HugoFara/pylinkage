# -*- coding: utf-8 -*-
"""
The optimizer module proposing different optimization algorithms.

The output of these functions is generally dimensions of linkages.

Created on Fri Mar 8, 13:51:45 2019.

@author: HugoFara
"""
import math
import itertools
import numpy as np
# Particle swarm optimization
from pyswarms.single.local_best import LocalBestPSO
# Progress bar
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


def tqdm_verbosity(iterable, verbose=True, *args, **kwargs):
    """Wrapper for tqdm, that let you specify if you want verbosity."""
    if verbose and tqdm:
        for i in tqdm(iterable, *args, **kwargs):
            yield i
    else:
        for i in iterable:
            yield i


def generate_bounds(center, min_ratio=5, max_factor=5):
    """
    Simple function to generate bounds from a linkage.

    Parameters
    ----------
    center : sequence
        1-D sequence, often in the form of ``linkage.get_num_constraints()``.
    min_ratio : float, optional
        Minimal compression ratio for the bounds. Minimal bounds will be of the
        shape center[x] / min_ratio.
        The default is 5.
    max_factor : float, optional
        Dilation factor for the upper bounds. Maximal bounds will be of the
        shape center[x] * max_factor.
        The default is 5.
    """
    np_center = np.array(center)
    return np_center / min_ratio, np_center * max_factor


def sequential_variator(center, divisions, bounds):
    """
    Return an iterable of each possible variation for the elements.

    Number of variations: ((max_dim - 1 / min_dim) / delta_dim) ** len(ite).

    Because linkage is not tolerant to violent changes, the order of output
    for the coefficients is very important.

    The coefficient is in order: middle → min (step 2), min → middle (step 2),
    middle → max (step 1), so that there is no huge variation.

    Parameters
    ----------
    center : sequence of float
        Elements that should vary.
    divisions : int
        Number of subdivisions between `bounds`.
    bounds : tuple[tuple[float]]
        2-uple of minimal then maximal bounds.

    Returns
    -------
    generator
        Each element is the list of floats with little variations.

    """
    # In the first place w go decreasing order to lower bound
    fall = np.linspace(center, bounds[0], int(divisions / 2))
    # We only look at one index over 2
    for dim in fall[::2]:
        yield dim
    # Then we go back to the center with the remaining indexes
    if divisions % 2:
        for dim in fall[-2::-2]:
            yield dim
    else:
        for dim in fall[::-2]:
            yield dim
    # Save memory, a list can be long
    del fall
    # And last indexes
    for dim in np.linspace(center, bounds[1], math.ceil(divisions / 2)):
        yield dim


def fast_variator(divisions, bounds):
    """
    Return an iterable of all possibles variations of elements.

    Number of variations: ((max_dim - 1 / min_dim) / delta_dim) ** len(ite).

    Here the order in the variations is not important.

    Parameters
    ----------
    divisions : int
        Number of subdivisions between `bounds`.
    bounds : tuple[tuple[float]]
        2-uple of minimal then maximal bounds.

    Returns
    -------
    generator
        An iterable of all the dimension combinations.

    See Also
    --------

    """
    lists = (
        iter(np.linspace(low, high, divisions))
        for low, high in zip(bounds[0], bounds[1])
    )
    for j in itertools.product(*lists):
        yield list(j)


def trials_and_errors_optimization(
        eval_func,
        linkage,
        parameters=None,
        n_results=10,
        divisions=5,
        **kwargs
):
    """
    Return the list of dimensions optimizing eval_func.

    Each dimension set has a score, which is added in an array of n_results
    results, contains the linkages with the best scores in a maximization problem
    by default.

    Parameters
    ----------
    eval_func : callable
        Evaluation function.
        Input: (linkage, num_constraints, initial_coordinates).
        Output: score (float)
    linkage : pylinkage.linkage.Linkage
        Linkage to evaluate.
    parameters : list, optional
        Parameters that will be modified. Geometric constraints.
        If not, it will be assigned tuple(linkage.get_num_constraints()).
        The default is None.
    n_results : int, optional
        Number of the best candidates to return. The default is 10.
    divisions : int, optional
        Number of subdivisions between bounds. The default is 5.
    **kwargs : dict, optional
        Extra arguments for the optimization.
        
        bounds : tuple[tuple], optional
            A 2-uple (tuple of two elements), containing the minimal and maximal
            bounds. If None, we will use parameters as a center.
            The default is None.
        order_relation : callable, optional
            A function of two arguments, should return the best score of two
            scores. Common examples are `min`, `max`, `abs`.
            The default is `max`.
        verbose : bool, optional
            The number of combinations will be printed in console if `True`.
            The default is `True`.
        sequential : bool
            If True, two consecutive linkages will have a small variation.

    Returns
    -------
    results : tuple[tuple[float, tuple[float], tuple[tuple[float]]]]
        3-uplet of score, dimensions and initial position for each Linkage to
        return. Its size is {n_results}.

    """
    if parameters is None:
        center = np.array(linkage.get_num_constraints())
    else:
        center = np.array(parameters)
    if 'bounds' not in kwargs or kwargs['bounds'] is None:
        bounds = generate_bounds(center)
    else:
        bounds = kwargs['bounds']

    # Results to output: scores, dimensions and initial positions
    # scores will be in decreasing order
    results = [[None, [], []] for _ in range(n_results)]
    prev = [i.coord() for i in linkage.joints]
    # We start by a "fall": we do not want to break the system by modifying
    # dimensions, so we assess it is normally behaving, and we change
    # dimensions progressively to minimal dimensions.
    postfix = ["best score", None, "best dimensions", []]
    if 'sequential' in kwargs and kwargs['sequential']:
        variator = sequential_variator(center, divisions, bounds)
    else:
        variator = fast_variator(divisions, bounds)
    if 'order_relation' in kwargs:
        order_relation = kwargs['order_relation']
    else:
        order_relation = max
    verbose = 'verbose' in kwargs and kwargs['verbose']
    # An iterable of all possible dimensions
    variations = tqdm_verbosity(
        iterable=variator,
        total=divisions ** len(center),
        desc='trials_and_errors_optimization',
        postfix=postfix,
        verbose=verbose,
    )
    for dim in variations:
        # Check performances
        score = eval_func(linkage, dim, prev)
        for r in results:
            if r[0] is None or order_relation(r[0], score) != r[0]:
                r[0] = score
                r[1] = dim.copy()
                r[2] = prev.copy()
                if verbose:
                    postfix[1] = results[0][0]
                    postfix[3] = results[0][1]
                break
        if score and False:
            # Save initial positions if score not null, for further
            # computations
            prev = [k.coord() for k in linkage.joints]
    if verbose:
        print(
            "Trials and errors optimization finished. "
            "Best score: {}, best dimensions: {}".format(
                *results[0][:2]
            )
        )
    return results


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
