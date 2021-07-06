# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:51:45 2019.

@author: HugoFara

Module proposing different function optimization algorithms. The output is
generally leg dimensions of walking linkages.
"""
from copy import deepcopy
import numpy as np
from numpy.random import rand
from numpy.linalg import norm
# Particle swarm optimization
from pyswarms.single.local_best import LocalBestPSO

from .geometry import sqr_dist


def variator(ite, delta_dim, min_dim=5, max_dim=5):
    """
    Return an iterable of all possibles variations of elements.

    Number of variations: ((max_dim - 1 / min_dim) / delta_dim) ** len(ite).

    Because linkage are not tolerant to violent changes, the order of output
    for the coefficients is very important.

    The coefficient is in order: middle → min (step 2), min → middle (step 2),
    middle → max (step 1), so that there is no huge variation.

    Parameters
    ----------
    ite : sequence of floats
        Elements that should vary.
    delta_dim : float
        Scale factor for each variation.
    min_dim : float, optional
        Minimal scale reduction (dimensions not shorter that dim/min_dim).
        The default is 5.
    max_dim : float, optional
        maximal scale augmentation (not above dim * max_dim). The default is 5.

    Returns
    -------
    generator
        Each element is the list of floats with little variations.

    """
    # We copy the sequence
    c = tuple(ite)
    inv = 1 / min_dim
    variations = int((max_dim - inv) / delta_dim)
    coef = tuple(inv + i * delta_dim for i in range(variations))
    middle, even = variations // 2, variations % 2
    # We reordinate the coo
    coef = (coef[middle:0:-2] + coef[0:1]
            + coef[even:middle:2] + coef[middle+1:])
    return recurs_variator(c, list(c), coef)


def recurs_variator(ite, copy, coef_list, num=0):
    """
    Recursive dimensions generator.

    Called by variator. Only "copy" is modified.
    """
    for coef in coef_list:
        copy[num] = ite[num] * coef
        yield copy
        if num < len(ite) - 1:
            for j in recurs_variator(ite, copy, coef_list, num + 1):
                yield j


def trials_and_errors_optimization(
        eval_func,
        linkage,
        parameters=None,
        n_results=10,
        delta_dim=.5,
        min_dim=2,
        max_dim=2
):
    """
    Return the list of dimensions optimizing eval_func.

    We start wy making the dimensions vary, then we try a crank revolution on
    10 points. If no error we try on 75 points (higher precision).

    Each dimensions set has a score, which is added in an array of n_results
    results, containg the linkages with best scores.

    Parameters
    ----------
    eval_func : callable
        Evaluation function. Its signature should be R^len(linkage.joints) → R.
    linkage : pylinkage.linkage.Linkage
        Linkage to evaluate.
    parameters : list
        Parameters that will be modified. Geometric constraints.
        If not, it will be assignated tuple(linkage.get_num_constraints()).
        The default is None.
    n_results : int, optional
        Number of best cancidates to return. The default is 10.
    delta_dim : float, optional
        Dimension variation between two consecutive tries. The default is .5.
    min_dim : float, optional
        Minimal scale reduction.
        Each parameter should not be below original size / min_dim.
        The default is 2.
    max_dim : float, optional
        Maximal scale augmentation factor. The default is 2.

    Returns
    -------
    results : tuple
        tuple of dimensions, score, initial position for each Linkage to
        return. Its size is {n_results}.

    """
    n = (max_dim - 1 / min_dim) / delta_dim
    if parameters is None:
        parameters = list(linkage.get_num_constraints())
    print("Computation running, about {} combinations...".format(
        int(n) ** len(parameters)))
    # Results to output: scores, dimensions and initial positions
    # scores will be in decreasing order
    results = [[-float('inf'), [], []] for i in range(n_results)]
    prev = [i.coord() for i in linkage.joints]
    # We start by a "fall" : we do not want to break the system by modifying
    # dimensions, so we assess it is normally behaving, and we change
    # dimensions progressively to minimal dimensions.
    # A list of dictionaries of all possible dimensions
    for dim in variator(parameters, delta_dim, min_dim, max_dim):
        # Check performances
        score = eval_func(linkage, dim, prev)
        for r in results:
            if r[0] < score:
                r[0] = score
                r[1] = dim.copy()
                r[2] = prev.copy()
                break
        if score and False:
            # Save initial positions if score not null, for further
            # computations
            prev = [k.coord() for k in linkage.joints]
    return results


def particle_swarm_optimization(
        eval_func,
        linkage,
        center=None,
        dimensions=None,
        n_particles=100,
        iner=.3, leader=.2, follower=.5,
        iters=200,
        bounds=None,
        **kwargs
):
    """
    Particle Swarm Optimization wrapper for pyswarms.

    This function is a simple wrapper to optimize a linkage using PSO. It will
    mainly call the LocalBestPSO function from pyswarms.single.

    Parameters
    ----------
    eval_func : callable, must return float
        The evaluation function.
        Input: a set of dimensions and intial position for the linkage
        Output: a score, a float
        The swarm will look for the HIGHEST score.
    linkage : Linkage
        Linkage to be optimized. Make sure to give an optimized linkage for
        better results
    center : list
        A list of initial dimension. If None, dimensions will be generated
        randomly between bounds. The default is None.
    dimensions : int
        Number of dimensions of the swarm space, number of parameters.
        If None, it takes the value len(tuple(linkage.get_num_constraints())).
        The default is None.
    n_particles : float, optional
        Number of particles in the swarm. The default is 100.
    iner : float, optional
        Inertia of each particle, w in pyswarms. The default is .3.
    leader : float, optional
        Learning coefficient of each particle, c1 in pyswarms.
        The default is .2.
    follower : float, optional
        Social coefficient, c2 in pyswarms. The default is .5.
    iters : int, optional
        Number of iterations to describe. The default is 200.
    **kwargs : dict
        keyword arguments to pass to pyswarm.local.single.LocalBestPSO.

    Returns
    -------
    list
        List of 3 elements: best dimensions, best score and initial positions.

    """
    if dimensions is None:
        dimensions = len(tuple(linkage.get_num_constraints()))
    options = {'c1': leader, 'c2': follower, 'w': iner, 'p': 1, 'k': 5}
    pos = tuple(j.coord() for j in linkage.joints)
    optimizer = LocalBestPSO(
        n_particles=n_particles,
        dimensions=dimensions,
        options=options,
        bounds=bounds,
        center=center if center is not None else 1.0,
        **kwargs
    )
    # vectorized_eval_func=np.vectorize(eval_func, signature='(n),(m,k)->()')
    out = optimizer.optimize(
        # vectorized_eval_func,
        lambda dims, pos: np.array([-eval_func(d, pos) for d in dims]),
        iters, pos=pos
    )
    return optimizer
    return [(out[0], out[1], pos)]
