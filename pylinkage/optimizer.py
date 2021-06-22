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
from pyswarms.utils.plotters import plot_cost_history

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
    Generator
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


def exhaustive_optimization(eval_func, linkage, parameters, n_results=10,
                            delta_dim=.5, min_dim=2, max_dim=2):
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


def particle_swarm_optimization(eval_func, linkage, begin, n_indi=21,
                                delta_dim=.3, iner=.5, leader=.3, follower=.6,
                                neigh=1, lifetime=0, blind_iter=5, merge=.1,
                                ite=1000, iterable=False, bounds=None,
                                **kwargs):
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
    begin : tuple of float
        initial dimensions.
    n_indi : float, optional
        Number of particles in the swarm. The default is 21.
    delta_dim : float, optional
        Unused, for legacy purposes. The default is .3.
    iner : float, optional
        Inertia of each particle, w in pyswarms. The default is .8.
    leader : float, optional
        Learning coefficient of each particle, c1 in pyswarms.
        The default is .6.
    follower : float, optional
        Social coefficient, c2 in pyswarms. The default is .8.
    neigh : float, optional
        With the Euclidian norm, distance under a particle is a neighbour.
        The default is 1.
    lifetime : TYPE, optional
        DESCRIPTION. The default is 0.
    blind_iter : TYPE, optional
        DESCRIPTION. The default is 5.
    merge : TYPE, optional
        DESCRIPTION. The default is .1.
    ite : TYPE, optional
        DESCRIPTION. The default is 1000.
    iterable : TYPE, optional
        DESCRIPTION. The default is False.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        List of 3 elements: best dimensions, best score and initial positions.

    """
    dims_len = len(begin)
    options = {'c1': leader, 'c2': follower, 'w': iner, 'p': 1, 'k': 5}
    pos = tuple(j.coord() for j in linkage.joints)
    optimizer = LocalBestPSO(
        n_particles=n_indi, dimensions=dims_len, options=options,
        bounds=bounds, **kwargs)
    # vectorized_eval_func=np.vectorize(eval_func, signature='(n),(m,k)->()')
    out = optimizer.optimize(
        # vectorized_eval_func,
        lambda dims, pos: np.array([-eval_func(d, pos) for d in dims]),
        ite, pos=pos)
    plot_cost_history(optimizer.cost_history)
    return optimizer
    return [(out[0], out[1], pos)]


class Particle(object):
    """Simple particle swarm optimization (PSO) class."""

    __slots__ = ("pos", "speed", "eval", "ini", "swarm",
                 "index", "max_dist", "score", "best_score",
                 "best_pos", "dists", "neighbours")

    def __init__(self, pos, speed, ini, eva, swarm, max_dist=float('inf')):
        """
        Generate swarm.

        Arguments:
        pos: particule position
        speed: intial velocity
        ini: additionnal data (initial position of the linkage)
        eva: evaluation function
        swarm: swarm the particule belongs to
        max_dist: maximum distance at which another particule can attract
        this one
        """

        self.pos = np.array(pos)
        self.speed = np.array(speed)
        self.ini, self.eval = ini, eva
        self.swarm = swarm
        self.index = len(self.swarm)
        self.swarm.append(self)
        self.max_dist = max_dist
        self.best_score = -float('inf')
        self.score = -float('inf')
        self.new_score()
        self.best_pos, self.best_score = self.pos, self.score
        self.dists = np.empty(len(self.swarm) * (len(self.swarm) - 1) // 2)
        self.neighbours = tuple()

    def __get_row__(self):
        """Get the row in swarm index."""
        return (self.index * (len(self.swarm) - 3)
                - (self.index - 1) * (self.index - 2) // 2)

    def __del__(self):
        """Update swarm indexes."""
        if len(self.swarm) > self.index:
            self.swarm.pop(self.index)
            for p in self.swarm[self.index:]:
                p.index -= 1
        self.update_dists()
        for p in self.swarm:
            p.update_neighbours()

    """Methods affecting a single agent"""

    def new_score(self):
        """
        Get score at current position, update best_score and best_position.

        The update is made  if the new score is superior OR equal to the
        previous one.

        Returns
        -------
        float
            Current score.

        """
        self.score = self.eval(self.pos, self.ini)
        # Large equality important when score is null.
        # (avoid staying on null position)
        if self.score >= self.best_score:
            self.best_pos, self.best_score = self.pos, self.score
        return self.score

    def update_neighbours(self):
        """Update this particle list of neighbours."""
        row = self.__get_row__()
        self.neighbours = tuple(
                filter(
                        lambda p: (p is not self
                                   and self.dists[row + p.index] < self.max_dist ** 2),
                                   self.swarm))

    def move(self, blind=False):
        """
        Move the particle.

        Start by updating position using speed, then update particles data.
        """
        self.pos += self.speed
        if blind:
            return
        self.update_neighbours()
        self.score = self.eval(self.pos, self.ini)
        # Large equality important when score is null.
        # (avoid staying on null position)
        if self.score >= self.best_score:
            self.best_pos, self.best_score = self.pos, self.score
        # update_dists()
        # new_score()

    def die(self):
        """Destroy particle. Equivalent to __del__."""
        self.__del__()

    # Methods affecting whole swarm.

    def update_dists(self):
        """Update the list of distances relative to this particle."""
        l_s = len(self.swarm)
        dists = np.empty(l_s * (l_s - 1) // 2)
        # TODO: IN PARALLEL
        for p1 in self.swarm:
            p1.dists = dists
            row = p1.__get_row__()
            for p2 in self.swarm[self.index + 1:]:
                if row + p2.index >= len(dists):
                    raise Warning(
                            "Index overflow!\nrow: {}"
                            ", index: {}, len(dists):  {}".format(
                                    row, p2.index, len(dists)))
                dists[row + p2.index] = sqr_dist(p1.pos, p2.pos)
                p2.dists = self.dists

    def kill_swarm(self):
        print("Decimation")
        #del dists
        for p in self.swarm:
            del p.dists
        self.swarm.clear()


def particle_swarm_optimization_legacy(
        eval_func, linkage, begin, n_indi=21, delta_dim=.3, iner=.8, leader=.6,
        follower=.8, neigh=1, lifetime=0, blind_iter=5, merge=.1,
        ite=1000, iterable=False):
    """
    Particle swarm optimization. We look for best solutions to eval_func,
    starting from a begin set with real values. We use a particle swarm of
    n_indi agents.

    Return an iterable of list of agents' position.

    Parameters:
        - eval_func: function giving the score to each position
        - linkage: linka["linkage"]
        - begin: iterable of initial positions (dimensions of the linkage)
        - n_indi: nomber of agents in swarm
        - min_dim: each dimension is divised by min_dim. If an agent is below
        begin[i] / min_dim, his score is set to 0
        - max_dim: upper bound, agents above begin[i] * max_dim have score 0
        - iner: agents inertia
        - leader: trust is self best position, learning coefficient
        - follower: social coefficient
        - lifetime: if an agent has score 0 for lifetime frames, he dies.
    """
    c = [i.coord() for i in linkage.joints]
    swarm = []
    Particle(begin.copy(), 0, c.copy(), eval_func, swarm, neigh)
    for p in range(n_indi - 1):
        # They run away for initial position at given speed
        v = rand(len(begin)) - .5
        Particle(begin.copy(), delta_dim * v, deepcopy(c),
                 eval_func, swarm, neigh)
    swarm[0].update_dists()

    #print("Identity : ", swarm[0] is swarm[-1])
    if iterable:
        yield [(p.pos, p.score, p.ini) for p in swarm], -1
    c = [0] * len(swarm)
    #print("agents : ", len(swarm))
    # We start calulations
    for j in range(ite):
        # Save intial positions
        c[:] = [p.pos.copy() for p in swarm]
        # Update parameters
        for p in swarm:
            # Update agent position
            p.move(j < blind_iter)
            if j < blind_iter:
                v = rand(len(begin)) - .5
                p.speed = delta_dim / norm(v) * v
                #if p.score:
                #    p.ini = [i.coord() for i in linkage]
                continue
            # Best position in neighborhood
            pg = np.array(max(p.neighbours, key=lambda x: x.score, default=p).pos)
            # print("Does" + ("" if max(filter(dists[i]) == pg else "not") + " work"))
            # Update speed
            # USE norm(pg - p.pos), norm(p.best_bos - p.pos), check stability
            p.speed = (iner * p.speed
                       + leader * rand() * (pg - p.pos)
                       + follower * rand() * (p.best_pos - p.pos))
            # Score evaluation
            if p.new_score():
                p.ini = [i.coord() for i in linkage.joints]
        if merge and j > blind_iter:
            for p in swarm:
                # TODO: add a second axis to self.dists
                deletion = False
                for k in filter(
                        lambda x: sqr_dist(p.speed, x.speed) < merge ** 2,
                        p.neighbours):
                    print('Merge mark')
                    # Least score particule to be deleted
                    (p if p.score < k.score else k).__del__()
                    #del swarm[max(p.index, b_score_part.index)]
                    #swarm[min(p.index, b_score_part.index)] = b_score_part
                    deletion = True
                    break
                if deletion:
                    #p.update_dists()
                    #p.update_neighbours()
                    break

        if iterable:
            # yield map(lambda p: (p.pos, p.score, p.ini), swarm), j
            yield [(p.pos, p.score, p.ini) for p in swarm], j
    if not iterable:
        coco = [(p.pos, p.score, p.ini) for p in swarm]
    # swarm detruction
    #swarm[0].kill_swarm()
    if not iterable:
        return coco
