"""
Implementation of a grid search optimization.

It should be used for reference only as the search space
will almost certainly be too big.
"""
import math
import itertools
import numpy as np
import tqdm
from .utils import generate_bounds
from .collections import MutableAgent


def tqdm_verbosity(iterable, verbose=True, *args, **kwargs):
    """Wrapper for tqdm, that let you specify if you want verbosity.
    
    .. deprecated:: 0.6.0
          `tqdm_verbosity` will be removed in pylinkage 0.7.0, as tqdm can be
            disabled with the argument disable=True.

    :param iterable: 
    :param verbose:  (Default value = True)
    :param args: Ordered args to pass to tqdm
    :param kwargs: Keyword args for tqdm

    """
    for i in tqdm.tqdm(iterable, disable=not verbose, *args, **kwargs):
        yield i


def sequential_variator(center, divisions, bounds):
    """Return an iterable of each possible variation for the elements.
    
    Number of variations: ((max_dim - 1 / min_dim) / delta_dim) ** len(ite).
    
    Because linkage is not tolerant to violent changes, the order of output
    for the coefficients is very important.
    
    The coefficient is in order: middle → min (step 2), min → middle (step 2),
    middle → max (step 1), so that there is no huge variation.

    :param center: Elements that should vary.
    :type center: Iterable[float]
    :param divisions: Number of subdivisions between `bounds`.
    :type divisions: int
    :param bounds: 2-uple of minimal then maximal bounds.
    :type bounds: tuple[tuple[float], tuple[float]]
    :returns: An iterable of all the dimension combinations.
    :rtype: Generator[float]
    """
    # In the first place, we go in decreasing order to lower bound
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
    """Return an iterable of elements' all possibles variations.
    
    Number of variations: ((max_dim - 1 / min_dim) / delta_dim) ** len(ite).
    
    Here the order in the variations is not important.

    :param divisions: Number of subdivisions between `bounds`.
    :type divisions: int
    :param bounds: 2-uple of minimal then maximal bounds.
    :type bounds: tuple[tuple[float], tuple[float]]
    :returns: An iterable of all the dimension combinations.
    :rtype: Generator[float]
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
    """Return the list of dimensions optimizing eval_func.
    
    Each dimension set has a score, which is added in an array of n_results
    results, contains the linkages with the best scores in a maximization problem
    by default.

    :param eval_func: Evaluation function.
        Input: (linkage, num_constraints, initial_coordinates).
        Output: score (float)
    :type eval_func: Callable
    :param linkage: Linkage to evaluate.
    :type linkage: pylinkage.linkage.Linkage
    :param parameters: Parameters that will be modified. Geometric constraints.
                       If not, it will be assigned tuple(linkage.get_num_constraints()).
                       The default is None.
    :type parameters: list
    :param n_results: Number of the best candidates to return. The default is 10.
    :type n_results: int
    :param divisions: Number of subdivisions between bounds. The default is 5.
    :type divisions: int
    :param kwargs:
        - Extra arguments for the optimization.
        - bounds : A 2-uple (tuple of two elements), containing the minimal and maximal bounds.
            If None, we will use parameters as a center.
            (Default value = None).
        - order_relation : A function of two arguments, should return the best score of two
            scores. Common examples are `min`, `max`, `abs`.
            (Default value = :func:`max`).
        - verbose : The number of combinations will be printed in console if `True`.
            (Default value = True).
        - sequential : If True, two consecutive linkages will have a small variation.

    :type kwargs: dict

    :returns: 3-uplet of score, dimensions and initial position for each Linkage to
        return.
        Its size is {n_results}.
    :rtype: list[MutableAgent]
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
    results = [MutableAgent() for _ in range(n_results)]
    prev = [i.coord() for i in linkage.joints]
    # We start by a "fall": we do not want to break the system by modifying
    # dimensions, so we assess it is normally behaving, and we change
    # dimensions progressively to minimal dimensions.
    postfix = {
        "best score": None,
        "best dimensions": []
    }
    if 'sequential' in kwargs and kwargs['sequential']:
        variations_generator = sequential_variator(center, divisions, bounds)
    else:
        variations_generator = fast_variator(divisions, bounds)
    if 'order_relation' in kwargs:
        order_relation = kwargs['order_relation']
    else:
        order_relation = max
    verbose = 'verbose' in kwargs and kwargs['verbose']
    # Iterable of all possible dimensions
    pbar = tqdm.tqdm(
        variations_generator,
        desc='Trials and errors optimization',
        total=divisions ** len(center),
        postfix=postfix,
        disable=not verbose
    )
    for dim in pbar:
        # Check performances
        new_score = eval_func(linkage, dim, prev)
        for result in results:
            if result.score is None or order_relation(result.score, new_score) != result.score:
                result[:] = new_score, dim.copy(), prev.copy()
                if verbose:
                    postfix.update({
                        "best score": results[0][0],
                        "best dimensions": results[0][1]
                    })
                    pbar.set_postfix(postfix)
                break
    if verbose:
        print(
            "Trials and errors optimization finished. "
            f"Best score: {results[0][0]}, best dimensions: {results[0][1]}"
        )
    return results
