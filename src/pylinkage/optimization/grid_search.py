"""
Implementation of a grid search optimization.

It should be used for reference only as the search space
will almost certainly be too big.
"""

import itertools
import math
from collections.abc import Callable, Generator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import tqdm
from numpy.typing import NDArray

from .._types import JointPositions
from ..exceptions import OptimizationError
from ..population import Ensemble
from .collections import MutableAgent
from .utils import generate_bounds

if TYPE_CHECKING:
    from typing import Any as Linkage  # accepts legacy/sim Linkage and Mechanism


def sequential_variator(
    center: Sequence[float] | NDArray[np.floating],
    divisions: int,
    bounds: tuple[Sequence[float], Sequence[float]],
) -> Generator[NDArray[np.floating], None, None]:
    """Return an iterable of each possible variation for the elements.

    Number of variations: ((max_dim - 1 / min_dim) / delta_dim) ** len(ite).

    Because linkage is not tolerant to violent changes, the order of output
    for the coefficients is very important.

    The coefficient is in order: middle → min (step 2), min → middle (step 2),
    middle → max (step 1), so that there is no huge variation.

    :param center: Elements that should vary.
    :param divisions: Number of subdivisions between `bounds`.
    :param bounds: 2-uple of minimal then maximal bounds.
    :returns: An iterable of all the dimension combinations.
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


def fast_variator(
    divisions: int,
    bounds: tuple[Sequence[float], Sequence[float]],
) -> Generator[list[float], None, None]:
    """Return an iterable of elements' all possibles variations.

    Number of variations: ((max_dim - 1 / min_dim) / delta_dim) ** len(ite).

    Here the order in the variations is not important.

    :param divisions: Number of subdivisions between `bounds`.
    :param bounds: 2-uple of minimal then maximal bounds.
    :returns: An iterable of all the dimension combinations.
    """
    lists = (
        iter(np.linspace(low, high, divisions))
        for low, high in zip(bounds[0], bounds[1], strict=False)
    )
    for j in itertools.product(*lists):
        yield list(j)


def trials_and_errors_optimization(
    eval_func: "Callable[[Linkage, Sequence[float], JointPositions], float]",
    linkage: "Linkage",
    parameters: Sequence[float] | None = None,
    n_results: int = 10,
    divisions: int = 5,
    **kwargs: Any,
) -> Ensemble:
    """Return the best dimensions optimizing eval_func as an Ensemble.

    Each dimension set has a score. The returned Ensemble contains up to
    *n_results* members with the best scores (maximization by default).

    :param eval_func: Evaluation function.
        Input: (linkage, num_constraints, initial_coordinates).
        Output: score (float).
    :param linkage: Linkage to evaluate.
    :param parameters: Parameters that will be modified. Geometric constraints.
                       If not, it will be assigned tuple(linkage.get_constraints()).
                       The default is None.
    :param n_results: Number of the best candidates to return. The default is 10.
    :param divisions: Number of subdivisions between bounds. The default is 5.
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

    :returns: Ensemble with up to *n_results* members.

    :raises OptimizationError: If parameters are invalid or no valid solution is found.
    """
    if n_results <= 0:
        raise OptimizationError(f"Number of results must be positive, got {n_results}")
    if divisions <= 0:
        raise OptimizationError(f"Number of divisions must be positive, got {divisions}")
    center: NDArray[np.floating] = (
        np.array(linkage.get_constraints()) if parameters is None else np.array(parameters)
    )
    if "bounds" not in kwargs or kwargs["bounds"] is None:
        bounds = generate_bounds(center)
    else:
        bounds = kwargs["bounds"]

    # Results to output: scores, dimensions and initial positions
    # scores will be in decreasing order
    results: list[MutableAgent] = [MutableAgent() for _ in range(n_results)]
    prev: list[tuple[float | None, float | None]] = list(linkage.get_coords())
    # We start by a "fall": we do not want to break the system by modifying
    # dimensions, so we assess it is normally behaving, and we change
    # dimensions progressively to minimal dimensions.
    postfix: dict[str, Any] = {
        "best score": None,
        "best dimensions": [],
    }
    if "sequential" in kwargs and kwargs["sequential"]:
        variations_generator: Generator[Any, None, None] = sequential_variator(
            center,
            divisions,
            bounds,  # type: ignore[arg-type]
        )
    else:
        variations_generator = fast_variator(divisions, bounds)  # type: ignore[arg-type]
    order_relation: Callable[[float, float], float] = kwargs.get("order_relation", max)
    verbose: bool = "verbose" in kwargs and kwargs["verbose"]
    # Iterable of all possible dimensions
    pbar = tqdm.tqdm(
        variations_generator,
        desc="Trials and errors optimization",
        total=divisions ** len(center),
        postfix=postfix,
        disable=not verbose,
    )
    for dim in pbar:
        # Check performances
        new_score = eval_func(linkage, dim, prev)
        for result in results:
            if result.score is None or order_relation(result.score, new_score) != result.score:
                result[:] = new_score, list(dim).copy(), prev.copy()
                if verbose:
                    postfix.update(
                        {
                            "best score": results[0][0],
                            "best dimensions": results[0][1],
                        }
                    )
                    pbar.set_postfix(postfix)
                break
    if results[0].score is None:
        raise OptimizationError(
            "Optimization failed: no valid solutions found. "
            "All evaluated configurations resulted in errors."
        )
    if verbose:
        print(
            "Trials and errors optimization finished. "
            f"Best score: {results[0][0]}, best dimensions: {results[0][1]}"
        )

    # Filter out empty results and convert to Ensemble
    valid = [r for r in results if r.score is not None]
    dims_arr = np.array(
        [list(r.dimensions) if r.dimensions is not None else [] for r in valid],
        dtype=np.float64,
    )
    scores_arr = np.array([r.score for r in valid], dtype=np.float64)
    pos_arr = np.array(
        [
            [
                (x if x is not None else 0.0, y if y is not None else 0.0)
                for x, y in (r.initial_positions or [])
            ]
            for r in valid
        ],
        dtype=np.float64,
    )
    return Ensemble(
        linkage=linkage,
        dimensions=dims_arr,
        initial_positions=pos_arr,
        scores={"score": scores_arr},
    )
