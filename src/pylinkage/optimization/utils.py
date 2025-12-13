"""
This utility module provides various useful functions for optimization.

Created on Mon Jul 12 00:00:01 2021.

@author: HugoFara
"""

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .._types import JointPositions
from ..exceptions import OptimizationError
from ..linkage.analysis import kinematic_default_test

if TYPE_CHECKING:
    from ..linkage.linkage import Linkage


def generate_bounds(
    center: Iterable[float],
    min_ratio: float = 5,
    max_factor: float = 5,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Simple function to generate bounds from a linkage.

    :param center: 1-D sequence, often in the form of ``linkage.get_num_constraints()``.
    :param min_ratio: Minimal compression ratio for the bounds. Minimal bounds will be of the
        shape center[x] / min_ratio.
        (Default value = 5).
    :param max_factor: Dilation factor for the upper bounds. Maximal bounds will be of the
        shape center[x] * max_factor.
        (Default value = 5).

    :raises OptimizationError: If min_ratio or max_factor are not positive.
    """
    if min_ratio <= 0:
        raise OptimizationError(f"min_ratio must be positive, got {min_ratio}")
    if max_factor <= 0:
        raise OptimizationError(f"max_factor must be positive, got {max_factor}")
    np_center = np.array(center)
    return np_center / min_ratio, np_center * max_factor


def kinematic_maximization(
    func: Callable[..., float],
) -> "Callable[[Linkage, Iterable[float], JointPositions | None], float]":
    """Standard run for any linkage before a complete fitness evaluation.

    This decorator makes a kinematic simulation, before passing the loci to the
    decorated function. In case of error, the penalty value is -float('inf').

    :param func: Fitness function to be decorated.
    """
    return kinematic_default_test(func, -float('inf'))


def kinematic_minimization(
    func: Callable[..., float],
) -> "Callable[[Linkage, Iterable[float], JointPositions | None], float]":
    """Standard run for any linkage before a complete fitness evaluation.

    This decorator makes a kinematic simulation, before passing the loci to the
    decorated function. In case of error, the penalty value is float('inf').

    :param func: Fitness function to be decorated.
    """
    return kinematic_default_test(func, float('inf'))
