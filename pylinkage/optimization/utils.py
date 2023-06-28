"""Utility for optimization."""
import numpy as np


def generate_bounds(center, min_ratio=5, max_factor=5):
    """Simple function to generate bounds from a linkage.

    :param center: 1-D sequence, often in the form of ``linkage.get_num_constraints()``.
    :type center: Iterable
    :param min_ratio: Minimal compression ratio for the bounds. Minimal bounds will be of the
        shape center[x] / min_ratio.
        (Default value = 5)
    :type min_ratio: float
    :param max_factor: Dilation factor for the upper bounds. Maximal bounds will be of the
        shape center[x] * max_factor.
        (Default value = 5)
    :type max_factor: float

    
    """
    np_center = np.array(center)
    return np_center / min_ratio, np_center * max_factor
