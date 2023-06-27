"""Utility for optimization."""
import numpy as np


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
