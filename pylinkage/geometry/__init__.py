"""
Basic geometry package.
"""

# For compatibility only, geometry.core.dist is deprecated
from math import dist

from .core import (
    cyl_to_cart,
    get_nearest_point,
    norm,
    sqr_dist,
)
from .secants import (
    circle_intersect,
    circle_line_from_points_intersection,
    circle_line_intersection,
    intersection,
)
