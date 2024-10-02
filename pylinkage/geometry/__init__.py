"""
Basic geometry package.
"""

from .core import (
    sqr_dist,
    norm,
    cyl_to_cart,
    get_nearest_point,
)
from .secants import (
    circle_intersect,
    circle_line_intersection,
    circle_line_from_points_intersection,
    intersection,
)
# For compatibility only, geometry.core.dist is deprecated
from math import dist
