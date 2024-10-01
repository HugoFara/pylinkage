"""Optimization package."""
from .grid_search import trials_and_errors_optimization
from .particle_swarm import particle_swarm_optimization
from .utils import (
    generate_bounds,
    kinematic_default_test,
    kinematic_maximization,
    kinematic_minimization,
    bounding_box,
)