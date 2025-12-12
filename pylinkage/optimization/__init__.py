"""Optimization package."""
from . import collections as collections
from .grid_search import (
    trials_and_errors_optimization as trials_and_errors_optimization,
)
from .particle_swarm import (
    particle_swarm_optimization as particle_swarm_optimization,
)
from .utils import (
    generate_bounds as generate_bounds,
)
from .utils import (
    kinematic_maximization as kinematic_maximization,
)
from .utils import (
    kinematic_minimization as kinematic_minimization,
)
