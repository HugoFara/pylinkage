#!/usr/bin/env python3
"""
PyLinkage is a module to create, optimize and visualize linkages.

Please see the documentation at https://hugofara.github.io/pylinkage/.
A copy of the documentation should have been distributed on your system in the
docs/ folder.

Created on Thu Jun 10 21:30:52 2021

@author: HugoFara
"""

# For compatibility only, geometry.dist is deprecated
from math import dist

from .exceptions import (
    HypostaticError,
    NotCompletelyDefinedError,
    UnbuildableError,
)
from .geometry import circle_intersect, cyl_to_cart, intersection, norm, sqr_dist
from .joints import (
    Crank,
    Fixed,
    Linear,
    Revolute,
    Static,
)
from .joints.revolute import Pivot
from .linkage import (
    Linkage,
    bounding_box,
    kinematic_default_test,
)
from .optimization import (
    collections,
    generate_bounds,
    kinematic_maximization,
    kinematic_minimization,
    particle_swarm_optimization,
    trials_and_errors_optimization,
)
from .visualizer import (
    plot_kinematic_linkage,
    plot_static_linkage,
    show_linkage,
    swarm_tiled_repr,
)

__version__ = "0.6.0"
