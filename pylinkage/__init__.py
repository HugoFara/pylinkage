#!/usr/bin/env python3
"""
PyLinkage is a module to create, optimize and visualize linkages.

Please see the documentation at https://hugofara.github.io/pylinkage/.
A copy of the documentation should have been distributed on your system in the
docs/ folder.

Created on Thu Jun 10 21:30:52 2021

@author: HugoFara
"""

from .exceptions import (
    HypostaticError as HypostaticError,
    NotCompletelyDefinedError as NotCompletelyDefinedError,
    UnbuildableError as UnbuildableError,
)
from .geometry import (
    circle_intersect as circle_intersect,
    cyl_to_cart as cyl_to_cart,
    intersection as intersection,
    norm as norm,
    sqr_dist as sqr_dist,
)
from .joints import (
    Crank as Crank,
    Fixed as Fixed,
    Linear as Linear,
    Revolute as Revolute,
    Static as Static,
)
from .joints.revolute import Pivot as Pivot
from .linkage import (
    Linkage as Linkage,
    bounding_box as bounding_box,
    kinematic_default_test as kinematic_default_test,
)
from .optimization import (
    collections as collections,
    generate_bounds as generate_bounds,
    kinematic_maximization as kinematic_maximization,
    kinematic_minimization as kinematic_minimization,
    particle_swarm_optimization as particle_swarm_optimization,
    trials_and_errors_optimization as trials_and_errors_optimization,
)
from .visualizer import (
    plot_kinematic_linkage as plot_kinematic_linkage,
    plot_static_linkage as plot_static_linkage,
    show_linkage as show_linkage,
    swarm_tiled_repr as swarm_tiled_repr,
)

__version__ = "0.6.0"
