#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyLinkage is a module to create, optimize and visualize linkages.

Please see the documentation at https://hugofara.github.io/pylinkage/.
A copy of the documentation should have been distributed on your system in the
docs/ folder.

Created on Thu Jun 10 21:30:52 2021

@author: HugoFara
"""

from .geometry import (
    sqr_dist,
    norm,
    cyl_to_cart,
    circle_intersect,
    intersection
)
from .exceptions import (
    UnbuildableError,
    HypostaticError,
    NotCompletelyDefinedError,
)
from .linkage import (
    Linkage,
    kinematic_default_test,
    bounding_box,
)
from .joints import (
    Crank,
    Fixed,
    Linear,
    Revolute,
    Static,
)
from .joints.revolute import Pivot
from .optimization import (
    generate_bounds,
    trials_and_errors_optimization,
    particle_swarm_optimization,
    kinematic_maximization,
    kinematic_minimization,
    collections,
)
from .visualizer import (
    plot_static_linkage,
    plot_kinematic_linkage,
    show_linkage,
    swarm_tiled_repr,
)
# For compatibility only, geometry.dist is deprecated
from math import dist

__version__ = "0.6.0"
