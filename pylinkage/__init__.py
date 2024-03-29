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
    dist,
    sqr_dist,
    norm,
    cyl_to_cart,
    circle_intersect,
    intersection
)
from .interface import (
    UnbuildableError,
    HypostaticError,
    Static,
    Revolute,
    Pivot,
    Crank,
    Fixed,
    Linkage
)
from .optimization import (
    generate_bounds,
    trials_and_errors_optimization,
    particle_swarm_optimization
)
from .utility import (
    kinematic_default_test,
    kinematic_maximization,
    kinematic_minimization,
    bounding_box
)
from .visualizer import (
    plot_static_linkage,
    plot_kinematic_linkage,
    show_linkage,
    swarm_tiled_repr,
)

__version__ = "0.5.3"
