#!/usr/bin/env python3
"""
PyLinkage is a module to create, optimize and visualize linkages.

Please see the documentation at https://hugofara.github.io/pylinkage/.
A copy of the documentation should have been distributed on your system in the
docs/ folder.

Created on Thu Jun 10 21:30:52 2021

@author: HugoFara
"""

__all__ = [
    # Assur group module
    "assur",
    # Exceptions
    "HypostaticError",  # Deprecated alias for UnderconstrainedError
    "NotCompletelyDefinedError",
    "OptimizationError",
    "UnbuildableError",
    "UnderconstrainedError",
    # Geometry
    "circle_intersect",
    "cyl_to_cart",
    "intersection",
    "norm",
    "sqr_dist",
    # Joints
    "Crank",
    "Fixed",
    "Linear",  # Deprecated alias for Prismatic
    "Prismatic",
    "Revolute",
    "Static",
    "Pivot",
    # Linkage
    "Linkage",
    "Simulation",
    "bounding_box",
    "kinematic_default_test",
    # Optimization
    "collections",
    "generate_bounds",
    "kinematic_maximization",
    "kinematic_minimization",
    "particle_swarm_optimization",
    "trials_and_errors_optimization",
    # Visualizer
    "plot_kinematic_linkage",
    "plot_static_linkage",
    "show_linkage",
    "swarm_tiled_repr",
]

# Assur group module for graph-based linkage representation
from . import assur as assur
from .exceptions import (
    HypostaticError as HypostaticError,  # Deprecated alias
)
from .exceptions import (
    NotCompletelyDefinedError as NotCompletelyDefinedError,
)
from .exceptions import (
    OptimizationError as OptimizationError,
)
from .exceptions import (
    UnbuildableError as UnbuildableError,
)
from .exceptions import (
    UnderconstrainedError as UnderconstrainedError,
)
from .geometry import (
    circle_intersect as circle_intersect,
)
from .geometry import (
    cyl_to_cart as cyl_to_cart,
)
from .geometry import (
    intersection as intersection,
)
from .geometry import (
    norm as norm,
)
from .geometry import (
    sqr_dist as sqr_dist,
)
from .joints import (
    Crank as Crank,
)
from .joints import (
    Fixed as Fixed,
)
from .joints import (
    Linear as Linear,  # Deprecated alias for Prismatic
)
from .joints import (
    Prismatic as Prismatic,
)
from .joints import (
    Revolute as Revolute,
)
from .joints import (
    Static as Static,
)
from .joints.revolute import Pivot as Pivot
from .linkage import (
    Linkage as Linkage,
)
from .linkage import (
    Simulation as Simulation,
)
from .linkage import (
    bounding_box as bounding_box,
)
from .linkage import (
    kinematic_default_test as kinematic_default_test,
)
from .optimization import (
    collections as collections,
)
from .optimization import (
    generate_bounds as generate_bounds,
)
from .optimization import (
    kinematic_maximization as kinematic_maximization,
)
from .optimization import (
    kinematic_minimization as kinematic_minimization,
)
from .optimization import (
    particle_swarm_optimization as particle_swarm_optimization,
)
from .optimization import (
    trials_and_errors_optimization as trials_and_errors_optimization,
)
from .visualizer import (
    plot_kinematic_linkage as plot_kinematic_linkage,
)
from .visualizer import (
    plot_static_linkage as plot_static_linkage,
)
from .visualizer import (
    show_linkage as show_linkage,
)
from .visualizer import (
    swarm_tiled_repr as swarm_tiled_repr,
)

__version__ = "0.7.0"
