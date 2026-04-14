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
    # Dyads module (user-facing API for mechanism building)
    "dyads",
    # Mechanism module (new Links + Joints model)
    "mechanism",
    # Symbolic computation module (lazy, requires sympy)
    "symbolic",
    # Synthesis module (lazy, requires scipy)
    "synthesis",
    # Canonical types (from _types.py)
    "JointType",
    "NodeRole",
    "NodeId",
    "EdgeId",
    "HyperedgeId",
    "ComponentId",
    "PortId",
    # Exceptions
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
    # Joints (legacy, lazy)
    "Crank",
    "Fixed",
    "Prismatic",
    "Revolute",
    "Static",
    "Pivot",
    # Linkage
    "Linkage",
    "Simulation",
    "bounding_box",
    "extract_trajectories",
    "extract_trajectory",
    "kinematic_default_test",
    # Optimization (lazy, some require scipy)
    "collections",
    "generate_bounds",
    "kinematic_maximization",
    "kinematic_minimization",
    "particle_swarm_optimization",
    "trials_and_errors_optimization",
    # Population (lazy)
    "Ensemble",
    "Member",
    "Population",
    # Visualizer (lazy, requires matplotlib/plotly/drawsvg)
    "plot_kinematic_linkage",
    "plot_static_linkage",
    "show_linkage",
    "swarm_tiled_repr",
]

import importlib as _importlib

# --- Eager imports (lightweight, always available) ---
# Assur group module for graph-based linkage representation
from . import assur as assur

# Dyads module - user-facing API for mechanism building
from . import dyads as dyads

# Mechanism module - new Links + Joints model
from . import mechanism as mechanism

# Canonical types (single source of truth for kinematic types)
from ._types import (
    ComponentId as ComponentId,
)
from ._types import (
    EdgeId as EdgeId,
)
from ._types import (
    HyperedgeId as HyperedgeId,
)
from ._types import (
    JointType as JointType,
)
from ._types import (
    NodeId as NodeId,
)
from ._types import (
    NodeRole as NodeRole,
)
from ._types import (
    PortId as PortId,
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
    extract_trajectories as extract_trajectories,
)
from .linkage import (
    extract_trajectory as extract_trajectory,
)
from .linkage import (
    kinematic_default_test as kinematic_default_test,
)

# --- Lazy imports (heavy optional dependencies) ---

_LAZY_SUBMODULES = {
    "population",
    "symbolic",
    "synthesis",
}

_LAZY_ATTRS: dict[str, str] = {
    # From .joints (legacy, deprecated)
    "Crank": ".joints",
    "Fixed": ".joints",
    "Prismatic": ".joints",
    "Revolute": ".joints",
    "Static": ".joints",
    "Pivot": ".joints.revolute",
    # From .population
    "Ensemble": ".population",
    "Member": ".population",
    "Population": ".population",
    # From .optimization
    "collections": ".optimization",
    "generate_bounds": ".optimization",
    "kinematic_maximization": ".optimization",
    "kinematic_minimization": ".optimization",
    "particle_swarm_optimization": ".optimization",
    "trials_and_errors_optimization": ".optimization",
    # From .visualizer
    "plot_kinematic_linkage": ".visualizer",
    "plot_static_linkage": ".visualizer",
    "show_linkage": ".visualizer",
    "swarm_tiled_repr": ".visualizer",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_SUBMODULES:
        mod = _importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name in _LAZY_ATTRS:
        mod = _importlib.import_module(_LAZY_ATTRS[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.8.0"
