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
    # Subpackages. Heavy or optional ones load on first access.
    "actuators",
    "assur",
    "cam",
    "components",
    "dimensions",
    "dyads",
    "exceptions",
    "geometry",
    "hypergraph",
    "linkage",
    "mechanism",
    "optimization",
    "population",
    "simulation",
    "solver",
    "symbolic",
    "synthesis",
    "topology",
    "visualizer",
    # The definition path: frame, actuators, dyads, container
    "Ground",
    "PointTracker",
    "Crank",
    "ArcCrank",
    "LinearActuator",
    "RRRDyad",
    "RRPDyad",
    "PPDyad",
    "FixedDyad",
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
from typing import TYPE_CHECKING

# --- Eager imports (lightweight, always available) ---
from . import actuators as actuators
from . import assur as assur
from . import cam as cam
from . import components as components
from . import dimensions as dimensions
from . import dyads as dyads
from . import exceptions as exceptions
from . import geometry as geometry
from . import hypergraph as hypergraph
from . import linkage as linkage
from . import mechanism as mechanism
from . import simulation as simulation
from . import topology as topology
from ._simulation_context import (
    Simulation as Simulation,
)

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
from .actuators import ArcCrank as ArcCrank
from .actuators import Crank as Crank
from .actuators import LinearActuator as LinearActuator
from .components import Ground as Ground
from .components import PointTracker as PointTracker
from .dyads import FixedDyad as FixedDyad
from .dyads import PPDyad as PPDyad
from .dyads import RRPDyad as RRPDyad
from .dyads import RRRDyad as RRRDyad
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
from .simulation import (
    Linkage as Linkage,
)

# --- Lazy imports (heavy optional dependencies) ---

if TYPE_CHECKING:
    # Eager imports for type checkers only; at runtime these load on first access.
    from . import optimization, population, solver, symbolic, synthesis, visualizer
    from .optimization import (
        collections,
        generate_bounds,
        kinematic_maximization,
        kinematic_minimization,
        particle_swarm_optimization,
        trials_and_errors_optimization,
    )
    from .population import Ensemble, Member, Population
    from .visualizer import (
        plot_kinematic_linkage,
        plot_static_linkage,
        show_linkage,
        swarm_tiled_repr,
    )

_LAZY_SUBMODULES = {
    "optimization",
    "population",
    "solver",
    "symbolic",
    "synthesis",
    "visualizer",
}

_LAZY_ATTRS: dict[str, str] = {
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


__version__ = "1.2.2"
