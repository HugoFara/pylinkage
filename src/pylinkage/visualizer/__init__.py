"""
Linkage visualization features.
"""

__all__ = [
    "COLOR_SWITCHER",
    "animate_dashboard",
    "animate_parallel_coordinates",
    "dashboard_layout",
    "parallel_coordinates_plot",
    "plot_kinematic_linkage",
    "plot_static_linkage",
    "show_linkage",
    "swarm_tiled_repr",
]

from .animated import (
    plot_kinematic_linkage as plot_kinematic_linkage,
)
from .animated import (
    show_linkage as show_linkage,
)
from .animated import (
    swarm_tiled_repr as swarm_tiled_repr,
)
from .core import COLOR_SWITCHER as COLOR_SWITCHER
from .pso_plots import animate_dashboard as animate_dashboard
from .pso_plots import animate_parallel_coordinates as animate_parallel_coordinates
from .pso_plots import dashboard_layout as dashboard_layout
from .pso_plots import parallel_coordinates_plot as parallel_coordinates_plot
from .static import plot_static_linkage as plot_static_linkage

