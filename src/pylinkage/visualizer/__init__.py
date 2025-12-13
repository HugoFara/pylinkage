"""
Linkage visualization features.

Backends:
    - matplotlib (default): plot_static_linkage, plot_kinematic_linkage, show_linkage
    - plotly: plot_linkage_plotly, animate_linkage_plotly (interactive HTML)
    - drawsvg: plot_linkage_svg, save_linkage_svg (publication-quality SVG)
"""

__all__ = [
    # Matplotlib backend (default)
    "COLOR_SWITCHER",
    "animate_dashboard",
    "animate_parallel_coordinates",
    "dashboard_layout",
    "parallel_coordinates_plot",
    "plot_kinematic_linkage",
    "plot_static_linkage",
    "show_linkage",
    "swarm_tiled_repr",
    # Plotly backend (interactive)
    "animate_linkage_plotly",
    "plot_linkage_plotly",
    # drawsvg backend (publication SVG)
    "plot_linkage_svg",
    "save_linkage_svg",
    # Symbol definitions
    "LINK_COLORS",
    "SYMBOL_SPECS",
    "LinkStyle",
    "SymbolType",
]

# Matplotlib backend
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

# drawsvg backend
from .drawsvg_viz import plot_linkage_svg as plot_linkage_svg
from .drawsvg_viz import save_linkage_svg as save_linkage_svg

# Plotly backend
from .plotly_viz import animate_linkage_plotly as animate_linkage_plotly
from .plotly_viz import plot_linkage_plotly as plot_linkage_plotly
from .pso_plots import animate_dashboard as animate_dashboard
from .pso_plots import animate_parallel_coordinates as animate_parallel_coordinates
from .pso_plots import dashboard_layout as dashboard_layout
from .pso_plots import parallel_coordinates_plot as parallel_coordinates_plot
from .static import plot_static_linkage as plot_static_linkage

# Symbol definitions
from .symbols import LINK_COLORS as LINK_COLORS
from .symbols import SYMBOL_SPECS as SYMBOL_SPECS
from .symbols import LinkStyle as LinkStyle
from .symbols import SymbolType as SymbolType

