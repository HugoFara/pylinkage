"""
Linkage visualization features.

Backends:
    - matplotlib (default): plot_static_linkage, plot_kinematic_linkage, show_linkage
    - plotly: plot_linkage_plotly, animate_linkage_plotly (interactive HTML)
    - drawsvg: plot_linkage_svg, save_linkage_svg (publication-quality SVG)
    - dxf: plot_linkage_dxf, save_linkage_dxf (CAD/CNC export)
    - step: build_linkage_3d, save_linkage_step (3D CAD interchange)
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
    # Kinematics visualization
    "animate_kinematics",
    "plot_acceleration_vectors",
    "plot_kinematics_frame",
    "plot_velocity_vectors",
    "show_kinematics",
    # Plotly backend (interactive)
    "animate_linkage_plotly",
    "plot_linkage_plotly",
    "plot_linkage_plotly_with_velocity",
    # drawsvg backend (publication SVG)
    "plot_linkage_svg",
    "plot_linkage_svg_with_velocity",
    "save_linkage_svg",
    "save_linkage_svg_with_velocity",
    # DXF export (CAD/CNC)
    "plot_linkage_dxf",
    "save_linkage_dxf",
    # STEP export (3D CAD)
    "build_linkage_3d",
    "save_linkage_step",
    "LinkProfile",
    "JointProfile",
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
from .drawsvg_viz import (
    plot_linkage_svg_with_velocity as plot_linkage_svg_with_velocity,
)
from .drawsvg_viz import save_linkage_svg as save_linkage_svg
from .drawsvg_viz import (
    save_linkage_svg_with_velocity as save_linkage_svg_with_velocity,
)

# DXF export (lazy import to avoid loading ezdxf when not needed)
from .dxf_export import plot_linkage_dxf as plot_linkage_dxf
from .dxf_export import save_linkage_dxf as save_linkage_dxf

# Kinematics visualization
from .kinematics import animate_kinematics as animate_kinematics
from .kinematics import plot_acceleration_vectors as plot_acceleration_vectors
from .kinematics import plot_kinematics_frame as plot_kinematics_frame
from .kinematics import plot_velocity_vectors as plot_velocity_vectors
from .kinematics import show_kinematics as show_kinematics

# Plotly backend
from .plotly_viz import animate_linkage_plotly as animate_linkage_plotly
from .plotly_viz import plot_linkage_plotly as plot_linkage_plotly
from .plotly_viz import (
    plot_linkage_plotly_with_velocity as plot_linkage_plotly_with_velocity,
)
from .pso_plots import animate_dashboard as animate_dashboard
from .pso_plots import animate_parallel_coordinates as animate_parallel_coordinates
from .pso_plots import dashboard_layout as dashboard_layout
from .pso_plots import parallel_coordinates_plot as parallel_coordinates_plot
from .static import plot_static_linkage as plot_static_linkage

# STEP export (lazy import to avoid loading build123d when not needed)
from .step_export import JointProfile as JointProfile
from .step_export import LinkProfile as LinkProfile
from .step_export import build_linkage_3d as build_linkage_3d
from .step_export import save_linkage_step as save_linkage_step

# Symbol definitions
from .symbols import LINK_COLORS as LINK_COLORS
from .symbols import SYMBOL_SPECS as SYMBOL_SPECS
from .symbols import LinkStyle as LinkStyle
from .symbols import SymbolType as SymbolType

