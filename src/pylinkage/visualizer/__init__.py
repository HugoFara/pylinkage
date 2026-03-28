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

import importlib as _importlib

# Mapping from public name to (module, attribute) within this package.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Matplotlib backend
    "plot_kinematic_linkage": (".animated", "plot_kinematic_linkage"),
    "show_linkage": (".animated", "show_linkage"),
    "swarm_tiled_repr": (".animated", "swarm_tiled_repr"),
    "COLOR_SWITCHER": (".core", "COLOR_SWITCHER"),
    "plot_static_linkage": (".static", "plot_static_linkage"),
    # PSO plots (matplotlib)
    "animate_dashboard": (".pso_plots", "animate_dashboard"),
    "animate_parallel_coordinates": (".pso_plots", "animate_parallel_coordinates"),
    "dashboard_layout": (".pso_plots", "dashboard_layout"),
    "parallel_coordinates_plot": (".pso_plots", "parallel_coordinates_plot"),
    # Kinematics visualization
    "animate_kinematics": (".kinematics", "animate_kinematics"),
    "plot_acceleration_vectors": (".kinematics", "plot_acceleration_vectors"),
    "plot_kinematics_frame": (".kinematics", "plot_kinematics_frame"),
    "plot_velocity_vectors": (".kinematics", "plot_velocity_vectors"),
    "show_kinematics": (".kinematics", "show_kinematics"),
    # Plotly backend
    "animate_linkage_plotly": (".plotly_viz", "animate_linkage_plotly"),
    "plot_linkage_plotly": (".plotly_viz", "plot_linkage_plotly"),
    "plot_linkage_plotly_with_velocity": (".plotly_viz", "plot_linkage_plotly_with_velocity"),
    # drawsvg backend
    "plot_linkage_svg": (".drawsvg_viz", "plot_linkage_svg"),
    "plot_linkage_svg_with_velocity": (".drawsvg_viz", "plot_linkage_svg_with_velocity"),
    "save_linkage_svg": (".drawsvg_viz", "save_linkage_svg"),
    "save_linkage_svg_with_velocity": (".drawsvg_viz", "save_linkage_svg_with_velocity"),
    # DXF export
    "plot_linkage_dxf": (".dxf_export", "plot_linkage_dxf"),
    "save_linkage_dxf": (".dxf_export", "save_linkage_dxf"),
    # STEP export
    "build_linkage_3d": (".step_export", "build_linkage_3d"),
    "save_linkage_step": (".step_export", "save_linkage_step"),
    "LinkProfile": (".step_export", "LinkProfile"),
    "JointProfile": (".step_export", "JointProfile"),
    # Symbol definitions
    "LINK_COLORS": (".symbols", "LINK_COLORS"),
    "SYMBOL_SPECS": (".symbols", "SYMBOL_SPECS"),
    "LinkStyle": (".symbols", "LinkStyle"),
    "SymbolType": (".symbols", "SymbolType"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        mod = _importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
