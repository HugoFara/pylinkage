"""
Core features for visualization.

This module provides shared utilities for matplotlib-based visualization.
For symbol definitions used by other backends (Plotly, drawsvg), see symbols.py.
"""

from ..joints.crank import Crank
from ..joints.fixed import Fixed
from ..joints.joint import Joint
from ..joints.joint import _StaticBase as Static
from ..joints.prismatic import Prismatic
from ..joints.revolute import Pivot, Revolute

# Colors to use for matplotlib plotting (backwards compatible)
COLOR_SWITCHER: dict[type[Joint], str] = {
    Static: "k",
    Crank: "g",
    Fixed: "r",
    Pivot: "b",
    Revolute: "b",
    Prismatic: "orange",
}

# Re-export symbol utilities for convenience


def _get_color(joint: Joint) -> str:
    """Search in COLOR_SWITCHER for the corresponding color.

    Args:
        joint: The joint to get the color for.

    Returns:
        The color string for matplotlib.
    """
    for joint_type, color in COLOR_SWITCHER.items():
        if isinstance(joint, joint_type):
            return color
    return ""
