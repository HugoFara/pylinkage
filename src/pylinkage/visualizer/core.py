"""
Core features for visualization.

This module provides shared utilities for matplotlib-based visualization.
For symbol definitions used by other backends (Plotly, drawsvg), see symbols.py.
"""

from __future__ import annotations

from ..joints import (
    Crank,
    Fixed,
    Linear,
    Revolute,
    Static,
)
from ..joints.joint import Joint
from ..joints.revolute import Pivot

# Colors to use for matplotlib plotting (backwards compatible)
COLOR_SWITCHER: dict[type[Joint], str] = {
    Static: 'k',
    Crank: 'g',
    Fixed: 'r',
    Pivot: 'b',
    Revolute: 'b',
    Linear: 'orange'
}

# Re-export symbol utilities for convenience
from .symbols import (  # noqa: E402
    LINK_COLORS,
    SYMBOL_SPECS,
    LinkStyle,
    SymbolType,
    get_link_color,
    get_symbol_spec,
)


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
    return ''

