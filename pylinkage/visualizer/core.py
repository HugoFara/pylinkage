"""
Core features for visualization.
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

# Colors to use for plotting
COLOR_SWITCHER: dict[type[Joint], str] = {
    Static: 'k',
    Crank: 'g',
    Fixed: 'r',
    Pivot: 'b',
    Revolute: 'b',
    Linear: 'orange'
}


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

