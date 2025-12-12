"""
Core features for visualization.
"""
from ..joints import (
    Crank,
    Fixed,
    Linear,
    Revolute,
    Static,
)
from ..joints.revolute import Pivot

# Colors to use for plotting
COLOR_SWITCHER = {
    Static: 'k',
    Crank: 'g',
    Fixed: 'r',
    Pivot: 'b',
    Revolute: 'b',
    Linear: 'orange'
}


def _get_color(joint):
    """Search in COLOR_SWITCHER for the corresponding color.

    :param joint:

    """
    for joint_type, color in COLOR_SWITCHER.items():
        if isinstance(joint, joint_type):
            return color
    return ''

