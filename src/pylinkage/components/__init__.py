"""Components - base classes and fixed frame elements.

This module provides the foundational classes for building kinematic mechanisms:

Classes:
    Component: Abstract base class for all kinematic elements
    ConnectedComponent: Base for elements with parent connections
    Ground: Fixed point on the frame (ground link)
    PointTracker: Sensor for tracking positions on links
    _AnchorProxy: Proxy for output position access (internal)

The Component class (and its alias Dyad) serves as the base for all
user-facing kinematic building blocks including:
- Ground points (this module)
- Actuators (see pylinkage.actuators)
- Assur dyads (see pylinkage.dyads)
"""

from ._base import Component as Component
from ._base import ConnectedComponent as ConnectedComponent

# Backwards compatibility aliases
from ._base import ConnectedDyad as ConnectedDyad
from ._base import Dyad as Dyad
from ._base import _AnchorProxy as _AnchorProxy
from .ground import Ground as Ground
from .point_tracker import PointTracker as PointTracker

__all__ = [
    "Component",
    "ConnectedComponent",
    "Ground",
    "PointTracker",
    "_AnchorProxy",
    # Backwards compatibility
    "Dyad",
    "ConnectedDyad",
]
