"""Components - base classes and fixed frame elements.

This module provides the foundational classes for building kinematic mechanisms:

Classes:
    Component: Abstract base class for all kinematic elements
    ConnectedComponent: Base for elements with parent connections
    Ground: Fixed point on the frame (ground link)
    PointTracker: Sensor for tracking positions on links
    _AnchorProxy: Proxy for output position access (internal)

The Component class serves as the base for all user-facing kinematic
building blocks including:
- Ground points (this module)
- Actuators (see pylinkage.actuators)
- Assur dyads (see pylinkage.dyads)
"""

from .._deprecation import DeprecatedAlias, deprecated_getattr
from ._base import Component as Component
from ._base import ConnectedComponent as ConnectedComponent
from ._base import _AnchorProxy as _AnchorProxy
from .ground import Ground as Ground
from .point_tracker import PointTracker as PointTracker

__all__ = [
    "Component",
    "ConnectedComponent",
    "Ground",
    "PointTracker",
    "_AnchorProxy",
    # Deprecated aliases, served by __getattr__ below.
    "Dyad",
    "ConnectedDyad",
]


# ``Dyad`` and ``ConnectedDyad`` never denoted dyads: they are plain aliases of
# ``Component`` and ``ConnectedComponent``, and the name collides with two real
# dyad classes elsewhere in the package.
_ALIAS_REASON = (
    "These are plain aliases of the component base classes rather than dyads, "
    "and the name collides with pylinkage.assur.Dyad and "
    "pylinkage.synthesis.BurmesterDyad, which are genuine dyads."
)

_DEPRECATED = {
    "Dyad": DeprecatedAlias(
        value=Component,
        replacement="pylinkage.components.Component",
        removed_in="2.0.0",
        reason=_ALIAS_REASON,
    ),
    "ConnectedDyad": DeprecatedAlias(
        value=ConnectedComponent,
        replacement="pylinkage.components.ConnectedComponent",
        removed_in="2.0.0",
        reason=_ALIAS_REASON,
    ),
}

__getattr__ = deprecated_getattr(__name__, _DEPRECATED)
