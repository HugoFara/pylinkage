"""Dyads - Assur group building blocks for planar linkages.

This module provides true Assur group dyads (0 DOF structural units):

Classes:
    RRRDyad: Circle-circle intersection (two links meeting at one joint)
    RRPDyad: Circle-line intersection (slider mechanism)
    PPDyad: Line-line intersection (double slider)
    FixedDyad: Deterministic polar projection
    BinaryDyad: Base class for binary Assur groups
    TranslatingCamFollower: Translating follower driven by cam profile
    OscillatingCamFollower: Oscillating (rocker) follower driven by cam profile

Functions:
    create_dyad: Factory function to create dyads from isomer signatures

For other kinematic elements, use the appropriate modules:
    - pylinkage.components: Ground, base classes (Component, ConnectedComponent)
    - pylinkage.actuators: Crank, LinearActuator
    - pylinkage.cam: CamProfile, FunctionProfile, motion laws
    - pylinkage.simulation: Linkage

Example:
    Build and simulate a four-bar linkage::

        from pylinkage.components import Ground
        from pylinkage.actuators import Crank
        from pylinkage.dyads import RRRDyad
        from pylinkage.simulation import Linkage

        # Ground points
        O1 = Ground(0.0, 0.0, name="O1")
        O2 = Ground(2.0, 0.0, name="O2")

        # Crank (driver)
        crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1)

        # Rocker (RRR dyad)
        rocker = RRRDyad(
            anchor1=crank.output,
            anchor2=O2,
            distance1=2.0,
            distance2=1.5,
        )

        # Build and simulate
        linkage = Linkage([O1, O2, crank, rocker], name="Four-Bar")
        for positions in linkage.step():
            print(positions)

Backwards Compatibility:
    For convenience, this module re-exports items from their canonical locations:
    - Ground: from pylinkage.components
    - Crank, LinearActuator: from pylinkage.actuators
    - Linkage: from pylinkage.simulation
    - Dyad, ConnectedDyad: deprecated aliases, see the Deprecations page
"""

# Primary exports are the true Assur groups defined in this package; the rest
# are re-exports kept for backwards compatibility.
from .._deprecation import DeprecatedAlias, deprecated_getattr
from ..actuators import ArcCrank as ArcCrank
from ..actuators import Crank as Crank
from ..actuators import LinearActuator as LinearActuator
from ..components import _ALIAS_REASON
from ..components import Component as Component
from ..components import ConnectedComponent as ConnectedComponent

# Ground and sensors from components
from ..components import Ground as Ground
from ..components import PointTracker as PointTracker
from ..components import _AnchorProxy as _AnchorProxy

# Linkage container
from ..simulation import Linkage as Linkage
from ._base import BinaryDyad as BinaryDyad
from ._conversion import to_mechanism as to_mechanism
from .factory import create_dyad as create_dyad
from .factory import get_isomer_geometry as get_isomer_geometry
from .factory import get_required_anchors as get_required_anchors
from .factory import get_required_constraints as get_required_constraints
from .fixed import FixedDyad as FixedDyad
from .oscillating_cam import OscillatingCamFollower as OscillatingCamFollower
from .pp import PPDyad as PPDyad
from .rrp import RRPDyad as RRPDyad
from .rrr import RRRDyad as RRRDyad
from .translating_cam import TranslatingCamFollower as TranslatingCamFollower

__all__ = [
    # True Assur groups (primary exports)
    "RRRDyad",
    "RRPDyad",
    "PPDyad",
    "FixedDyad",
    "BinaryDyad",
    # Cam-follower mechanisms
    "TranslatingCamFollower",
    "OscillatingCamFollower",
    # Factory function for creating dyads from isomer signatures
    "create_dyad",
    # Conversion to the low-level Joint/Link mechanism model
    "to_mechanism",
    "get_isomer_geometry",
    "get_required_anchors",
    "get_required_constraints",
    # Re-exports for backwards compatibility
    "Ground",
    "PointTracker",
    "Crank",
    "ArcCrank",
    "LinearActuator",
    "Linkage",
    # Base classes
    "Component",
    "ConnectedComponent",
    "_AnchorProxy",
    # Deprecated aliases, served by __getattr__ below.
    "Dyad",
    "ConnectedDyad",
]


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
