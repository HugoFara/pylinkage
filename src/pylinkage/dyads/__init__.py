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

Deprecated re-exports:
    This module used to re-export ``Ground``, ``PointTracker``, ``Crank``,
    ``ArcCrank``, ``LinearActuator``, ``Linkage``, ``Component`` and
    ``ConnectedComponent`` from their home modules. Those names still resolve
    here but warn; import them from ``pylinkage.components``,
    ``pylinkage.actuators`` and ``pylinkage.simulation``. See the
    Deprecations page.
"""

# Primary exports are the true Assur groups defined in this package. Names
# from sibling packages are served as deprecated aliases by __getattr__ below,
# so they are imported under private names to keep them out of the namespace.
from .._deprecation import DeprecatedAlias, deprecated_getattr
from ..actuators import ArcCrank as _ArcCrank
from ..actuators import Crank as _Crank
from ..actuators import LinearActuator as _LinearActuator
from ..components import _ALIAS_REASON
from ..components import Component as _Component
from ..components import ConnectedComponent as _ConnectedComponent

# Ground and sensors from components
from ..components import Ground as _Ground
from ..components import PointTracker as _PointTracker
from ..components import _AnchorProxy as _AnchorProxy

# Linkage container
from ..simulation import Linkage as _Linkage
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
    # Deprecated aliases, served by __getattr__ below.
    "Ground",
    "PointTracker",
    "Crank",
    "ArcCrank",
    "LinearActuator",
    "Linkage",
    "Component",
    "ConnectedComponent",
    "Dyad",
    "ConnectedDyad",
]

_REEXPORT_REASON = (
    "pylinkage.dyads only defines dyads; each of these names has a single home "
    "module, and importing it from there is the documented path."
)


_DEPRECATED = {
    "Ground": DeprecatedAlias(_Ground, "pylinkage.components.Ground", "2.0.0", _REEXPORT_REASON),
    "PointTracker": DeprecatedAlias(
        _PointTracker, "pylinkage.components.PointTracker", "2.0.0", _REEXPORT_REASON
    ),
    "Component": DeprecatedAlias(
        _Component, "pylinkage.components.Component", "2.0.0", _REEXPORT_REASON
    ),
    "ConnectedComponent": DeprecatedAlias(
        _ConnectedComponent, "pylinkage.components.ConnectedComponent", "2.0.0", _REEXPORT_REASON
    ),
    "Crank": DeprecatedAlias(_Crank, "pylinkage.actuators.Crank", "2.0.0", _REEXPORT_REASON),
    "ArcCrank": DeprecatedAlias(
        _ArcCrank, "pylinkage.actuators.ArcCrank", "2.0.0", _REEXPORT_REASON
    ),
    "LinearActuator": DeprecatedAlias(
        _LinearActuator, "pylinkage.actuators.LinearActuator", "2.0.0", _REEXPORT_REASON
    ),
    "Linkage": DeprecatedAlias(_Linkage, "pylinkage.simulation.Linkage", "2.0.0", _REEXPORT_REASON),
    "Dyad": DeprecatedAlias(
        value=_Component,
        replacement="pylinkage.components.Component",
        removed_in="2.0.0",
        reason=_ALIAS_REASON,
    ),
    "ConnectedDyad": DeprecatedAlias(
        value=_ConnectedComponent,
        replacement="pylinkage.components.ConnectedComponent",
        removed_in="2.0.0",
        reason=_ALIAS_REASON,
    ),
}

__getattr__ = deprecated_getattr(__name__, _DEPRECATED)
