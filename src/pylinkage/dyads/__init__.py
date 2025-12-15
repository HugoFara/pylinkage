"""Dyads - Assur group building blocks for planar linkages.

This module provides the primary user-facing API for building
planar linkage mechanisms using Assur group formalism.

Classes:
    Ground: Fixed point on the frame (ground link)
    Crank: Motor-driven rotary input
    RRRDyad: Circle-circle intersection (two links meeting at one joint)
    RRPDyad: Circle-line intersection (slider mechanism)
    FixedDyad: Deterministic polar projection
    Linkage: Container orchestrating dyads into a mechanism

Example:
    Build and simulate a four-bar linkage::

        from pylinkage.dyads import Ground, Crank, RRRDyad, Linkage

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

Migration from joints/:
    The deprecated ``pylinkage.joints`` module has been replaced by this module.
    Migration is straightforward:

    - ``Static(x, y)`` -> ``Ground(x, y)``
    - ``Crank(joint0=A, distance=r, angle=v)`` -> ``Crank(anchor=A, radius=r, angular_velocity=v)``
    - ``Revolute(joint0=A, joint1=B, distance0=d0, distance1=d1)`` ->
      ``RRRDyad(anchor1=A, anchor2=B, distance1=d0, distance2=d1)``
    - ``Prismatic(...)`` -> ``RRPDyad(...)``
    - ``Fixed(...)`` -> ``FixedDyad(...)``
"""

from ._base import BinaryDyad as BinaryDyad
from ._base import ConnectedDyad as ConnectedDyad
from ._base import Dyad as Dyad
from .crank import Crank as Crank
from .fixed import FixedDyad as FixedDyad
from .ground import Ground as Ground
from .linkage import Linkage as Linkage
from .rrp import RRPDyad as RRPDyad
from .rrr import RRRDyad as RRRDyad

__all__ = [
    "Dyad",
    "ConnectedDyad",
    "BinaryDyad",
    "Ground",
    "Crank",
    "RRRDyad",
    "RRPDyad",
    "FixedDyad",
    "Linkage",
]
