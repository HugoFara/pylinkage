"""Simulation - containers for running mechanism simulations.

This module provides container classes for orchestrating mechanism simulations:

Classes:
    Linkage: Container that manages components and runs step-by-step simulation

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
"""

from .linkage import Linkage as Linkage

__all__ = [
    "Linkage",
]
