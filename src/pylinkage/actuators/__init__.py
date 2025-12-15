"""Actuators - motor-driven input drivers for linkage mechanisms.

This module provides actuator classes that provide input motion to mechanisms:

Classes:
    Crank: Motor-driven rotary input (rotating around a ground point)
    LinearActuator: Motor-driven linear input (oscillating piston/cylinder)

Actuators are input drivers that actively move during simulation, as opposed
to passive dyads which react to their parent positions.

Example:
    Create a crank that rotates around a ground point::

        from pylinkage.components import Ground
        from pylinkage.actuators import Crank

        O1 = Ground(0.0, 0.0, name="O1")
        crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1)

    Create a linear actuator::

        from pylinkage.components import Ground
        from pylinkage.actuators import LinearActuator

        O1 = Ground(0.0, 0.0, name="O1")
        actuator = LinearActuator(anchor=O1, angle=0.0, stroke=2.0, velocity=0.1)
"""

from .crank import Crank as Crank
from .linear import LinearActuator as LinearActuator

__all__ = [
    "Crank",
    "LinearActuator",
]
