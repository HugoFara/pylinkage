"""Population abstractions for working with collections of mechanisms.

- :class:`Member`: A single mechanism variant (dimensions + scores + trajectory).
- :class:`Ensemble`: N parameter variants of one topology (fast batch simulation).
- :class:`Population`: Heterogeneous collection of mechanisms (different topologies).
"""

__all__ = ["Ensemble", "Member", "Population"]

from ._ensemble import Ensemble as Ensemble
from ._member import Member as Member
from ._population import Population as Population
