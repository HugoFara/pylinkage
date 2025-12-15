"""Legacy joint definitions (Assur groups).

.. deprecated:: 0.8.0
    This module uses misleading terminology. What are called "joints" here
    are actually Assur groups (combinations of joints and links).

    Migration guide:
    - ``Static`` -> ``mechanism.GroundJoint``
    - ``Crank`` -> ``mechanism.create_crank()`` returns ``(DriverLink, RevoluteJoint)``
    - ``Revolute`` -> ``mechanism.create_rrr_dyad()`` returns ``(Link, Link, RevoluteJoint)``
    - ``Prismatic`` -> ``mechanism.create_rrp_dyad()`` returns ``(Link, Link, PrismaticJoint)``
    - ``Fixed`` -> ``mechanism.create_fixed_dyad()``

    For Assur group decomposition, use ``pylinkage.assur`` module directly.

    Example migration::

        # Old (deprecated):
        from pylinkage.joints import Static, Crank, Revolute
        A = Static(x=0, y=0, name="A")
        B = Crank(x=1, y=0, joint0=A, distance=1, angle=0.1)

        # New (preferred):
        from pylinkage.mechanism import (
            GroundJoint, GroundLink, Mechanism,
            create_crank, create_rrr_dyad,
        )
        A = GroundJoint("A", position=(0, 0))
        ground = GroundLink("ground", joints=[A])
        crank, B = create_crank(A, radius=1.0, angular_velocity=0.1)
"""

import warnings

__all__ = ["Crank", "Fixed", "Linear", "Prismatic", "Revolute", "Static"]

# Emit deprecation warning on import
warnings.warn(
    "The pylinkage.joints module is deprecated. "
    "Use pylinkage.mechanism for the new Joint/Link API, "
    "or pylinkage.assur for Assur group decomposition. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

from .crank import Crank as Crank
from .fixed import Fixed as Fixed
from .joint import Static as Static
from .prismatic import Linear as Linear  # Deprecated alias
from .prismatic import Prismatic as Prismatic
from .revolute import Revolute as Revolute
