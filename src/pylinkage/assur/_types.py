"""Type definitions for the Assur group module."""

from __future__ import annotations

import sys
from enum import Enum, auto

# TypeAlias is available in typing from Python 3.10+
# For 3.9 compatibility, we use typing_extensions
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

# Type aliases for node and edge identifiers
NodeId: TypeAlias = str
EdgeId: TypeAlias = str


class JointType(Enum):
    """Kinematic joint type for graph nodes.

    In planar mechanisms, joints are classified as:
    - REVOLUTE (R): Allows rotation only (pin joint)
    - PRISMATIC (P): Allows translation only (slider joint)
    """

    REVOLUTE = auto()
    PRISMATIC = auto()

    def __str__(self) -> str:
        """Return single-letter representation (R or P)."""
        return self.name[0]


class NodeRole(Enum):
    """Role of a node in the linkage mechanism.

    - GROUND: Fixed frame point (does not move)
    - DRIVER: Input/motor joint that provides motion
    - DRIVEN: Part of an Assur group, position computed from constraints
    """

    GROUND = auto()
    DRIVER = auto()
    DRIVEN = auto()
