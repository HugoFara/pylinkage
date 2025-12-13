"""Type definitions for the hypergraph module.

This module provides type aliases and enums used throughout the hypergraph
representation of planar linkages.
"""


from enum import Enum, auto
from typing import TypeAlias

# Type aliases for identifiers
NodeId: TypeAlias = str
EdgeId: TypeAlias = str
HyperedgeId: TypeAlias = str
ComponentId: TypeAlias = str
PortId: TypeAlias = str


class JointType(Enum):
    """Kinematic joint type for graph nodes.

    Attributes:
        REVOLUTE: Pin joint allowing rotation (R).
        PRISMATIC: Slider joint allowing translation (P).
    """

    REVOLUTE = auto()
    PRISMATIC = auto()

    def __str__(self) -> str:
        """Return single-letter representation."""
        return self.name[0]


class NodeRole(Enum):
    """Role of a node in the linkage mechanism.

    Attributes:
        GROUND: Fixed frame point that does not move.
        DRIVER: Input/motor joint that provides motion.
        DRIVEN: Position computed from constraints (part of Assur groups).
    """

    GROUND = auto()
    DRIVER = auto()
    DRIVEN = auto()
