"""Mechanism module - proper Links + Joints model for planar linkages.

This module provides the low-level API for defining planar mechanisms
using standard mechanical engineering terminology:

- **Joints** are actual connection points (revolute pins, prismatic sliders)
- **Links** are rigid bodies connecting joints
- **Mechanism** orchestrates joints and links for simulation

For a higher-level API using Assur group building blocks, see ``pylinkage.dyads``.

Basic Usage:
    >>> from pylinkage.mechanism import (
    ...     Mechanism, GroundJoint, GroundLink,
    ...     RevoluteJoint, Link, DriverLink,
    ... )
    >>>
    >>> # Create ground joints
    >>> O1 = GroundJoint("O1", position=(0.0, 0.0))
    >>> O2 = GroundJoint("O2", position=(2.0, 0.0))
    >>> ground = GroundLink("ground", joints=[O1, O2])

Classes:
    Joint: Base class for all joints
    RevoluteJoint: Pin joint (1 DOF rotation)
    PrismaticJoint: Slider joint (1 DOF translation)
    GroundJoint: Fixed revolute joint on the frame

    Link: Rigid body connecting joints
    GroundLink: The stationary frame
    DriverLink: Motor-driven input link

    Mechanism: The main orchestrator class
    MechanismBuilder: Links-first builder for creating mechanisms

Conversion:
    mechanism_from_linkage: Convert legacy Linkage to Mechanism
    mechanism_to_linkage: Convert Mechanism to legacy Linkage

Serialization:
    mechanism_to_dict: Serialize to dictionary
    mechanism_from_dict: Deserialize from dictionary
    mechanism_to_json: Save to JSON file
    mechanism_from_json: Load from JSON file

See Also:
    pylinkage.dyads: High-level API using Assur group building blocks
"""

# Joint classes
# Builder
from .builder import MechanismBuilder

# Conversion utilities
from .conversion import (
    convert_legacy_dict,
    mechanism_from_linkage,
    mechanism_to_linkage,
)

# Factory functions for common mechanisms
from .factories import (
    fourbar,
    slider_crank,
)
from .joint import (
    AnyJoint,
    GroundJoint,
    Joint,
    JointType,
    PrismaticJoint,
    RevoluteJoint,
)

# Link classes
from .link import (
    AnyLink,
    DriverLink,
    GroundLink,
    Link,
    LinkType,
)

# Mechanism class
from .mechanism import Mechanism

# Serialization
from .serialization import (
    is_legacy_format,
    joint_from_dict,
    joint_to_dict,
    link_from_dict,
    link_to_dict,
    mechanism_from_dict,
    mechanism_from_json,
    mechanism_to_dict,
    mechanism_to_json,
)

__all__ = [
    # Joint classes
    "Joint",
    "RevoluteJoint",
    "PrismaticJoint",
    "GroundJoint",
    "JointType",
    "AnyJoint",
    # Link classes
    "Link",
    "GroundLink",
    "DriverLink",
    "LinkType",
    "AnyLink",
    # Mechanism
    "Mechanism",
    # Builder
    "MechanismBuilder",
    # Factories
    "fourbar",
    "slider_crank",
    # Conversion
    "mechanism_from_linkage",
    "mechanism_to_linkage",
    "convert_legacy_dict",
    # Serialization
    "mechanism_to_dict",
    "mechanism_from_dict",
    "mechanism_to_json",
    "mechanism_from_json",
    "joint_to_dict",
    "joint_from_dict",
    "link_to_dict",
    "link_from_dict",
    "is_legacy_format",
]
