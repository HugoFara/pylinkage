"""Mechanism module - proper Links + Joints model for planar linkages.

This module provides a clear, intuitive API for defining planar mechanisms
using standard mechanical engineering terminology:

- **Joints** are actual connection points (revolute pins, prismatic sliders)
- **Links** are rigid bodies connecting joints
- **Dyads** are factory functions for common Assur groups

This contrasts with the legacy `joints` module where "joints" were actually
Assur groups (combinations of multiple joints and links).

Basic Usage:
    >>> from pylinkage.mechanism import (
    ...     Mechanism, GroundJoint, GroundLink,
    ...     create_crank, create_rrr_dyad,
    ... )
    >>>
    >>> # Create ground joints
    >>> O1 = GroundJoint("O1", position=(0.0, 0.0))
    >>> O2 = GroundJoint("O2", position=(2.0, 0.0))
    >>> ground = GroundLink("ground", joints=[O1, O2])
    >>>
    >>> # Create crank (driver link + output joint)
    >>> crank, A = create_crank(O1, radius=1.0, angular_velocity=0.1)
    >>>
    >>> # Create rocker via RRR dyad
    >>> link1, link2, B = create_rrr_dyad(A, O2, distance1=2.0, distance2=1.5)
    >>>
    >>> # Build mechanism
    >>> mechanism = Mechanism(
    ...     name="Four-Bar",
    ...     joints=[O1, O2, A, B],
    ...     links=[ground, crank, link1, link2],
    ... )
    >>>
    >>> # Simulate
    >>> for positions in mechanism.step():
    ...     print(positions)

Classes:
    Joint: Base class for all joints
    RevoluteJoint: Pin joint (1 DOF rotation)
    PrismaticJoint: Slider joint (1 DOF translation)
    GroundJoint: Fixed revolute joint on the frame

    Link: Rigid body connecting joints
    GroundLink: The stationary frame
    DriverLink: Motor-driven input link

    Mechanism: The main orchestrator class

Factory Functions:
    create_crank: Create a driver crank mechanism
    create_rrr_dyad: Create an RRR dyad (circle-circle intersection)
    create_rrp_dyad: Create an RRP dyad (circle-line intersection)
    create_fixed_dyad: Create a fixed angular constraint

Conversion:
    mechanism_from_linkage: Convert legacy Linkage to Mechanism
    mechanism_to_linkage: Convert Mechanism to legacy Linkage

Serialization:
    mechanism_to_dict: Serialize to dictionary
    mechanism_from_dict: Deserialize from dictionary
    mechanism_to_json: Save to JSON file
    mechanism_from_json: Load from JSON file
"""

# Joint classes
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

# Dyad factory functions
from .dyads import (
    create_crank,
    create_fixed_dyad,
    create_rrp_dyad,
    create_rrr_dyad,
)

# Conversion utilities
from .conversion import (
    convert_legacy_dict,
    mechanism_from_linkage,
    mechanism_to_linkage,
)

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
    # Factory functions
    "create_crank",
    "create_rrr_dyad",
    "create_rrp_dyad",
    "create_fixed_dyad",
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
