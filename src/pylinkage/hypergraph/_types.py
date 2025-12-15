"""Type definitions for the hypergraph module.

This module re-exports canonical types from pylinkage._types for backward
compatibility. New code should import directly from pylinkage._types.

.. deprecated:: 0.8.0
    Import types directly from ``pylinkage._types`` instead of this module.
    This module is maintained for backward compatibility only.

Example:
    Preferred::

        from pylinkage._types import JointType, NodeRole, NodeId

    Deprecated (but still works)::

        from pylinkage.hypergraph._types import JointType, NodeRole
"""

# Re-export all types from the canonical source
from .._types import (
    ComponentId,
    EdgeId,
    HyperedgeId,
    JointType,
    NodeId,
    NodeRole,
    PortId,
)

__all__ = [
    "ComponentId",
    "EdgeId",
    "HyperedgeId",
    "JointType",
    "NodeId",
    "NodeRole",
    "PortId",
]
