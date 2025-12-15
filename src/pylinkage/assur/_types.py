"""Type definitions for the Assur group module.

This module re-exports canonical types from pylinkage._types for consistency.
The _types module is the single source of truth for kinematic type definitions.
"""

# Re-export from the canonical source
from .._types import EdgeId, JointType, NodeId, NodeRole

__all__ = ["NodeId", "EdgeId", "JointType", "NodeRole"]
