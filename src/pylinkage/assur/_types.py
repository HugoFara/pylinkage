"""Type definitions for the Assur group module.

This module re-exports types from hypergraph._types for consistency.
The hypergraph module provides the canonical definitions as the
foundational mathematical layer.
"""

# Re-export from the canonical source (hypergraph is the foundational layer)
from ..hypergraph._types import EdgeId, JointType, NodeId, NodeRole

__all__ = ["NodeId", "EdgeId", "JointType", "NodeRole"]
