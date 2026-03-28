"""Topology analysis tools for planar linkages.

This module provides functions for analyzing the topological properties
of linkage mechanisms, independent of dimensional data.
"""

from .analysis import compute_dof, compute_mobility

__all__ = [
    "compute_dof",
    "compute_mobility",
]
