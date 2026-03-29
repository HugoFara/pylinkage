"""Topology analysis and enumeration tools for planar linkages.

This module provides:
- DOF computation using Grübler's formula
- Graph isomorphism detection for topology deduplication
- Systematic enumeration of all valid 1-DOF topologies
- A built-in catalog of known topologies (up to 8 links)
"""

from .analysis import MobilityInfo, compute_dof, compute_mobility
from .catalog import CatalogEntry, TopologyCatalog, load_catalog
from .enumeration import enumerate_all, enumerate_topologies
from .isomorphism import are_isomorphic, canonical_form, canonical_hash

__all__ = [
    # Analysis (Phase 2.1)
    "compute_dof",
    "compute_mobility",
    "MobilityInfo",
    # Isomorphism (Phase 2.2)
    "are_isomorphic",
    "canonical_form",
    "canonical_hash",
    # Enumeration (Phase 2.3)
    "enumerate_topologies",
    "enumerate_all",
    # Catalog (Phase 2.4)
    "TopologyCatalog",
    "CatalogEntry",
    "load_catalog",
]
