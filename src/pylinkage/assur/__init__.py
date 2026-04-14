"""Assur group decomposition for planar linkages.

This module provides a graph-based representation of linkage mechanisms
and tools for Assur group decomposition. It offers an alternative way
to define and analyze linkages using formal kinematic theory.

IMPORTANT: This module defines logical/structural properties only.
Assur groups are pure data classes without solving behavior.
Use pylinkage.solver.solve.solve_group() for position computation.

Key components:
- AssurMechanism: Wrapper adding Assur analysis to a Mechanism
- LinkageGraph: Graph representation with nodes (joints) and edges (links)
- AssurGroup: Base class for Assur groups (structural units)
- DyadRRR, DyadRRP: Class I Assur groups (dyads)
- decompose_assur_groups: Decomposition algorithm
- Conversion functions between graph and Mechanism representations

Example usage:

    >>> from pylinkage.mechanism import Mechanism
    >>> from pylinkage.assur import AssurMechanism
    >>>
    >>> # Wrap a mechanism for Assur analysis
    >>> mechanism = Mechanism(joints=[...], links=[...])
    >>> assur = AssurMechanism(mechanism)
    >>>
    >>> # Access structural properties
    >>> print(f"DOF: {assur.degree_of_freedom}")
    >>> print(f"Groups: {[g.joint_signature for g in assur.assur_groups]}")
    >>>
    >>> # Simulation via delegation
    >>> for positions in assur.step():
    ...     print(positions)
"""

__all__ = [
    # Types
    "JointType",
    "NodeRole",
    "NodeId",
    "EdgeId",
    # Wrapper class
    "AssurMechanism",
    # Analysis results
    "MobilityResult",
    "StructuralAnalysis",
    # Graph structures
    "LinkageGraph",
    "Node",
    "Edge",
    # Assur groups (pure data classes)
    "AssurGroup",
    "Dyad",
    "Triad",
    "identify_group_type",
    # Backwards-compatible aliases
    "DyadRRR",
    "DyadRRP",
    "DyadRPR",
    "DyadPRR",
    "DYAD_TYPES",
    "identify_dyad_type",
    # Signature parsing and hypergraph generation
    "AssurGroupClass",
    "AssurSignature",
    "parse_signature",
    "signature_to_hypergraph",
    "signature_to_group_class",
    # Decomposition
    "DecompositionResult",
    "decompose_assur_groups",
    "validate_decomposition",
    # Conversion to/from Mechanism (preferred)
    "graph_to_mechanism",
    "mechanism_to_graph",
    # Conversion to/from Hypergraph
    "from_hypergraph",
    "to_hypergraph",
    # Serialization
    "graph_to_dict",
    "graph_from_dict",
    "graph_to_json",
    "graph_from_json",
]

from ._types import EdgeId, JointType, NodeId, NodeRole
from .analysis import MobilityResult, StructuralAnalysis
from .assur_mechanism import AssurMechanism
from .decomposition import (
    DecompositionResult,
    decompose_assur_groups,
    validate_decomposition,
)
from .graph import Edge, LinkageGraph, Node
from .groups import (
    DYAD_TYPES,
    AssurGroup,
    Dyad,
    DyadPRR,
    DyadRPR,
    DyadRRP,
    DyadRRR,
    Triad,
    identify_dyad_type,
    identify_group_type,
)
from .hypergraph_conversion import from_hypergraph, to_hypergraph
from .mechanism_conversion import graph_to_mechanism, mechanism_to_graph
from .serialization import (
    graph_from_dict,
    graph_from_json,
    graph_to_dict,
    graph_to_json,
)
from .signature import (
    AssurGroupClass,
    AssurSignature,
    parse_signature,
    signature_to_group_class,
    signature_to_hypergraph,
)
