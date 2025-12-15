"""Assur group signature parsing and hypergraph generation.

This module provides tools for defining Assur groups using formal kinematic
syntax (e.g., "RRR", "RPR", "PRR") and generating pure topological hypergraphs
from these signatures.

The generated hypergraphs contain only structure (nodes, edges, joint types)
without distance constraints - they are mathematical objects representing
topology only.

Example:
    >>> from pylinkage.assur import parse_signature, signature_to_hypergraph
    >>>
    >>> # Parse formal syntax
    >>> sig = parse_signature("RRR")
    >>> print(sig.joint_signature)
    'RRR'
    >>>
    >>> # Generate pure topology
    >>> graph = signature_to_hypergraph(sig)
    >>> print(list(graph.nodes.keys()))
    ['anchor_0', 'anchor_1', 'internal_0']
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from .._types import JointType, NodeRole
from ..hypergraph.core import Edge, Node
from ..hypergraph.graph import HypergraphLinkage

if TYPE_CHECKING:
    from .groups import AssurGroup


class AssurGroupClass(IntEnum):
    """Classification of Assur groups by structural complexity.

    Assur groups are classified by their "class" (k), which corresponds
    to the number of internal nodes and structural complexity:

    - Class I (DYAD): 2 links, 3 joints, 1 internal node
    - Class II (TRIAD): 4 links, 6 joints, 2 internal nodes
    - Class III (TETRAD): 6 links, 9 joints, 3 internal nodes

    Higher classes follow the pattern: k links = 2*class, joints = 3*class.
    """

    DYAD = 1  # Class I: 3 joints
    TRIAD = 2  # Class II: 6 joints (future)
    TETRAD = 3  # Class III: 9 joints (future)


# Character to JointType mapping
# Both "P" and "T" map to PRISMATIC (P preferred)
_CHAR_TO_JOINT: dict[str, JointType] = {
    "R": JointType.REVOLUTE,
    "P": JointType.PRISMATIC,  # Preferred
    "T": JointType.PRISMATIC,  # Translation (alias)
}

# JointType to character mapping (P preferred for canonical form)
_JOINT_TO_CHAR: dict[JointType, str] = {
    JointType.REVOLUTE: "R",
    JointType.PRISMATIC: "P",
}

# Expected joint count per group class
_JOINTS_PER_CLASS: dict[AssurGroupClass, int] = {
    AssurGroupClass.DYAD: 3,
    AssurGroupClass.TRIAD: 6,
    AssurGroupClass.TETRAD: 9,
}


@dataclass(frozen=True)
class AssurSignature:
    """Parsed representation of an Assur group signature.

    This is an immutable, hashable representation of the joint sequence
    defining an Assur group's topology.

    Attributes:
        joints: Tuple of JointType values in order.
        group_class: The Assur group class (dyad, triad, etc.).
        raw_string: The original signature string for display.

    Example:
        >>> sig = parse_signature("RRR")
        >>> sig.joints
        (JointType.REVOLUTE, JointType.REVOLUTE, JointType.REVOLUTE)
        >>> sig.group_class
        <AssurGroupClass.DYAD: 1>
        >>> sig.canonical_form
        'RRR'
    """

    joints: tuple[JointType, ...]
    group_class: AssurGroupClass
    raw_string: str = ""

    def __post_init__(self) -> None:
        """Validate joint count matches group class."""
        expected = _JOINTS_PER_CLASS.get(self.group_class)
        if expected is not None and len(self.joints) != expected:
            msg = (
                f"Group class {self.group_class.name} requires "
                f"{expected} joints, got {len(self.joints)}"
            )
            raise ValueError(msg)

    @property
    def canonical_form(self) -> str:
        """Return canonical string representation (e.g., 'RRR', 'RRP').

        Uses 'P' for prismatic joints (preferred over 'T').
        """
        return "".join(_JOINT_TO_CHAR[j] for j in self.joints)

    @property
    def num_internal_nodes(self) -> int:
        """Number of internal (driven) nodes in this group.

        Dyad: 1, Triad: 2, Tetrad: 3, etc.
        """
        return self.group_class.value

    @property
    def num_anchor_nodes(self) -> int:
        """Number of anchor nodes (connections to known positions).

        Dyad: 2, Triad: 3, Tetrad: 4, etc.
        """
        return self.group_class.value + 1

    @property
    def num_links(self) -> int:
        """Number of binary links in this group.

        Dyad: 2, Triad: 4, Tetrad: 6, etc.
        """
        return 2 * self.group_class.value


def parse_signature(signature: str) -> AssurSignature:
    """Parse a signature string into an AssurSignature.

    The signature describes the joint types in the Assur group:
    - R = Revolute (pin joint)
    - P or T = Prismatic (slider joint) - both accepted, P preferred
    - _ = Optional separator for readability

    Args:
        signature: The signature string (e.g., "RRR", "R_P_R", "RPR").

    Returns:
        Parsed AssurSignature.

    Raises:
        ValueError: If signature is empty, contains invalid characters,
            or has wrong joint count for any known group class.

    Example:
        >>> sig = parse_signature("RRR")
        >>> sig.joints
        (JointType.REVOLUTE, JointType.REVOLUTE, JointType.REVOLUTE)

        >>> sig = parse_signature("R_P_R")  # With separators
        >>> sig.canonical_form
        'RPR'

        >>> sig = parse_signature("RTR")  # T accepted as alias for P
        >>> sig.canonical_form
        'RPR'
    """
    # Remove separators and whitespace, convert to uppercase
    cleaned = signature.upper().replace("_", "").replace(" ", "")

    if not cleaned:
        msg = "Signature cannot be empty"
        raise ValueError(msg)

    # Parse each character
    joints: list[JointType] = []
    for i, char in enumerate(cleaned):
        if char not in _CHAR_TO_JOINT:
            msg = (
                f"Invalid character '{char}' at position {i} in signature "
                f"'{signature}'. Valid characters: R (revolute), P or T (prismatic)"
            )
            raise ValueError(msg)
        joints.append(_CHAR_TO_JOINT[char])

    # Determine group class based on joint count
    joint_count = len(joints)
    group_class: AssurGroupClass | None = None

    for cls, count in _JOINTS_PER_CLASS.items():
        if joint_count == count:
            group_class = cls
            break

    if group_class is None:
        valid_counts = sorted(_JOINTS_PER_CLASS.values())
        msg = (
            f"Invalid joint count {joint_count} in signature '{signature}'. "
            f"Valid counts: {valid_counts} (dyad=3, triad=6, tetrad=9)."
        )
        raise ValueError(msg)

    return AssurSignature(
        joints=tuple(joints),
        group_class=group_class,
        raw_string=signature,
    )


def signature_to_hypergraph(
    signature: AssurSignature | str,
    *,
    prefix: str = "",
    name: str | None = None,
) -> HypergraphLinkage:
    """Generate a topological hypergraph from an Assur signature.

    Creates a pure topological hypergraph with NO distance constraints.
    All edges have distance=None, representing only structural connections.

    The generated graph follows standard naming conventions:
    - Anchor nodes: "anchor_0", "anchor_1", ...
    - Internal nodes: "internal_0", "internal_1", ...
    - Edges (links): "link_0", "link_1", ...

    Args:
        signature: Either an AssurSignature or a string to parse.
        prefix: Optional prefix for all node/edge IDs (useful for assembly).
        name: Optional name for the hypergraph.

    Returns:
        HypergraphLinkage with pure topology (no constraints).

    Raises:
        NotImplementedError: For group classes not yet implemented (triad, tetrad).

    Example:
        >>> graph = signature_to_hypergraph("RRR")
        >>> len(graph.nodes)
        3  # 2 anchors + 1 internal
        >>> len(graph.edges)
        2  # 2 links
        >>> graph.edges["link_0"].distance is None
        True

        >>> # With prefix for unique IDs
        >>> leg = signature_to_hypergraph("RRR", prefix="leg1_")
        >>> "leg1_anchor_0" in leg.nodes
        True
    """
    if isinstance(signature, str):
        signature = parse_signature(signature)

    # Dispatch based on group class
    if signature.group_class == AssurGroupClass.DYAD:
        return _generate_dyad_hypergraph(signature, prefix, name)
    elif signature.group_class == AssurGroupClass.TRIAD:
        msg = (
            "Hypergraph generation for TRIAD (Class II) not yet implemented. "
            "Dyads (RRR, RRP, RPR, PRR, etc.) are currently supported."
        )
        raise NotImplementedError(msg)
    elif signature.group_class == AssurGroupClass.TETRAD:
        msg = (
            "Hypergraph generation for TETRAD (Class III) not yet implemented. "
            "Dyads (RRR, RRP, RPR, PRR, etc.) are currently supported."
        )
        raise NotImplementedError(msg)
    else:
        msg = f"Unknown group class: {signature.group_class}"
        raise NotImplementedError(msg)


def _generate_dyad_hypergraph(
    signature: AssurSignature,
    prefix: str,
    name: str | None,
) -> HypergraphLinkage:
    """Generate hypergraph for a dyad (Class I Assur group).

    Dyad topology::

        anchor_0 ----link_0---- internal_0 ----link_1---- anchor_1
        (joint_0)               (joint_1)                 (joint_2)

    The joint types from the signature map to:
    - joints[0]: anchor_0's joint type
    - joints[1]: internal_0's joint type
    - joints[2]: anchor_1's joint type
    """

    def make_id(base: str) -> str:
        return f"{prefix}{base}" if prefix else base

    graph_name = name or f"Dyad-{signature.canonical_form}"
    graph = HypergraphLinkage(name=graph_name)

    # Map joint types to nodes
    joint_types = signature.joints

    # Create anchor nodes (role=DRIVEN by default, user can change)
    graph.add_node(
        Node(
            id=make_id("anchor_0"),
            position=(None, None),
            role=NodeRole.DRIVEN,
            joint_type=joint_types[0],
            name=f"Anchor 0 ({_JOINT_TO_CHAR[joint_types[0]]})",
        )
    )
    graph.add_node(
        Node(
            id=make_id("anchor_1"),
            position=(None, None),
            role=NodeRole.DRIVEN,
            joint_type=joint_types[2],
            name=f"Anchor 1 ({_JOINT_TO_CHAR[joint_types[2]]})",
        )
    )

    # Create internal node
    graph.add_node(
        Node(
            id=make_id("internal_0"),
            position=(None, None),
            role=NodeRole.DRIVEN,
            joint_type=joint_types[1],
            name=f"Internal ({_JOINT_TO_CHAR[joint_types[1]]})",
        )
    )

    # Create edges (links) - NO distance constraints (pure topology)
    graph.add_edge(
        Edge(
            id=make_id("link_0"),
            source=make_id("anchor_0"),
            target=make_id("internal_0"),
            distance=None,
        )
    )
    graph.add_edge(
        Edge(
            id=make_id("link_1"),
            source=make_id("internal_0"),
            target=make_id("anchor_1"),
            distance=None,
        )
    )

    return graph


def signature_to_group_class(signature: AssurSignature | str) -> type[AssurGroup] | None:
    """Get the AssurGroup class corresponding to a signature.

    This bridges the formal signature syntax to the existing AssurGroup
    class hierarchy (DyadRRR, DyadRRP, etc.).

    Args:
        signature: The parsed signature or string to parse.

    Returns:
        The matching AssurGroup subclass, or None if not implemented.

    Example:
        >>> sig = parse_signature("RRR")
        >>> cls = signature_to_group_class(sig)
        >>> cls.__name__
        'DyadRRR'

        >>> cls = signature_to_group_class("RPR")
        >>> cls.__name__
        'DyadRPR'
    """
    # Import here to avoid circular imports
    from .groups import DYAD_TYPES

    if isinstance(signature, str):
        signature = parse_signature(signature)

    return DYAD_TYPES.get(signature.canonical_form)
