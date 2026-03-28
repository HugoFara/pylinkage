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
    "T": JointType.PRISMATIC,  # Translation/slider (alias)
}

# Extended notation for isomer signatures
# T = slider (translating element)
# _ = guide (rail/slot)
# Together T and _ form a prismatic pair
_ISOMER_CHAR_TO_ROLE: dict[str, str] = {
    "R": "revolute",
    "T": "slider",    # Translating element of prismatic pair
    "_": "guide",     # Guide of prismatic pair
    "P": "prismatic", # Generic prismatic (for backwards compatibility)
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
        return _generate_triad_hypergraph(signature, prefix, name)
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
            role=NodeRole.DRIVEN,
            joint_type=joint_types[0],
            name=f"Anchor 0 ({_JOINT_TO_CHAR[joint_types[0]]})",
        )
    )
    graph.add_node(
        Node(
            id=make_id("anchor_1"),
            role=NodeRole.DRIVEN,
            joint_type=joint_types[2],
            name=f"Anchor 1 ({_JOINT_TO_CHAR[joint_types[2]]})",
        )
    )

    # Create internal node
    graph.add_node(
        Node(
            id=make_id("internal_0"),
            role=NodeRole.DRIVEN,
            joint_type=joint_types[1],
            name=f"Internal ({_JOINT_TO_CHAR[joint_types[1]]})",
        )
    )

    # Create edges (links) - pure topology, no distance constraints
    graph.add_edge(
        Edge(
            id=make_id("link_0"),
            source=make_id("anchor_0"),
            target=make_id("internal_0"),
        )
    )
    graph.add_edge(
        Edge(
            id=make_id("link_1"),
            source=make_id("internal_0"),
            target=make_id("anchor_1"),
        )
    )

    return graph


def _generate_triad_hypergraph(
    signature: AssurSignature,
    prefix: str,
    name: str | None,
) -> HypergraphLinkage:
    """Generate hypergraph for a triad (Class II Assur group).

    Triad topology (one arrangement)::

        anchor_0 ---link_0--- internal_0 ---link_1--- anchor_1
                                   |
                                 link_2
                                   |
                              internal_1 ---link_3--- anchor_2

    2 internal nodes, 3 anchor nodes, 4 edges.

    The 6 joint types from the signature map to the nodes:
    - joints[0..2]: anchor_0, anchor_1, anchor_2
    - joints[3..4]: internal_0, internal_1
    - joints[5]: the constraint between the two internals
      (encoded as joint type of internal_0's connection hub)
    """

    def make_id(base: str) -> str:
        return f"{prefix}{base}" if prefix else base

    graph_name = name or f"Triad-{signature.canonical_form}"
    graph = HypergraphLinkage(name=graph_name)

    joint_types = signature.joints

    # Create anchor nodes (3)
    for i in range(3):
        graph.add_node(
            Node(
                id=make_id(f"anchor_{i}"),
                role=NodeRole.DRIVEN,
                joint_type=joint_types[i],
                name=f"Anchor {i} ({_JOINT_TO_CHAR[joint_types[i]]})",
            )
        )

    # Create internal nodes (2)
    for i in range(2):
        graph.add_node(
            Node(
                id=make_id(f"internal_{i}"),
                role=NodeRole.DRIVEN,
                joint_type=joint_types[3 + i],
                name=f"Internal {i} ({_JOINT_TO_CHAR[joint_types[3 + i]]})",
            )
        )

    # Create edges (4 links)
    # internal_0 connects to anchor_0 and anchor_1
    graph.add_edge(
        Edge(
            id=make_id("link_0"),
            source=make_id("anchor_0"),
            target=make_id("internal_0"),
        )
    )
    graph.add_edge(
        Edge(
            id=make_id("link_1"),
            source=make_id("internal_0"),
            target=make_id("anchor_1"),
        )
    )
    # internal_0 connects to internal_1
    graph.add_edge(
        Edge(
            id=make_id("link_2"),
            source=make_id("internal_0"),
            target=make_id("internal_1"),
        )
    )
    # internal_1 connects to anchor_2
    graph.add_edge(
        Edge(
            id=make_id("link_3"),
            source=make_id("internal_1"),
            target=make_id("anchor_2"),
        )
    )

    return graph


def parse_isomer_signature(signature: str) -> tuple[str, tuple[str, ...]]:
    """Parse an extended isomer signature with T/_ notation.

    The extended notation distinguishes between:
    - R = Revolute joint
    - T = Slider (translating element of prismatic pair)
    - _ = Guide (rail/slot of prismatic pair)

    This allows representing all 12 dyadic isomers explicitly.

    Args:
        signature: Isomer signature (e.g., "RRR", "RT_R", "T_R_T").

    Returns:
        Tuple of (normalized_signature, joint_roles) where joint_roles
        is a tuple of role strings ("revolute", "slider", "guide").

    Raises:
        ValueError: If signature contains invalid characters.

    Example:
        >>> sig, roles = parse_isomer_signature("RT_R")
        >>> sig
        'RT_R'
        >>> roles
        ('revolute', 'slider', 'guide', 'revolute')

        >>> sig, roles = parse_isomer_signature("T_R_T")
        >>> roles
        ('slider', 'guide', 'revolute', 'guide', 'slider')
    """
    # Normalize to uppercase, keep underscores
    normalized = signature.upper()

    roles: list[str] = []
    for i, char in enumerate(normalized):
        if char == " ":
            continue  # Skip spaces
        if char not in _ISOMER_CHAR_TO_ROLE:
            valid = ", ".join(_ISOMER_CHAR_TO_ROLE.keys())
            msg = (
                f"Invalid character '{char}' at position {i} in isomer "
                f"signature '{signature}'. Valid characters: {valid}"
            )
            raise ValueError(msg)
        roles.append(_ISOMER_CHAR_TO_ROLE[char])

    return (normalized, tuple(roles))


def isomer_to_canonical(signature: str) -> str:
    """Convert an isomer signature to canonical form.

    Maps extended T/_ notation to standard R/P notation by treating
    T and _ as parts of a single prismatic joint.

    Args:
        signature: Isomer signature (e.g., "RT_R", "T_R_T").

    Returns:
        Canonical signature (e.g., "RPR", "PPR").

    Example:
        >>> isomer_to_canonical("RT_R")
        'RPR'
        >>> isomer_to_canonical("T_R_T")
        'PRP'
        >>> isomer_to_canonical("RRR")
        'RRR'
    """
    # Remove guides and convert sliders to P
    normalized = signature.upper()

    # Count joint types (each T_ or _T pair = one P)
    result: list[str] = []
    i = 0
    while i < len(normalized):
        char = normalized[i]
        if char == "R":
            result.append("R")
            i += 1
        elif char == "T":
            result.append("P")
            # Skip following guide if present
            if i + 1 < len(normalized) and normalized[i + 1] == "_":
                i += 2
            else:
                i += 1
        elif char == "_":
            # Guide without preceding slider - skip if followed by T
            if i + 1 < len(normalized) and normalized[i + 1] == "T":
                result.append("P")
                i += 2
            else:
                i += 1
        elif char == "P":
            result.append("P")
            i += 1
        elif char == " ":
            i += 1  # Skip spaces
        else:
            i += 1  # Skip unknown

    return "".join(result)


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
    from .groups import DYAD_TYPES, Triad

    if isinstance(signature, str):
        signature = parse_signature(signature)

    if signature.group_class == AssurGroupClass.DYAD:
        return DYAD_TYPES.get(signature.canonical_form)
    elif signature.group_class == AssurGroupClass.TRIAD:
        return Triad
    return None
