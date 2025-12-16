"""Assur group classes for kinematic analysis (topology only).

This module provides the base class and implementations for Assur groups,
which are the fundamental structural units in planar mechanism decomposition.

An Assur group is a kinematic substructure with DOF = 0 when attached to
the frame (or previously solved groups). It represents the minimal structural
unit that can be identified and analyzed independently.

Assur groups are classified by:
- Class (k): number of links minus 1 divided by 2 for dyads
- Order: related to the number of binary links

Class I groups (k=1) are dyads - two binary links, three joints.

IMPORTANT: These classes are pure topological data structures.
They do NOT contain solving behavior or dimensional data.
Use pylinkage.solver.solve.solve_group() with a Dimensions object
to compute positions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ._types import JointType, NodeId
from .graph import LinkageGraph


@dataclass
class AssurGroup(ABC):
    """Base class for Assur groups (topology only).

    An Assur group is a kinematic substructure with DOF = 0 when
    attached to the frame (or previously solved groups). It represents
    the minimal structural unit that can be identified independently.

    This is a pure topological data class. Dimensional data (distances,
    angles) is stored separately in a Dimensions object.
    Use pylinkage.solver.solve.solve_group() to compute positions.

    Attributes:
        internal_nodes: Node IDs that are part of this group.
        anchor_nodes: Node IDs that connect this group to the rest.
        internal_edges: Edge IDs within this group.

    Subclasses must implement:
        - group_class: The class k of this Assur group
        - joint_signature: String like "RRR", "RRP", etc.
        - can_form: Check if given elements can form this group type
    """

    internal_nodes: tuple[NodeId, ...] = field(default_factory=tuple)
    anchor_nodes: tuple[NodeId, ...] = field(default_factory=tuple)
    internal_edges: tuple[str, ...] = field(default_factory=tuple)

    @property
    @abstractmethod
    def group_class(self) -> int:
        """Return the class (k) of this Assur group.

        Class I groups (dyads) have k=1.
        """
        ...

    @property
    @abstractmethod
    def joint_signature(self) -> str:
        """Return the joint type signature (e.g., 'RRR', 'RRP').

        The signature describes the joint types in the group:
        - R = Revolute (pin joint)
        - P = Prismatic (slider joint)
        """
        ...

    @classmethod
    @abstractmethod
    def can_form(
        cls,
        internal_node_ids: list[NodeId],
        anchor_node_ids: list[NodeId],
        graph: LinkageGraph,
    ) -> bool:
        """Check if the given nodes can form this group type.

        Args:
            internal_node_ids: Candidate internal nodes.
            anchor_node_ids: Candidate anchor nodes.
            graph: The linkage graph to check against.

        Returns:
            True if these elements can form this Assur group type.
        """
        ...


@dataclass
class DyadRRR(AssurGroup):
    """RRR Dyad: Three revolute joints forming a triangle (topology only).

    This is the most common Assur group, equivalent to the Revolute joint.
    It consists of two binary links connecting one internal node to two
    anchor nodes.

    Structure::

        anchor0 ----d0---- internal ----d1---- anchor1

    The internal node position is found at the intersection of two circles:
    - Circle 1: centered at anchor0 with radius from Dimensions
    - Circle 2: centered at anchor1 with radius from Dimensions

    This is a pure topological data class. Distance constraints are stored
    in a Dimensions object. Use pylinkage.solver.solve.solve_group()
    to compute positions.
    """

    @property
    def group_class(self) -> int:
        return 1

    @property
    def joint_signature(self) -> str:
        return "RRR"

    @classmethod
    def can_form(
        cls,
        internal_node_ids: list[NodeId],
        anchor_node_ids: list[NodeId],
        graph: LinkageGraph,
    ) -> bool:
        """Check if nodes form an RRR dyad.

        Requires:
        - Exactly 1 internal node (revolute type)
        - Exactly 2 anchor nodes
        - Edges connecting internal to both anchors
        """
        if len(internal_node_ids) != 1:
            return False
        if len(anchor_node_ids) < 2:
            return False

        internal_id = internal_node_ids[0]
        internal_node = graph.nodes.get(internal_id)
        if internal_node is None:
            return False

        # Check joint type is revolute
        if internal_node.joint_type != JointType.REVOLUTE:
            return False

        # Check edges exist to both anchors
        for anchor_id in anchor_node_ids[:2]:
            edge = graph.get_edge_between(internal_id, anchor_id)
            if edge is None:
                return False

        return True


@dataclass
class DyadRRP(AssurGroup):
    """RRP Dyad: Two revolute joints and one prismatic joint (topology only).

    This corresponds to the Linear joint - a point constrained by a circle
    (from revolute connection) and a line (from prismatic connection).

    Structure::

        revolute_anchor ----distance---- internal
                                            |
                                     (slides on line)
                                            |
        line_node1 ................... line_node2

    The internal node is at the intersection of:
    - A circle centered at revolute_anchor with radius from Dimensions
    - A line passing through line_node1 and line_node2

    This is a pure topological data class. Distance constraints are stored
    in a Dimensions object. Use pylinkage.solver.solve.solve_group()
    to compute positions.

    Attributes:
        line_node1: First node defining the prismatic axis (topological).
        line_node2: Second node defining the prismatic axis (topological).
    """

    line_node1: NodeId | None = None
    line_node2: NodeId | None = None

    @property
    def group_class(self) -> int:
        return 1

    @property
    def joint_signature(self) -> str:
        return "RRP"

    @classmethod
    def can_form(
        cls,
        internal_node_ids: list[NodeId],
        anchor_node_ids: list[NodeId],
        graph: LinkageGraph,
    ) -> bool:
        """Check if nodes form an RRP dyad.

        This is more complex than RRR - needs to identify:
        - One revolute connection (edge to anchor)
        - One line constraint (defined by two other nodes)
        """
        if len(internal_node_ids) != 1:
            return False

        internal_id = internal_node_ids[0]
        internal_node = graph.nodes.get(internal_id)
        if internal_node is None:
            return False

        # The internal node should have connections suggesting a line constraint
        # This is a simplified check - in practice we'd need more info
        # about which connection is the prismatic one
        edges = graph.get_edges_for_node(internal_id)

        # Need at least 3 connections: 1 revolute + 2 for line
        return len(edges) >= 3


@dataclass
class DyadRPR(AssurGroup):
    """RPR Dyad: Revolute-Prismatic-Revolute configuration (topology only).

    The internal node has a prismatic (sliding) constraint, positioned at the
    intersection of a circle (from revolute anchor) and a line (defined by
    the prismatic axis).

    Geometrically equivalent to RRP but with the prismatic at the internal node.

    Structure::

        anchor0 ----distance---- internal (prismatic) ----line---- anchor1
                                      |
                                 slides on line
    """

    # Line-defining nodes for the prismatic axis
    line_node1: NodeId | None = None
    line_node2: NodeId | None = None

    @property
    def group_class(self) -> int:
        return 1

    @property
    def joint_signature(self) -> str:
        return "RPR"

    @classmethod
    def can_form(
        cls,
        internal_node_ids: list[NodeId],
        anchor_node_ids: list[NodeId],
        graph: LinkageGraph,
    ) -> bool:
        """Check if nodes form an RPR dyad.

        Requires:
        - Exactly 1 internal node (prismatic type)
        - At least 1 revolute anchor connection
        - Line constraint from other anchors
        """
        if len(internal_node_ids) != 1:
            return False

        internal_id = internal_node_ids[0]
        internal_node = graph.nodes.get(internal_id)
        if internal_node is None:
            return False

        # Internal node should be prismatic type
        if internal_node.joint_type != JointType.PRISMATIC:
            return False

        # Check for at least one revolute anchor connection
        has_revolute_anchor = False
        for anchor_id in anchor_node_ids:
            anchor = graph.nodes.get(anchor_id)
            if anchor and anchor.joint_type == JointType.REVOLUTE:
                edge = graph.get_edge_between(internal_id, anchor_id)
                if edge is not None:
                    has_revolute_anchor = True
                    break

        return has_revolute_anchor


@dataclass
class DyadPRR(AssurGroup):
    """PRR Dyad: Prismatic-Revolute-Revolute configuration (topology only).

    Similar to RRP but with different anchor types. The internal node is
    revolute, with one anchor providing a prismatic constraint.

    Geometrically equivalent to RRP - circle-line intersection.
    """

    # Line-defining nodes for the prismatic axis
    line_node1: NodeId | None = None
    line_node2: NodeId | None = None

    @property
    def group_class(self) -> int:
        return 1

    @property
    def joint_signature(self) -> str:
        return "PRR"

    @classmethod
    def can_form(
        cls,
        internal_node_ids: list[NodeId],
        anchor_node_ids: list[NodeId],
        graph: LinkageGraph,
    ) -> bool:
        """Check if nodes form a PRR dyad.

        Requires:
        - Exactly 1 internal node (revolute type)
        - At least one prismatic anchor
        - At least one revolute anchor
        """
        if len(internal_node_ids) != 1:
            return False

        internal_id = internal_node_ids[0]
        internal_node = graph.nodes.get(internal_id)
        if internal_node is None:
            return False

        # Internal node should be revolute type
        if internal_node.joint_type != JointType.REVOLUTE:
            return False

        # Check for at least one prismatic anchor
        has_prismatic = False
        has_revolute = False
        for anchor_id in anchor_node_ids:
            anchor = graph.nodes.get(anchor_id)
            if anchor:
                edge = graph.get_edge_between(internal_id, anchor_id)
                if edge is not None:
                    if anchor.joint_type == JointType.PRISMATIC:
                        has_prismatic = True
                    elif anchor.joint_type == JointType.REVOLUTE:
                        has_revolute = True

        return has_prismatic and has_revolute


@dataclass
class DyadPP(AssurGroup):
    """PP Dyad: Two prismatic joints - line-line intersection (topology only).

    The internal node is positioned at the intersection of two lines.
    This covers isomers like T_R_T, T_RT_, _TRT_.

    Structure::

        line1_node1 ......... internal ......... line1_node2
                                 |
                                 |
        line2_node1 .............|.............. line2_node2

    The internal node is at the intersection of:
    - Line 1: passing through line1_node1 and line1_node2
    - Line 2: passing through line2_node1 and line2_node2

    Attributes:
        line1_node1: First node of line 1.
        line1_node2: Second node of line 1.
        line2_node1: First node of line 2.
        line2_node2: Second node of line 2.
    """

    line1_node1: NodeId | None = None
    line1_node2: NodeId | None = None
    line2_node1: NodeId | None = None
    line2_node2: NodeId | None = None

    @property
    def group_class(self) -> int:
        return 1

    @property
    def joint_signature(self) -> str:
        return "PP"

    @classmethod
    def can_form(
        cls,
        internal_node_ids: list[NodeId],
        anchor_node_ids: list[NodeId],
        graph: LinkageGraph,
    ) -> bool:
        """Check if nodes form a PP dyad (line-line intersection).

        Requires:
        - Exactly 1 internal node
        - At least 4 anchor nodes (2 per line)
        - Suitable edge structure for two line constraints
        """
        if len(internal_node_ids) != 1:
            return False

        # Need 4 anchors to define two lines
        if len(anchor_node_ids) < 4:
            return False

        internal_id = internal_node_ids[0]
        internal_node = graph.nodes.get(internal_id)
        if internal_node is None:
            return False

        # Count prismatic-type connections
        prismatic_count = 0
        for anchor_id in anchor_node_ids:
            anchor = graph.nodes.get(anchor_id)
            if anchor and anchor.joint_type == JointType.PRISMATIC:
                prismatic_count += 1

        # Need at least 2 prismatic connections for line-line
        return prismatic_count >= 2


# Registry of dyad types for identification
DYAD_TYPES: dict[str, type[AssurGroup]] = {
    "RRR": DyadRRR,
    "RRP": DyadRRP,
    "RPR": DyadRPR,
    "PRR": DyadPRR,
    "PP": DyadPP,
    # Aliases for isomer notation
    "PRP": DyadPP,
    "PPR": DyadPP,
}


def identify_dyad_type(
    internal_node_ids: list[NodeId],
    anchor_node_ids: list[NodeId],
    graph: LinkageGraph,
) -> type[AssurGroup] | None:
    """Identify which dyad type can be formed from the given elements.

    Tries each registered dyad type in order and returns the first
    one that matches.

    Args:
        internal_node_ids: Candidate internal nodes.
        anchor_node_ids: Candidate anchor nodes.
        graph: The linkage graph.

    Returns:
        The matching AssurGroup subclass, or None if no match found.
    """
    for dyad_cls in DYAD_TYPES.values():
        if dyad_cls.can_form(internal_node_ids, anchor_node_ids, graph):
            return dyad_cls
    return None
