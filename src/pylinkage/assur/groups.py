"""Assur group classes for kinematic analysis.

This module provides the base class and implementations for Assur groups,
which are the fundamental structural units in planar mechanism decomposition.

An Assur group is a kinematic substructure with DOF = 0 when attached to
the frame (or previously solved groups). It represents the minimal structural
unit that can be identified and analyzed independently.

Assur groups are classified by:
- Class (k): number of links minus 1 divided by 2 for dyads
- Order: related to the number of binary links

Class I groups (k=1) are dyads - two binary links, three joints.

IMPORTANT: These classes are pure data structures defining logical properties.
They do NOT contain solving behavior. Use pylinkage.solver.solve.solve_group()
to compute positions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ._types import JointType, NodeId
from .graph import LinkageGraph


@dataclass
class AssurGroup(ABC):
    """Base class for Assur groups.

    An Assur group is a kinematic substructure with DOF = 0 when
    attached to the frame (or previously solved groups). It represents
    the minimal structural unit that can be identified independently.

    This is a pure data class defining structural properties only.
    Use pylinkage.solver.solve.solve_group() to compute positions.

    Attributes:
        internal_nodes: Node IDs that are part of this group.
        anchor_nodes: Node IDs that connect this group to the rest.
        internal_edges: Edge IDs within this group.

    Subclasses must implement:
        - group_class: The class k of this Assur group
        - joint_signature: String like "RRR", "RRP", etc.
        - constraints: Dict of constraint values for this group
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

    @property
    @abstractmethod
    def constraints(self) -> dict[str, Any]:
        """Return the constraint values for this group.

        Returns a dict mapping constraint names to their values.
        This allows solvers to access constraint data uniformly.
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
    """RRR Dyad: Three revolute joints forming a triangle.

    This is the most common Assur group, equivalent to the Revolute joint.
    It consists of two binary links connecting one internal node to two
    anchor nodes.

    Structure::

        anchor0 ----d0---- internal ----d1---- anchor1

    The internal node position is found at the intersection of two circles:
    - Circle 1: centered at anchor0 with radius distance0
    - Circle 2: centered at anchor1 with radius distance1

    This is a pure data class. Use pylinkage.solver.solve.solve_group()
    to compute positions.

    Attributes:
        distance0: Distance from anchor0 to internal node.
        distance1: Distance from anchor1 to internal node.
    """

    distance0: float | None = None
    distance1: float | None = None

    @property
    def group_class(self) -> int:
        return 1

    @property
    def joint_signature(self) -> str:
        return "RRR"

    @property
    def constraints(self) -> dict[str, Any]:
        """Return the constraint values for this RRR dyad."""
        return {
            "distance0": self.distance0,
            "distance1": self.distance1,
        }

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
        - Edges connecting internal to both anchors with distances
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
            if edge is None or edge.distance is None:
                return False

        return True


@dataclass
class DyadRRP(AssurGroup):
    """RRP Dyad: Two revolute joints and one prismatic joint.

    This corresponds to the Linear joint - a point constrained by a circle
    (from revolute connection) and a line (from prismatic connection).

    Structure::

        revolute_anchor ----distance---- internal
                                            |
                                     (slides on line)
                                            |
        line_node1 ................... line_node2

    The internal node is at the intersection of:
    - A circle centered at revolute_anchor with radius revolute_distance
    - A line passing through line_node1 and line_node2

    This is a pure data class. Use pylinkage.solver.solve.solve_group()
    to compute positions.

    Attributes:
        revolute_distance: Distance from revolute anchor to internal node.
        line_node1: First node defining the prismatic axis.
        line_node2: Second node defining the prismatic axis.
    """

    revolute_distance: float | None = None
    line_node1: NodeId | None = None
    line_node2: NodeId | None = None

    @property
    def group_class(self) -> int:
        return 1

    @property
    def joint_signature(self) -> str:
        return "RRP"

    @property
    def constraints(self) -> dict[str, Any]:
        """Return the constraint values for this RRP dyad."""
        return {
            "revolute_distance": self.revolute_distance,
            "line_node1": self.line_node1,
            "line_node2": self.line_node2,
        }

    @classmethod
    def can_form(
        cls,
        internal_node_ids: list[NodeId],
        anchor_node_ids: list[NodeId],
        graph: LinkageGraph,
    ) -> bool:
        """Check if nodes form an RRP dyad.

        This is more complex than RRR - needs to identify:
        - One revolute connection (edge with distance)
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
    """RPR Dyad: Revolute-Prismatic-Revolute configuration.

    This is a stub implementation for extensibility.
    Solving is not yet implemented in the solver module.
    """

    @property
    def group_class(self) -> int:
        return 1

    @property
    def joint_signature(self) -> str:
        return "RPR"

    @property
    def constraints(self) -> dict[str, Any]:
        """Return the constraint values for this RPR dyad."""
        return {}  # Not yet implemented

    @classmethod
    def can_form(
        cls,
        internal_node_ids: list[NodeId],
        anchor_node_ids: list[NodeId],
        graph: LinkageGraph,
    ) -> bool:
        return False  # Not implemented yet


@dataclass
class DyadPRR(AssurGroup):
    """PRR Dyad: Prismatic-Revolute-Revolute configuration.

    This is a stub implementation for extensibility.
    Solving is not yet implemented in the solver module.
    """

    @property
    def group_class(self) -> int:
        return 1

    @property
    def joint_signature(self) -> str:
        return "PRR"

    @property
    def constraints(self) -> dict[str, Any]:
        """Return the constraint values for this PRR dyad."""
        return {}  # Not yet implemented

    @classmethod
    def can_form(
        cls,
        internal_node_ids: list[NodeId],
        anchor_node_ids: list[NodeId],
        graph: LinkageGraph,
    ) -> bool:
        return False  # Not implemented yet


# Registry of dyad types for identification
DYAD_TYPES: dict[str, type[AssurGroup]] = {
    "RRR": DyadRRR,
    "RRP": DyadRRP,
    "RPR": DyadRPR,
    "PRR": DyadPRR,
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
