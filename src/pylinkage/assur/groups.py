"""Assur group classes for kinematic analysis.

This module provides the base class and implementations for Assur groups,
which are the fundamental structural units in planar mechanism decomposition.

An Assur group is a kinematic substructure with DOF = 0 when attached to
the frame (or previously solved groups). It represents the minimal structural
unit that can be solved independently.

Assur groups are classified by:
- Class (k): number of links minus 1 divided by 2 for dyads
- Order: related to the number of binary links

Class I groups (k=1) are dyads - two binary links, three joints.

The actual solving logic is delegated to the solver module, which provides
numba-optimized implementations. This module focuses on the graph-based
representation and orchestration.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .._types import Coord
from ..exceptions import UnbuildableError
from ..solver.joints import solve_linear, solve_revolute
from ._types import JointType, NodeId
from .graph import LinkageGraph


@dataclass
class AssurGroup(ABC):
    """Base class for Assur groups.

    An Assur group is a kinematic substructure with DOF = 0 when
    attached to the frame (or previously solved groups). It represents
    the minimal structural unit that can be solved independently.

    Attributes:
        internal_nodes: Node IDs that are part of this group (positions computed).
        anchor_nodes: Node IDs that connect this group to the rest (must be solved first).
        internal_edges: Edge IDs within this group.

    Subclasses must implement:
        - group_class: The class k of this Assur group
        - joint_signature: String like "RRR", "RRP", etc.
        - solve: Compute positions of internal nodes
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

    @abstractmethod
    def solve(
        self,
        graph: LinkageGraph,
        previous_positions: dict[NodeId, Coord],
    ) -> dict[NodeId, Coord]:
        """Solve the positions of internal nodes.

        Given the positions of anchor nodes (from previous_positions),
        compute the positions of all internal nodes using the geometric
        constraints stored in the graph.

        Args:
            graph: The full linkage graph containing constraint information.
            previous_positions: Already-solved node positions (anchors + earlier groups).

        Returns:
            Mapping from internal node IDs to their computed (x, y) positions.

        Raises:
            UnbuildableError: If the configuration is geometrically impossible.
            ValueError: If constraints or anchors are missing.
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

    def solve(
        self,
        graph: LinkageGraph,
        previous_positions: dict[NodeId, Coord],
    ) -> dict[NodeId, Coord]:
        """Solve the RRR dyad using the solver's circle-circle intersection.

        Delegates to solver.joints.solve_revolute() for the actual computation.
        The internal node is at the intersection of two circles centered
        at the anchor nodes. When two solutions exist, the one nearest
        to the current position is chosen (hysteresis).

        Args:
            graph: The linkage graph.
            previous_positions: Positions of anchor nodes.

        Returns:
            Dict mapping the internal node ID to its computed position.

        Raises:
            UnbuildableError: If circles don't intersect (impossible config).
            ValueError: If distances are not set or wrong number of nodes.
        """
        if len(self.internal_nodes) != 1:
            raise ValueError("DyadRRR must have exactly 1 internal node")
        if len(self.anchor_nodes) != 2:
            raise ValueError("DyadRRR must have exactly 2 anchor nodes")

        internal_id = self.internal_nodes[0]
        anchor0_id, anchor1_id = self.anchor_nodes

        # Get anchor positions
        if anchor0_id not in previous_positions:
            raise ValueError(f"Anchor {anchor0_id} position not found")
        if anchor1_id not in previous_positions:
            raise ValueError(f"Anchor {anchor1_id} position not found")

        pos0 = previous_positions[anchor0_id]
        pos1 = previous_positions[anchor1_id]

        if self.distance0 is None or self.distance1 is None:
            raise ValueError("Distances not set for DyadRRR")

        # Get current position for disambiguation (hysteresis)
        current_node = graph.nodes[internal_id]
        current_pos = current_node.position
        current_x = current_pos[0] if current_pos[0] is not None else pos0[0]
        current_y = current_pos[1] if current_pos[1] is not None else pos0[1]

        # Delegate to solver function (single source of truth)
        new_x, new_y = solve_revolute(
            current_x, current_y,
            pos0[0], pos0[1], self.distance0,
            pos1[0], pos1[1], self.distance1,
        )

        if math.isnan(new_x):
            raise UnbuildableError(
                internal_id,
                message=f"No circle intersection for RRR dyad at {internal_id}"
            )

        return {internal_id: (new_x, new_y)}

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

    def solve(
        self,
        graph: LinkageGraph,
        previous_positions: dict[NodeId, Coord],
    ) -> dict[NodeId, Coord]:
        """Solve the RRP dyad using the solver's circle-line intersection.

        Delegates to solver.joints.solve_linear() for the actual computation.

        Args:
            graph: The linkage graph.
            previous_positions: Positions of anchor and line-defining nodes.

        Returns:
            Dict mapping the internal node ID to its computed position.

        Raises:
            UnbuildableError: If circle and line don't intersect.
            ValueError: If constraints or nodes are missing.
        """
        if len(self.internal_nodes) != 1:
            raise ValueError("DyadRRP must have exactly 1 internal node")
        if len(self.anchor_nodes) < 1:
            raise ValueError("DyadRRP must have at least 1 anchor node")

        internal_id = self.internal_nodes[0]
        revolute_anchor = self.anchor_nodes[0]

        # Validate all required positions exist
        if revolute_anchor not in previous_positions:
            raise ValueError(f"Anchor {revolute_anchor} position not found")
        if self.line_node1 is None or self.line_node2 is None:
            raise ValueError("Line nodes not set for DyadRRP")
        if self.line_node1 not in previous_positions:
            raise ValueError(f"Line node {self.line_node1} position not found")
        if self.line_node2 not in previous_positions:
            raise ValueError(f"Line node {self.line_node2} position not found")
        if self.revolute_distance is None:
            raise ValueError("Revolute distance not set for DyadRRP")

        pos_anchor = previous_positions[revolute_anchor]
        pos_line1 = previous_positions[self.line_node1]
        pos_line2 = previous_positions[self.line_node2]

        # Get current position for disambiguation (hysteresis)
        current_node = graph.nodes[internal_id]
        current_pos = current_node.position
        current_x = current_pos[0] if current_pos[0] is not None else pos_anchor[0]
        current_y = current_pos[1] if current_pos[1] is not None else pos_anchor[1]

        # Delegate to solver function (single source of truth)
        new_x, new_y = solve_linear(
            current_x, current_y,
            pos_anchor[0], pos_anchor[1], self.revolute_distance,
            pos_line1[0], pos_line1[1],
            pos_line2[0], pos_line2[1],
        )

        if math.isnan(new_x):
            raise UnbuildableError(
                internal_id,
                message=f"No circle-line intersection for RRP dyad at {internal_id}"
            )

        return {internal_id: (new_x, new_y)}

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
    """

    @property
    def group_class(self) -> int:
        return 1

    @property
    def joint_signature(self) -> str:
        return "RPR"

    def solve(
        self,
        graph: LinkageGraph,
        previous_positions: dict[NodeId, Coord],
    ) -> dict[NodeId, Coord]:
        raise NotImplementedError("RPR dyad solving not yet implemented")

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
    """

    @property
    def group_class(self) -> int:
        return 1

    @property
    def joint_signature(self) -> str:
        return "PRR"

    def solve(
        self,
        graph: LinkageGraph,
        previous_positions: dict[NodeId, Coord],
    ) -> dict[NodeId, Coord]:
        raise NotImplementedError("PRR dyad solving not yet implemented")

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
