"""Assur group classes for kinematic analysis (topology only).

This module provides the base class and implementations for Assur groups,
which are the fundamental structural units in planar mechanism decomposition.

An Assur group is a kinematic substructure with DOF = 0 when attached to
the frame (or previously solved groups). It represents the minimal structural
unit that can be identified and analyzed independently.

Groups are **parameterized by signature** rather than having one class per
joint-type combination. This scales to triads (64 combinations) and
tetrads (512 combinations) without an explosion of classes.

IMPORTANT: These classes are pure topological data structures.
They do NOT contain solving behavior or dimensional data.
Use pylinkage.solver.solve.solve_group() with a Dimensions object
to compute positions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ._types import JointType, NodeId
from .graph import LinkageGraph


def _count_prismatic(signature: str) -> int:
    """Count prismatic joints in a canonical signature string."""
    return sum(1 for c in signature if c == "P")


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

    @property
    def solver_category(self) -> str:
        """Return the solving geometry category.

        The solver dispatches based on this, not on the specific class.
        Categories for dyads (group_class=1):
        - "circle_circle": All-revolute constraints (e.g., RRR)
        - "circle_line": Mixed revolute + prismatic (e.g., RRP, RPR, PRR)
        - "line_line": All-prismatic constraints (e.g., PP, PPR)
        Categories for higher-order groups (group_class>=2):
        - "newton_raphson": Simultaneous constraint solving
        """
        if self.group_class >= 2:
            return "newton_raphson"
        sig = self.joint_signature
        n_prismatic = _count_prismatic(sig)
        if n_prismatic == 0:
            return "circle_circle"
        elif n_prismatic >= 2:
            return "line_line"
        else:
            return "circle_line"


@dataclass
class Dyad(AssurGroup):
    """Class I Assur group — parameterized by joint signature.

    A dyad consists of 2 binary links, 3 joints, and 1 internal node.
    The signature (e.g., "RRR", "RRP", "RPR") determines the joint
    types and solving geometry.

    Structure::

        anchor_0 ----link_0---- internal_0 ----link_1---- anchor_1
        (joint_0)               (joint_1)                 (joint_2)

    For prismatic variants, additional line-defining nodes are stored:
    - RRP/RPR/PRR: one line defined by (line_node1, line_node2)
    - PP: two lines defined by (line_node1, line_node2) and
      (line2_node1, line2_node2)

    Attributes:
        _signature: Canonical joint signature string (e.g., "RRR").
        line_node1: First node defining prismatic line constraint.
        line_node2: Second node defining prismatic line constraint.
        line2_node1: First node of second line (PP only).
        line2_node2: Second node of second line (PP only).
    """

    _signature: str = "RRR"

    # Line constraint nodes (for prismatic variants)
    line_node1: NodeId | None = None
    line_node2: NodeId | None = None

    # Second line constraint nodes (for PP / line-line variants)
    line2_node1: NodeId | None = None
    line2_node2: NodeId | None = None

    @property
    def group_class(self) -> int:
        return 1

    @property
    def joint_signature(self) -> str:
        return self._signature

    @classmethod
    def can_form(
        cls,
        internal_node_ids: list[NodeId],
        anchor_node_ids: list[NodeId],
        graph: LinkageGraph,
        *,
        signature: str | None = None,
    ) -> bool:
        """Check if nodes can form a dyad with the given (or any) signature.

        If signature is provided, checks against that specific joint pattern.
        Otherwise, checks if any valid dyad can be formed.

        Args:
            internal_node_ids: Candidate internal nodes (must be exactly 1).
            anchor_node_ids: Candidate anchor nodes.
            graph: The linkage graph to check against.
            signature: Optional specific signature to check (e.g., "RRR").

        Returns:
            True if these elements can form a dyad.
        """
        if len(internal_node_ids) != 1:
            return False

        internal_id = internal_node_ids[0]
        internal_node = graph.nodes.get(internal_id)
        if internal_node is None:
            return False

        if signature is not None:
            return _check_dyad_signature(
                signature,
                internal_id,
                internal_node,
                anchor_node_ids,
                graph,
            )

        # Try all common signatures
        for sig in ("RRR", "RRP", "RPR", "PRR", "PP"):
            if _check_dyad_signature(
                sig,
                internal_id,
                internal_node,
                anchor_node_ids,
                graph,
            ):
                return True
        return False


def _check_dyad_signature(
    signature: str,
    internal_id: NodeId,
    internal_node,
    anchor_node_ids: list[NodeId],
    graph: LinkageGraph,
) -> bool:
    """Check if a specific dyad signature can be formed."""
    n_prismatic = _count_prismatic(signature)

    if n_prismatic == 0:
        # RRR: revolute internal, 2 anchors with edges
        if internal_node.joint_type != JointType.REVOLUTE:
            return False
        if len(anchor_node_ids) < 2:
            return False
        for anchor_id in anchor_node_ids[:2]:
            if graph.get_edge_between(internal_id, anchor_id) is None:
                return False
        return True

    elif n_prismatic == 1:
        # One prismatic joint — check position in signature
        if signature == "RPR":
            # Internal node is prismatic
            if internal_node.joint_type != JointType.PRISMATIC:
                return False
            for anchor_id in anchor_node_ids:
                anchor = graph.nodes.get(anchor_id)
                if (
                    anchor and anchor.joint_type == JointType.REVOLUTE
                    and graph.get_edge_between(internal_id, anchor_id) is not None
                ):
                        return True
            return False
        elif signature == "PRR":
            # Internal is revolute, need prismatic + revolute anchors
            if internal_node.joint_type != JointType.REVOLUTE:
                return False
            has_p = has_r = False
            for anchor_id in anchor_node_ids:
                anchor = graph.nodes.get(anchor_id)
                if anchor and graph.get_edge_between(internal_id, anchor_id) is not None:
                    if anchor.joint_type == JointType.PRISMATIC:
                        has_p = True
                    elif anchor.joint_type == JointType.REVOLUTE:
                        has_r = True
            return has_p and has_r
        else:
            # RRP: need >= 3 connections (1 revolute edge + 2 line nodes)
            edges = graph.get_edges_for_node(internal_id)
            return len(edges) >= 3

    else:
        # PP: line-line intersection, need >= 4 anchors with >= 2 prismatic
        if len(anchor_node_ids) < 4:
            return False
        prismatic_count = sum(
            1
            for aid in anchor_node_ids
            if (a := graph.nodes.get(aid)) and a.joint_type == JointType.PRISMATIC
        )
        return prismatic_count >= 2


@dataclass
class Triad(AssurGroup):
    """Class II Assur group — parameterized by joint signature.

    A triad consists of 4 links, 6 joints, and 2 internal nodes connected
    to 3 anchor nodes. The signature has 6 characters (e.g., "RRRRRR").

    Triads require solving 3 simultaneous constraints (Newton-Raphson),
    unlike dyads which only intersect 2 constraints.

    Structure (one possible arrangement)::

        anchor_0 ------- internal_0 ------- anchor_1
                             |
                         internal_1
                             |
                         anchor_2

    Attributes:
        _signature: Canonical joint signature string (6 chars).
        edge_map: Maps edge ID → (node_a, node_b) for all internal edges.
            The solver uses this to know which pair of nodes each
            distance constraint connects.
    """

    _signature: str = "RRRRRR"
    edge_map: dict[str, tuple[NodeId, NodeId]] = field(default_factory=dict)

    @property
    def group_class(self) -> int:
        return 2

    @property
    def joint_signature(self) -> str:
        return self._signature

    @classmethod
    def can_form(
        cls,
        internal_node_ids: list[NodeId],
        anchor_node_ids: list[NodeId],
        graph: LinkageGraph,
    ) -> bool:
        """Check if nodes can form a triad.

        Requires:
        - Exactly 2 internal nodes
        - At least 3 anchor nodes
        - Edges connecting internals to anchors (4 total)
        """
        if len(internal_node_ids) != 2:
            return False
        if len(anchor_node_ids) < 3:
            return False

        # Check that internal nodes exist and are connected
        for nid in internal_node_ids:
            if nid not in graph.nodes:
                return False

        # Count edges between internals and anchors
        edge_count = 0
        for iid in internal_node_ids:
            for aid in anchor_node_ids:
                if graph.get_edge_between(iid, aid) is not None:
                    edge_count += 1
        # Also count edge between the two internals
        if (
            graph.get_edge_between(
                internal_node_ids[0],
                internal_node_ids[1],
            )
            is not None
        ):
            edge_count += 1

        return edge_count >= 4


def identify_group_type(
    internal_node_ids: list[NodeId],
    anchor_node_ids: list[NodeId],
    graph: LinkageGraph,
) -> Dyad | Triad | None:
    """Identify which Assur group can be formed from the given elements.

    Tries dyad first (simpler), then triad. Returns an instance
    with the matching signature, or None.

    Args:
        internal_node_ids: Candidate internal nodes.
        anchor_node_ids: Candidate anchor nodes.
        graph: The linkage graph.

    Returns:
        A Dyad or Triad instance, or None if no match found.
    """
    if len(internal_node_ids) == 1:
        for sig in ("RRR", "RRP", "RPR", "PRR", "PP"):
            internal_id = internal_node_ids[0]
            internal_node = graph.nodes.get(internal_id)
            if internal_node is None:
                return None
            if _check_dyad_signature(
                sig,
                internal_id,
                internal_node,
                anchor_node_ids,
                graph,
            ):
                return Dyad(_signature=sig)
        return None

    if len(internal_node_ids) == 2:
        if Triad.can_form(internal_node_ids, anchor_node_ids, graph):
            return Triad()
        return None

    return None


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------
# Old code used per-signature classes (DyadRRR, DyadRRP, etc.).
# These factory functions produce Dyad instances with the right signature,
# preserving the old constructor interface.


def _dyad_factory(signature: str):
    """Create a backwards-compatible alias class for a specific dyad signature."""

    class _CompatDyad(Dyad):
        """Backwards-compatible alias. Use Dyad(_signature=...) for new code."""

        def __init__(self, **kwargs):
            kwargs.setdefault("_signature", signature)
            super().__init__(**kwargs)

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)

    _CompatDyad.__name__ = f"Dyad{signature}"
    _CompatDyad.__qualname__ = f"Dyad{signature}"
    return _CompatDyad


DyadRRR = _dyad_factory("RRR")
DyadRRP = _dyad_factory("RRP")
DyadRPR = _dyad_factory("RPR")
DyadPRR = _dyad_factory("PRR")
DyadPP = _dyad_factory("PP")


# Registry mapping signature strings to group types
DYAD_TYPES: dict[str, type[AssurGroup]] = {
    "RRR": DyadRRR,
    "RRP": DyadRRP,
    "RPR": DyadRPR,
    "PRR": DyadPRR,
    "PP": DyadPP,
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
    if len(internal_node_ids) != 1:
        return None
    internal_id = internal_node_ids[0]
    internal_node = graph.nodes.get(internal_id)
    if internal_node is None:
        return None
    for sig, dyad_cls in DYAD_TYPES.items():
        if _check_dyad_signature(
            sig,
            internal_id,
            internal_node,
            anchor_node_ids,
            graph,
        ):
            return dyad_cls
    return None
