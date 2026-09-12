"""Topological analysis of planar linkages.

Provides DOF (degree of freedom) computation using Grübler's formula
and related mobility analysis, operating on pure topology (no dimensions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._types import JointType, NodeRole

if TYPE_CHECKING:
    from ..hypergraph.graph import HypergraphLinkage


@dataclass(frozen=True)
class MobilityInfo:
    """Result of Grübler mobility analysis.

    Attributes:
        dof: Degree of freedom of the mechanism.
        num_links: Number of links (including ground).
        num_full_joints: Number of 1-DOF joints (revolute, prismatic).
        num_half_joints: Number of 2-DOF joints (higher pairs).
    """

    dof: int
    num_links: int
    num_full_joints: int
    num_half_joints: int = 0


def compute_dof(graph: HypergraphLinkage) -> int:
    """Compute the degree of freedom of a planar linkage using Grübler's formula.

    DOF = 3*(n - 1) - 2*j1 - j2

    where:
    - n = number of links (including ground)
    - j1 = number of 1-DOF joints (revolute, prismatic)
    - j2 = number of 2-DOF joints (higher pairs)

    Links are the rigid bodies of the graph: the ground (every
    ``GROUND`` node, or an implicit link when there is none), one body
    per edge and one per hyperedge, except that bodies sharing two or
    more nodes are pinned together and count as one. A hyperedge over a
    triangle whose sides are also edges is therefore one link, as is an
    edge between two ground nodes.

    Joints are counted per node: a node shared by ``k`` bodies is
    ``k - 1`` joints, so a coupler point that belongs to a single body
    is not a joint and a pin shared by three links is two. A
    ``PRISMATIC`` node is a slider block of its own with one prismatic
    joint to the hyperedge it slides along. All joints are 1-DOF.

    Three edges closing a triangle are counted as three bars pinned
    together rather than one body; the DOF is the same either way.

    Args:
        graph: A HypergraphLinkage (topology only, no dimensions needed).

    Returns:
        The computed degree of freedom. Typical values:
        - 1: Single-input mechanism (four-bar, slider-crank)
        - 0: Rigid structure (truss)
        - <0: Over-constrained (statically indeterminate)

    Example:
        >>> from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole
        >>> # Four-bar linkage: 4 nodes, 3 driven edges + ground = 4 links, 4 joints
        >>> hg = HypergraphLinkage()
        >>> hg.add_node(Node("A", role=NodeRole.GROUND))
        >>> hg.add_node(Node("B", role=NodeRole.DRIVER))
        >>> hg.add_node(Node("C", role=NodeRole.DRIVEN))
        >>> hg.add_node(Node("D", role=NodeRole.GROUND))
        >>> hg.add_edge(Edge("AB", "A", "B"))
        >>> hg.add_edge(Edge("BC", "B", "C"))
        >>> hg.add_edge(Edge("CD", "C", "D"))
        >>> compute_dof(hg)
        1
    """
    return compute_mobility(graph).dof


def compute_mobility(graph: HypergraphLinkage) -> MobilityInfo:
    """Compute full mobility analysis of a planar linkage.

    See :func:`compute_dof` for the formula and link/joint counting rules.

    Args:
        graph: A HypergraphLinkage (topology only).

    Returns:
        MobilityInfo with DOF, link count, and joint counts.
    """
    bodies, n_prismatic = _rigid_bodies(graph)
    has_ground = any(n.role == NodeRole.GROUND for n in graph.nodes.values())
    n_links = len(bodies) + (0 if has_ground else 1)

    # A node shared by k bodies is k - 1 one-DOF joints; a slider adds
    # the prismatic joint to its guide.
    j1 = n_prismatic
    for node_id in graph.nodes:
        k = sum(1 for body in bodies if node_id in body)
        j1 += max(k - 1, 0)
    j2 = 0  # half joints (2-DOF) — none currently supported

    dof = 3 * (n_links - 1) - 2 * j1 - j2

    return MobilityInfo(
        dof=dof,
        num_links=n_links,
        num_full_joints=j1,
        num_half_joints=j2,
    )


def _rigid_bodies(graph: HypergraphLinkage) -> tuple[list[frozenset[str]], int]:
    """Partition the graph into rigid bodies.

    Returns the bodies as node sets, and the number of prismatic
    joints. Bodies that share two or more nodes are merged: two links
    pinned at two points cannot move relative to each other.
    """
    prismatic = {
        n.id for n in graph.nodes.values() if n.joint_type == JointType.PRISMATIC
    }
    bodies: list[frozenset[str]] = []
    ground = frozenset(n.id for n in graph.nodes.values() if n.role == NodeRole.GROUND)
    if ground:
        bodies.append(ground)
    bodies.extend(frozenset((e.source, e.target)) for e in graph.edges.values())
    n_prismatic = 0
    for he in graph.hyperedges.values():
        sliders = [n for n in he.nodes if n in prismatic]
        guide = frozenset(n for n in he.nodes if n not in prismatic)
        if guide:
            bodies.append(guide)
        # Each slider is a block of its own, joined to the guide by a
        # prismatic joint.
        for slider in sliders:
            bodies.append(frozenset((slider,)))
            n_prismatic += 1

    merged = True
    while merged:
        merged = False
        result: list[frozenset[str]] = []
        for body in bodies:
            for i, other in enumerate(result):
                if len(body & other) >= 2:
                    result[i] = other | body
                    merged = True
                    break
            else:
                result.append(body)
        bodies = result
    return bodies, n_prismatic
