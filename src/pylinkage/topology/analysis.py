"""Topological analysis of planar linkages.

Provides DOF (degree of freedom) computation using Grübler's formula
and related mobility analysis, operating on pure topology (no dimensions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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

    Links are counted from edges and hyperedges. Each edge is one binary
    link. Each hyperedge with k nodes represents a rigid body contributing
    (k-1) binary link equivalents. The ground link is always counted as 1.

    All current joint types (REVOLUTE, PRISMATIC) are 1-DOF.

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
    # Count links:
    # - 1 ground link (always present)
    # - Each edge is one binary link
    # - Each hyperedge with k nodes contributes (k-1) equivalent binary links
    n_links = 1  # ground
    n_links += len(graph.edges)
    for he in graph.hyperedges.values():
        n_links += len(he.nodes) - 1

    # Count joints:
    # Each node is a joint. All current types (REVOLUTE, PRISMATIC) are 1-DOF.
    j1 = len(graph.nodes)  # full joints (1-DOF)
    j2 = 0  # half joints (2-DOF) — none currently supported

    dof = 3 * (n_links - 1) - 2 * j1 - j2

    return MobilityInfo(
        dof=dof,
        num_links=n_links,
        num_full_joints=j1,
        num_half_joints=j2,
    )
