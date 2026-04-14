"""Type definitions for topology-aware synthesis.

Extends the four-bar-specific types in ``_types.py`` with general
N-bar solution structures used by six-bar, generalized, and
multi-topology synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._types import Point2D

if TYPE_CHECKING:
    from ..simulation import Linkage
    from ..topology.catalog import CatalogEntry
    from .core import Dyad

# A partition of precision point indices across Assur groups.
# E.g., ((0, 1, 2), (3, 4)) assigns points 0-2 to group 0 and 3-4 to group 1.
PointPartition = tuple[tuple[int, ...], ...]


@dataclass
class GroupSynthesisResult:
    """Result of synthesizing a single Assur group in the decomposition chain.

    Attributes:
        group_index: Index in the decomposition order.
        group_signature: Joint signature (e.g., "RRR").
        dyads: Burmester dyad pair if group is a dyad, None for triads.
        joint_positions: Computed positions for this group's internal nodes.
        precision_indices: Which precision points were assigned to this group.
        residual: Synthesis error (0.0 for exact Burmester solutions).
    """

    group_index: int
    group_signature: str
    dyads: tuple[Dyad, Dyad] | None = None
    joint_positions: dict[str, Point2D] = field(default_factory=dict)
    precision_indices: tuple[int, ...] = ()
    residual: float = 0.0


@dataclass
class NBarSolution:
    """A general N-bar linkage solution from topology-aware synthesis.

    Generalizes FourBarSolution to arbitrary topologies. Instead of
    named pivots (A, B, C, D), stores a dict of joint positions keyed
    by node ID, plus the topology graph connectivity.

    Attributes:
        topology_id: ID from the topology catalog (e.g., "watt").
        joint_positions: Map from node ID to (x, y) position.
        link_lengths: Map from edge ID to link length.
        group_results: Per-group synthesis results in decomposition order.
        coupler_node: Node ID of the traced coupler point (if any).
        coupler_point: World-frame position of the traced point.
    """

    topology_id: str
    joint_positions: dict[str, Point2D]
    link_lengths: dict[str, float]
    group_results: list[GroupSynthesisResult] = field(default_factory=list)
    coupler_node: str | None = None
    coupler_point: Point2D | None = None


@dataclass
class QualityMetrics:
    """Quality metrics for ranking synthesis solutions.

    All metrics are computed by simulating the synthesized linkage
    and evaluating its kinematic properties.

    Attributes:
        path_accuracy: RMS error of coupler path vs. precision points
            (lower is better).
        min_transmission_angle: Worst transmission angle in degrees
            over the motion cycle (higher is better, ideal is 90).
        link_ratio: Max/min link length ratio (lower is better,
            ideal is 1.0).
        compactness: Bounding box area of the mechanism trajectory
            (lower is better).
        num_links: Number of links (fewer is simpler).
        is_grashof: Whether the driving loop is Grashof (full crank
            rotation possible).
        overall_score: Weighted composite score (lower is better).
    """

    path_accuracy: float = float("inf")
    min_transmission_angle: float = 0.0
    link_ratio: float = float("inf")
    compactness: float = float("inf")
    num_links: int = 4
    is_grashof: bool = False
    overall_score: float = float("inf")


@dataclass
class TopologySolution:
    """A ranked solution from multi-topology synthesis.

    Wraps an NBarSolution with its quality metrics, the converted
    Linkage object for simulation, and the catalog entry identifying
    which topology was used.

    Attributes:
        solution: The raw N-bar synthesis solution.
        linkage: Converted Linkage object ready for simulation.
        topology_entry: Catalog entry for the topology used.
        metrics: Computed quality metrics.
    """

    solution: NBarSolution
    linkage: Linkage
    topology_entry: CatalogEntry
    metrics: QualityMetrics
