"""Analysis results and utilities for mechanism structural analysis.

This module provides data classes for representing the results of
various structural analyses on mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MobilityResult:
    """Result of mobility analysis for a mechanism.

    Mobility analysis uses Gruebler's equation (or Chebyshev-Gruebler-Kutzbach)
    to determine the degree of freedom of a planar mechanism:

    DOF = 3(n - 1) - 2*j1 - j2

    where:
    - n = number of links (including ground)
    - j1 = number of 1-DOF joints (revolute, prismatic)
    - j2 = number of 2-DOF joints (roll-slide, cam-follower)

    Attributes:
        degree_of_freedom: Computed DOF of the mechanism.
        num_links: Number of links in the mechanism.
        num_joints: Total number of joints.
        num_1dof_joints: Number of 1-DOF joints (revolute, prismatic).
        num_2dof_joints: Number of 2-DOF joints.
        num_ground_joints: Number of joints fixed to ground.
        num_driver_joints: Number of driver/motor joints.
        is_determinate: True if DOF equals number of drivers.
        is_overconstrained: True if DOF < 0 (statically indeterminate).
        is_underconstrained: True if DOF > number of drivers.
        warnings: List of warning messages from analysis.
    """

    degree_of_freedom: int
    num_links: int
    num_joints: int
    num_1dof_joints: int = 0
    num_2dof_joints: int = 0
    num_ground_joints: int = 0
    num_driver_joints: int = 0
    is_determinate: bool = False
    is_overconstrained: bool = False
    is_underconstrained: bool = False
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and compute derived properties."""
        # DOF should match drivers for determinate mechanisms
        self.is_determinate = self.degree_of_freedom == self.num_driver_joints
        self.is_overconstrained = self.degree_of_freedom < 0
        self.is_underconstrained = self.degree_of_freedom > self.num_driver_joints

    @property
    def is_valid_mechanism(self) -> bool:
        """Check if this represents a valid, solvable mechanism.

        Returns:
            True if mechanism is determinate and not overconstrained.
        """
        return self.is_determinate and not self.is_overconstrained

    @property
    def mobility_status(self) -> str:
        """Get a human-readable status string.

        Returns:
            String describing the mobility status.
        """
        if self.is_overconstrained:
            return "overconstrained"
        elif self.is_underconstrained:
            return "underconstrained"
        elif self.is_determinate:
            return "determinate"
        else:
            return "unknown"


@dataclass
class StructuralAnalysis:
    """Complete structural analysis of a mechanism.

    Combines decomposition and mobility analysis into a single result.

    Attributes:
        mobility: Mobility analysis result.
        num_assur_groups: Number of Assur groups in decomposition.
        group_signatures: Joint signatures of each group (e.g., ["RRR", "RRR"]).
        validation_messages: Messages from structural validation.
        is_valid: True if no validation issues.
    """

    mobility: MobilityResult
    num_assur_groups: int = 0
    group_signatures: list[str] = field(default_factory=list)
    validation_messages: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if the structure is valid.

        Returns:
            True if no validation messages and mobility is valid.
        """
        return len(self.validation_messages) == 0 and self.mobility.is_valid_mechanism
