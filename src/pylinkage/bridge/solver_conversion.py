"""Conversion between Linkage objects and SolverData arrays.

This module provides the bridge between the user-friendly Linkage/Joint
API and the numba-optimized numeric arrays used by the solver.

This bridge module exists to keep the solver module pure (only numpy/numba
dependencies) while allowing conversion from Python objects when needed.
"""

from typing import TYPE_CHECKING

import numpy as np

from ..joints.joint import Joint
from ..solver.types import (
    JOINT_CRANK,
    JOINT_FIXED,
    JOINT_LINEAR,
    JOINT_REVOLUTE,
    JOINT_STATIC,
    MAX_PARENTS,
    SolverData,
)

if TYPE_CHECKING:
    from ..linkage import Linkage


def linkage_to_solver_data(linkage: "Linkage") -> SolverData:
    """Convert a Linkage object to numba-compatible SolverData.

    This extracts all joint positions, constraints, and topology into
    contiguous numpy arrays for efficient numba processing.

    Args:
        linkage: The linkage to convert.

    Returns:
        SolverData containing all numeric arrays needed for simulation.

    Raises:
        ValueError: If joint references cannot be resolved.
    """
    # Import here to avoid circular imports
    from ..joints import Crank, Fixed, Prismatic, Revolute
    from ..joints.joint import Static

    # Build index map from joint identity to array index
    # First, collect all joints including implicit Static joints
    all_joints: list[Joint] = list(linkage.joints)
    joint_to_idx: dict[int, int] = {}
    for i, joint in enumerate(linkage.joints):
        joint_to_idx[id(joint)] = i

    # Find implicit Static joints (referenced but not in linkage.joints)
    def add_implicit_joint(parent_joint: Joint) -> None:
        if id(parent_joint) not in joint_to_idx:
            idx = len(all_joints)
            joint_to_idx[id(parent_joint)] = idx
            all_joints.append(parent_joint)

    for joint in linkage.joints:
        if hasattr(joint, "joint0") and joint.joint0 is not None:
            add_implicit_joint(joint.joint0)
        if hasattr(joint, "joint1") and joint.joint1 is not None:
            add_implicit_joint(joint.joint1)
        if hasattr(joint, "joint2") and joint.joint2 is not None:
            add_implicit_joint(joint.joint2)

    n_joints = len(all_joints)

    # Initialize arrays
    positions = np.zeros((n_joints, 2), dtype=np.float64)
    joint_types = np.zeros(n_joints, dtype=np.int32)
    parent_indices = np.full((n_joints, MAX_PARENTS), -1, dtype=np.int32)
    constraints_list: list[float] = []
    offsets: list[int] = []
    counts: list[int] = []

    for i, joint in enumerate(all_joints):
        offsets.append(len(constraints_list))
        positions[i, 0] = joint.x if joint.x is not None else 0.0
        positions[i, 1] = joint.y if joint.y is not None else 0.0

        if isinstance(joint, Static) and not isinstance(joint, Crank):
            # Base Static class (not Crank which inherits from Joint)
            joint_types[i] = JOINT_STATIC
            counts.append(0)

        elif isinstance(joint, Crank):
            joint_types[i] = JOINT_CRANK
            # Parent: joint0 (anchor)
            if joint.joint0 is not None:
                parent_idx = joint_to_idx.get(id(joint.joint0))
                if parent_idx is not None:
                    parent_indices[i, 0] = parent_idx
            # Constraints: radius, angle_rate
            constraints_list.append(joint.r if joint.r is not None else 0.0)
            constraints_list.append(joint.angle if joint.angle is not None else 0.0)
            counts.append(2)

        elif isinstance(joint, Revolute):
            joint_types[i] = JOINT_REVOLUTE
            # Parents: joint0, joint1
            if joint.joint0 is not None:
                parent_idx = joint_to_idx.get(id(joint.joint0))
                if parent_idx is not None:
                    parent_indices[i, 0] = parent_idx
            if joint.joint1 is not None:
                parent_idx = joint_to_idx.get(id(joint.joint1))
                if parent_idx is not None:
                    parent_indices[i, 1] = parent_idx
            # Constraints: r0, r1
            constraints_list.append(joint.r0 if joint.r0 is not None else 0.0)
            constraints_list.append(joint.r1 if joint.r1 is not None else 0.0)
            counts.append(2)

        elif isinstance(joint, Fixed):
            joint_types[i] = JOINT_FIXED
            # Parents: joint0, joint1
            if joint.joint0 is not None:
                parent_idx = joint_to_idx.get(id(joint.joint0))
                if parent_idx is not None:
                    parent_indices[i, 0] = parent_idx
            if joint.joint1 is not None:
                parent_idx = joint_to_idx.get(id(joint.joint1))
                if parent_idx is not None:
                    parent_indices[i, 1] = parent_idx
            # Constraints: radius, angle
            constraints_list.append(joint.r if joint.r is not None else 0.0)
            constraints_list.append(joint.angle if joint.angle is not None else 0.0)
            counts.append(2)

        elif isinstance(joint, Prismatic):
            joint_types[i] = JOINT_LINEAR
            # Parents: joint0 (circle center), joint1, joint2 (line definition)
            if joint.joint0 is not None:
                parent_idx = joint_to_idx.get(id(joint.joint0))
                if parent_idx is not None:
                    parent_indices[i, 0] = parent_idx
            if joint.joint1 is not None:
                parent_idx = joint_to_idx.get(id(joint.joint1))
                if parent_idx is not None:
                    parent_indices[i, 1] = parent_idx
            if joint.joint2 is not None:
                parent_idx = joint_to_idx.get(id(joint.joint2))
                if parent_idx is not None:
                    parent_indices[i, 2] = parent_idx
            # Constraint: revolute_radius
            constraints_list.append(
                joint.revolute_radius if joint.revolute_radius is not None else 0.0
            )
            counts.append(1)

        else:
            # Fallback: treat as static
            joint_types[i] = JOINT_STATIC
            counts.append(0)

    # Build solve order array
    solve_order_list: list[int] = []
    for joint in linkage._solve_order:
        joint_idx = joint_to_idx.get(id(joint))
        if joint_idx is not None:
            solve_order_list.append(joint_idx)

    return SolverData(
        positions=positions,
        constraints=np.array(constraints_list, dtype=np.float64),
        joint_types=joint_types,
        parent_indices=parent_indices,
        constraint_offsets=np.array(offsets, dtype=np.int32),
        constraint_counts=np.array(counts, dtype=np.int32),
        solve_order=np.array(solve_order_list, dtype=np.int32),
    )


def solver_data_to_linkage(data: SolverData, linkage: "Linkage") -> None:
    """Update linkage joint positions from solver data.

    This synchronizes the numeric positions back to the Joint objects
    after simulation.

    Args:
        data: SolverData containing updated positions.
        linkage: Linkage to update.
    """
    for i, joint in enumerate(linkage.joints):
        joint.x = float(data.positions[i, 0])
        joint.y = float(data.positions[i, 1])


def update_solver_constraints(data: SolverData, linkage: "Linkage") -> None:
    """Update solver constraints from linkage.

    Call this after modifying joint constraints via set_constraints()
    to keep the solver data in sync.

    Args:
        data: SolverData to update.
        linkage: Linkage containing updated constraints.
    """
    from ..joints import Crank, Fixed, Prismatic, Revolute

    for i, joint in enumerate(linkage.joints):
        offset = data.constraint_offsets[i]

        if isinstance(joint, Crank):
            data.constraints[offset] = joint.r if joint.r is not None else 0.0
            data.constraints[offset + 1] = (
                joint.angle if joint.angle is not None else 0.0
            )

        elif isinstance(joint, Revolute):
            data.constraints[offset] = joint.r0 if joint.r0 is not None else 0.0
            data.constraints[offset + 1] = joint.r1 if joint.r1 is not None else 0.0

        elif isinstance(joint, Fixed):
            data.constraints[offset] = joint.r if joint.r is not None else 0.0
            data.constraints[offset + 1] = (
                joint.angle if joint.angle is not None else 0.0
            )

        elif isinstance(joint, Prismatic):
            data.constraints[offset] = (
                joint.revolute_radius if joint.revolute_radius is not None else 0.0
            )


def update_solver_positions(data: SolverData, linkage: "Linkage") -> None:
    """Update solver positions from linkage.

    Call this after modifying joint positions via set_coord()
    to keep the solver data in sync.

    Args:
        data: SolverData to update.
        linkage: Linkage containing updated positions.
    """
    for i, joint in enumerate(linkage.joints):
        data.positions[i, 0] = joint.x if joint.x is not None else 0.0
        data.positions[i, 1] = joint.y if joint.y is not None else 0.0
