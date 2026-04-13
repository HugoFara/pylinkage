"""Conversion between Linkage objects and SolverData arrays.

This module provides the bridge between the user-friendly Linkage/Joint
API and the numba-optimized numeric arrays used by the solver.

This bridge module exists to keep the solver module pure (only numpy/numba
dependencies) while allowing conversion from Python objects when needed.
"""

from typing import TYPE_CHECKING

import numpy as np

from ..components._base import Component
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

    Works with both legacy ``Linkage`` (joints API) and modern
    ``SimLinkage`` (components API).

    Args:
        linkage: The linkage to convert.

    Returns:
        SolverData containing all numeric arrays needed for simulation.

    Raises:
        ValueError: If joint references cannot be resolved.
    """
    # Dispatch to the modern converter if this is a SimLinkage
    if hasattr(linkage, "components") and not hasattr(linkage, "joints"):
        return _sim_linkage_to_solver_data(linkage)

    # Import here to avoid circular imports
    from ..joints.crank import Crank
    from ..joints.fixed import Fixed
    from ..joints.joint import Static
    from ..joints.prismatic import Prismatic
    from ..joints.revolute import Revolute

    # Build index map from joint identity to array index
    # First, collect all joints including implicit Static joints
    all_joints: list[Component] = list(linkage.joints)
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


def _sim_linkage_to_solver_data(linkage: "object") -> SolverData:
    """Convert a modern SimLinkage to SolverData.

    Maps the component/actuator/dyad API to the solver's numeric arrays.
    """
    from .._compat import get_parts, is_driver, is_dyad, is_ground

    parts = get_parts(linkage)

    # Build index map — includes resolving AnchorProxy references
    part_to_idx: dict[int, int] = {}
    for i, part in enumerate(parts):
        part_to_idx[id(part)] = i

    n_parts = len(parts)
    positions = np.zeros((n_parts, 2), dtype=np.float64)
    joint_types = np.zeros(n_parts, dtype=np.int32)
    parent_indices = np.full((n_parts, MAX_PARENTS), -1, dtype=np.int32)
    constraints_list: list[float] = []
    offsets: list[int] = []
    counts: list[int] = []

    for i, part in enumerate(parts):
        offsets.append(len(constraints_list))
        positions[i, 0] = part.x if part.x is not None else 0.0
        positions[i, 1] = part.y if part.y is not None else 0.0

        if is_ground(part):
            joint_types[i] = JOINT_STATIC
            counts.append(0)

        elif is_driver(part):
            joint_types[i] = JOINT_CRANK
            anchor = getattr(part, "anchor", None)
            if anchor is not None:
                idx = part_to_idx.get(id(anchor))
                if idx is not None:
                    parent_indices[i, 0] = idx
            radius = getattr(part, "radius", 0.0)
            omega = getattr(part, "angular_velocity", 0.0)
            constraints_list.append(radius)
            constraints_list.append(omega)
            counts.append(2)

        elif is_dyad(part):
            joint_types[i] = JOINT_REVOLUTE
            for p_idx, attr in enumerate(("anchor1", "anchor2")):
                parent = getattr(part, attr, None)
                if parent is None:
                    continue
                # Resolve AnchorProxy
                actual = getattr(parent, "_parent", parent)
                idx = part_to_idx.get(id(actual))
                if idx is not None:
                    parent_indices[i, p_idx] = idx
            d1 = getattr(part, "distance1", 0.0)
            d2 = getattr(part, "distance2", 0.0)
            constraints_list.append(d1)
            constraints_list.append(d2)
            counts.append(2)

        else:
            joint_types[i] = JOINT_STATIC
            counts.append(0)

    # Build solve order — use the linkage's own order if available
    solve_order_list: list[int] = []
    # Trigger lazy computation if needed
    if hasattr(linkage, "_find_solve_order") and not hasattr(linkage, "_solve_order"):
        linkage._find_solve_order()
    solve_order = getattr(linkage, "_solve_order", None)
    if solve_order is not None:
        for part in solve_order:
            idx = part_to_idx.get(id(part))
            if idx is not None:
                solve_order_list.append(idx)
    else:
        # Fallback: all non-ground parts in order
        for i, part in enumerate(parts):
            if not is_ground(part):
                solve_order_list.append(i)

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
    """Update linkage joint positions and velocities from solver data.

    This synchronizes the numeric positions and velocities back to the
    Joint objects after simulation.

    Works with both legacy ``Linkage`` and modern ``SimLinkage``.

    Args:
        data: SolverData containing updated positions and optionally velocities.
        linkage: Linkage to update.
    """
    from .._compat import get_parts

    parts = get_parts(linkage)
    for i, part in enumerate(parts):
        part.x = float(data.positions[i, 0])
        part.y = float(data.positions[i, 1])

        if data.velocities is not None:
            part.velocity = (
                float(data.velocities[i, 0]),
                float(data.velocities[i, 1]),
            )

        if data.accelerations is not None:
            part.acceleration = (
                float(data.accelerations[i, 0]),
                float(data.accelerations[i, 1]),
            )


def update_solver_constraints(data: SolverData, linkage: "Linkage") -> None:
    """Update solver constraints from linkage.

    Call this after modifying joint constraints via set_constraints()
    to keep the solver data in sync.

    Works with both legacy ``Linkage`` and modern ``SimLinkage``.

    Args:
        data: SolverData to update.
        linkage: Linkage containing updated constraints.
    """
    from .._compat import get_parts, is_driver

    parts = get_parts(linkage)
    for i, part in enumerate(parts):
        offset = data.constraint_offsets[i]
        name = type(part).__name__

        if is_driver(part):
            radius = getattr(part, "radius", None) or getattr(part, "r", None)
            omega = getattr(part, "angular_velocity", None) or getattr(
                part, "angle", None
            )
            data.constraints[offset] = radius if radius is not None else 0.0
            data.constraints[offset + 1] = omega if omega is not None else 0.0

        elif name in ("Revolute", "RRRDyad"):
            d1 = getattr(part, "distance1", None) or getattr(part, "r0", None)
            d2 = getattr(part, "distance2", None) or getattr(part, "r1", None)
            data.constraints[offset] = d1 if d1 is not None else 0.0
            data.constraints[offset + 1] = d2 if d2 is not None else 0.0

        elif name in ("Fixed", "FixedDyad"):
            r = getattr(part, "distance", None) or getattr(part, "r", None)
            a = getattr(part, "angle", None)
            data.constraints[offset] = r if r is not None else 0.0
            data.constraints[offset + 1] = a if a is not None else 0.0

        elif name in ("Prismatic", "Linear", "RRPDyad"):
            rr = getattr(part, "revolute_radius", None) or getattr(
                part, "distance1", None
            )
            data.constraints[offset] = rr if rr is not None else 0.0


def update_solver_positions(data: SolverData, linkage: "Linkage") -> None:
    """Update solver positions from linkage.

    Call this after modifying joint positions via set_coord()
    to keep the solver data in sync.

    Works with both legacy ``Linkage`` and modern ``SimLinkage``.

    Args:
        data: SolverData to update.
        linkage: Linkage containing updated positions.
    """
    from .._compat import get_parts

    parts = get_parts(linkage)
    for i, part in enumerate(parts):
        data.positions[i, 0] = part.x if part.x is not None else 0.0
        data.positions[i, 1] = part.y if part.y is not None else 0.0
