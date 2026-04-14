"""Conversion between Linkage objects and SolverData arrays.

This module provides the bridge between the user-friendly Linkage/Component
API and the numba-optimized numeric arrays used by the solver.

This bridge module exists to keep the solver module pure (only numpy/numba
dependencies) while allowing conversion from Python objects when needed.
"""

from typing import Any

import numpy as np

from ..solver.types import (
    JOINT_CRANK,
    JOINT_REVOLUTE,
    JOINT_STATIC,
    MAX_PARENTS,
    SolverData,
)


def linkage_to_solver_data(linkage: Any) -> SolverData:
    """Convert a Linkage object to numba-compatible SolverData.

    Extracts all joint positions, constraints, and topology into
    contiguous numpy arrays for efficient numba processing.

    Works with:

    - ``simulation.Linkage`` (component/actuator/dyad API)
    - ``mechanism.Mechanism`` (Links + Joints model — dispatched separately)
    - any container with a ``.joints`` or ``.components`` attribute that
      can be classified through :mod:`pylinkage._compat`

    Args:
        linkage: The linkage to convert.

    Returns:
        SolverData containing all numeric arrays needed for simulation.
    """
    # Mechanism has both ``.joints`` and ``.links``; route through the
    # link-aware converter so we can find drivers and constraints.
    if hasattr(linkage, "links") and hasattr(linkage, "joints"):
        return _mechanism_to_solver_data(linkage)

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


def _mechanism_to_solver_data(mechanism: Any) -> SolverData:
    """Convert a ``Mechanism`` (Links + Joints model) to ``SolverData``.

    The Mechanism stores constraints on links rather than on joints, so
    we walk each non-driver joint's link list to find anchors and link
    distances. Driver outputs are typed as ``JOINT_CRANK`` and pull
    their radius/angular_velocity from the owning ``DriverLink``.
    """
    from ..mechanism.joint import GroundJoint, PrismaticJoint, RevoluteJoint
    from ..mechanism.link import ArcDriverLink, DriverLink

    joints = list(mechanism.joints)
    n_joints = len(joints)
    joint_idx: dict[int, int] = {id(j): i for i, j in enumerate(joints)}

    # Map driver output joint id → owning DriverLink for fast lookup.
    driver_for: dict[int, Any] = {}
    for link in mechanism.links:
        if isinstance(link, (DriverLink, ArcDriverLink)) and link.output_joint is not None:
            driver_for[id(link.output_joint)] = link

    positions = np.zeros((n_joints, 2), dtype=np.float64)
    joint_types = np.zeros(n_joints, dtype=np.int32)
    parent_indices = np.full((n_joints, MAX_PARENTS), -1, dtype=np.int32)
    constraints_list: list[float] = []
    offsets: list[int] = []
    counts: list[int] = []

    for i, joint in enumerate(joints):
        offsets.append(len(constraints_list))
        x, y = joint.position
        positions[i, 0] = x if x is not None else 0.0
        positions[i, 1] = y if y is not None else 0.0

        if isinstance(joint, GroundJoint):
            joint_types[i] = JOINT_STATIC
            counts.append(0)
            continue

        driver = driver_for.get(id(joint))
        if driver is not None and driver.motor_joint is not None:
            joint_types[i] = JOINT_CRANK
            anchor_idx = joint_idx.get(id(driver.motor_joint))
            if anchor_idx is not None:
                parent_indices[i, 0] = anchor_idx
            radius = driver.radius if driver.radius is not None else 0.0
            constraints_list.append(radius)
            constraints_list.append(driver.angular_velocity)
            counts.append(2)
            continue

        if isinstance(joint, PrismaticJoint):
            # Prismatic kinematics through the bridge are not yet wired;
            # treat as STATIC so the simulator at least leaves it in place.
            joint_types[i] = JOINT_STATIC
            counts.append(0)
            continue

        # Revolute-like driven joint: pull two anchors + their link distances.
        if isinstance(joint, RevoluteJoint):
            joint_types[i] = JOINT_REVOLUTE
            anchors_seen: set[int] = set()
            anchors: list[tuple[Any, float]] = []
            for link in joint._links:
                for other in link.joints:
                    if other is joint or id(other) in anchors_seen:
                        continue
                    dist = link.get_distance(joint, other)
                    if dist is None:
                        continue
                    anchors.append((other, dist))
                    anchors_seen.add(id(other))
                    if len(anchors) >= 2:
                        break
                if len(anchors) >= 2:
                    break

            for slot, (anchor, dist) in enumerate(anchors[:2]):
                idx = joint_idx.get(id(anchor))
                if idx is not None:
                    parent_indices[i, slot] = idx
                constraints_list.append(dist)
            # Pad to 2 constraints if only one anchor was found.
            while len(constraints_list) - offsets[i] < 2:
                constraints_list.append(0.0)
            counts.append(2)
            continue

        # Fallback: leave fixed.
        joint_types[i] = JOINT_STATIC
        counts.append(0)

    # Solve order: prefer mechanism's own ordering, fall back to drivers
    # then dependents.
    solve_order_list: list[int] = []
    raw_order = getattr(mechanism, "_solve_order", None) or joints
    for joint in raw_order:
        idx = joint_idx.get(id(joint))
        if idx is not None and not isinstance(joint, GroundJoint):
            solve_order_list.append(idx)

    return SolverData(
        positions=positions,
        constraints=np.array(constraints_list, dtype=np.float64),
        joint_types=joint_types,
        parent_indices=parent_indices,
        constraint_offsets=np.array(offsets, dtype=np.int32),
        constraint_counts=np.array(counts, dtype=np.int32),
        solve_order=np.array(solve_order_list, dtype=np.int32),
    )


def solver_data_to_linkage(data: SolverData, linkage: Any) -> None:
    """Update linkage joint positions and velocities from solver data.

    Synchronizes the numeric positions and velocities back to the
    Component objects after simulation.

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


def update_solver_constraints(data: SolverData, linkage: Any) -> None:
    """Update solver constraints from linkage.

    Call after modifying joint constraints via ``set_constraints()`` to keep
    the solver data in sync.

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
            radius = getattr(part, "radius", 0.0)
            omega = getattr(part, "angular_velocity", 0.0)
            data.constraints[offset] = radius if radius is not None else 0.0
            data.constraints[offset + 1] = omega if omega is not None else 0.0

        elif name == "RRRDyad":
            d1 = getattr(part, "distance1", 0.0)
            d2 = getattr(part, "distance2", 0.0)
            data.constraints[offset] = d1 if d1 is not None else 0.0
            data.constraints[offset + 1] = d2 if d2 is not None else 0.0

        elif name == "FixedDyad":
            r = getattr(part, "distance", 0.0)
            a = getattr(part, "angle", 0.0)
            data.constraints[offset] = r if r is not None else 0.0
            data.constraints[offset + 1] = a if a is not None else 0.0

        elif name == "RRPDyad":
            rr = getattr(part, "distance1", 0.0)
            data.constraints[offset] = rr if rr is not None else 0.0


def update_solver_positions(data: SolverData, linkage: Any) -> None:
    """Update solver positions from linkage.

    Call after modifying joint positions via ``set_coord()`` to keep the
    solver data in sync.

    Args:
        data: SolverData to update.
        linkage: Linkage containing updated positions.
    """
    from .._compat import get_parts

    parts = get_parts(linkage)
    for i, part in enumerate(parts):
        data.positions[i, 0] = part.x if part.x is not None else 0.0
        data.positions[i, 1] = part.y if part.y is not None else 0.0
