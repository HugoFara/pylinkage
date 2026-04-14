"""Batched simulation for topology-bound populations.

Runs N parameter sets through the same linkage topology using the
numba-compiled solver. Each member shares the same structural arrays
(joint_types, parent_indices, constraint_offsets, solve_order);
only constraints and positions vary.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..solver.simulation import simulate
from ..solver.types import JOINT_CRANK, SolverData


def _build_user_to_solver_map(
    template: SolverData,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Build mapping from user constraints to solver constraints.

    The user-facing constraint vector (from ``get_constraints()``)
    omits topology-fixed values like the crank ``angle_rate``. The solver
    needs the full vector. This function returns:

    1. An index array mapping each user constraint to its solver position.
    2. The template solver constraints (used as base, with user values
       overwritten at the mapped positions).

    Returns:
        Tuple of (user_to_solver_indices, template_constraints).
    """
    user_indices: list[int] = []
    for j in range(template.n_joints):
        offset = int(template.constraint_offsets[j])
        count = int(template.constraint_counts[j])
        jtype = int(template.joint_types[j])

        if jtype == JOINT_CRANK:
            # User exposes only the radius (first constraint).
            # The angle_rate (second constraint) is topology-fixed.
            user_indices.append(offset)
        else:
            for k in range(count):
                user_indices.append(offset + k)

    return (
        np.array(user_indices, dtype=np.intp),
        template.constraints.copy(),
    )


def simulate_batch(
    template: SolverData,
    all_constraints: NDArray[np.float64],
    all_positions: NDArray[np.float64],
    iterations: int,
    dt: float = 1.0,
) -> NDArray[np.float64]:
    """Simulate N parameter sets through the same topology.

    Args:
        template: SolverData defining the shared topology. Its structural
            arrays (joint_types, parent_indices, constraint_offsets,
            solve_order) are reused for every member.
        all_constraints: Shape (n_members, n_user_constraints). One
            user-facing constraint vector per member (as returned by
            ``get_constraints()``).
        all_positions: Shape (n_members, n_joints, 2). Initial positions
            per member.
        iterations: Number of simulation steps per member.
        dt: Time step for crank rotation.

    Returns:
        Trajectory array of shape (n_members, iterations, n_joints, 2).
        If a member's configuration becomes unbuildable, the corresponding
        positions will be NaN.
    """
    n_members = all_constraints.shape[0]
    n_joints = template.n_joints
    trajectories = np.empty(
        (n_members, iterations, n_joints, 2), dtype=np.float64,
    )

    # Shared structural arrays (read-only across members)
    joint_types = template.joint_types
    parent_indices = template.parent_indices
    constraint_offsets = template.constraint_offsets
    solve_order = template.solve_order

    # Build mapping from user constraints → solver constraints.
    # The solver may have extra fixed constraints (e.g. crank angle_rate)
    # that are not in the user vector.
    n_user = all_constraints.shape[1]
    n_solver = template.n_constraints
    need_expand = n_user != n_solver

    if need_expand:
        user_map, base_constraints = _build_user_to_solver_map(template)
    else:
        user_map = None
        base_constraints = None

    for i in range(n_members):
        # Each call to simulate() mutates positions in place, so we
        # must copy per member.
        positions = all_positions[i].copy()

        if need_expand:
            assert base_constraints is not None
            assert user_map is not None
            # Start from template constraints (contains fixed values)
            # and overwrite user-variable positions.
            constraints = base_constraints.copy()
            constraints[user_map] = all_constraints[i]
        else:
            constraints = all_constraints[i].copy()

        trajectories[i] = simulate(
            positions,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            iterations,
            dt,
        )

    return trajectories
