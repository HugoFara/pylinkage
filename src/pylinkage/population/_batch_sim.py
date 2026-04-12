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
from ..solver.types import SolverData


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
        all_constraints: Shape (n_members, n_constraints). One constraint
            vector per member.
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

    for i in range(n_members):
        # Each call to simulate() mutates positions in place, so we
        # must copy per member.
        positions = all_positions[i].copy()
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
