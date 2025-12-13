"""Pure-numba simulation loop for linkage mechanisms.

This module provides the core simulation functions that operate on
numeric arrays, avoiding Python object overhead for maximum performance.
"""


import numpy as np
from numba import njit

from .joints import solve_crank, solve_fixed, solve_linear, solve_revolute
from .types import (
    JOINT_CRANK,
    JOINT_FIXED,
    JOINT_LINEAR,
    JOINT_REVOLUTE,
    JOINT_STATIC,
)


@njit(cache=True)  # type: ignore[untyped-decorator]
def step_single(
    positions: np.ndarray,
    constraints: np.ndarray,
    joint_types: np.ndarray,
    parent_indices: np.ndarray,
    constraint_offsets: np.ndarray,
    solve_order: np.ndarray,
    dt: float,
) -> None:
    """Perform one simulation step, updating positions in-place.

    Args:
        positions: Joint positions, shape (n_joints, 2). Modified in-place.
        constraints: Flat array of constraint values.
        joint_types: Joint type code for each joint, shape (n_joints,).
        parent_indices: Parent joint indices, shape (n_joints, max_parents).
        constraint_offsets: Start index in constraints for each joint.
        solve_order: Indices of joints to solve, in order.
        dt: Time step for crank rotation.
    """
    for i in range(len(solve_order)):
        joint_idx = solve_order[i]
        joint_type = joint_types[joint_idx]
        offset = constraint_offsets[joint_idx]

        if joint_type == JOINT_STATIC:
            # Static joints never move
            pass

        elif joint_type == JOINT_CRANK:
            p0_idx = parent_indices[joint_idx, 0]
            new_x, new_y = solve_crank(
                positions[joint_idx, 0],
                positions[joint_idx, 1],
                positions[p0_idx, 0],
                positions[p0_idx, 1],
                constraints[offset],      # radius
                constraints[offset + 1],  # angle_rate
                dt,
            )
            positions[joint_idx, 0] = new_x
            positions[joint_idx, 1] = new_y

        elif joint_type == JOINT_REVOLUTE:
            p0_idx = parent_indices[joint_idx, 0]
            p1_idx = parent_indices[joint_idx, 1]
            new_x, new_y = solve_revolute(
                positions[joint_idx, 0],
                positions[joint_idx, 1],
                positions[p0_idx, 0],
                positions[p0_idx, 1],
                constraints[offset],      # r0
                positions[p1_idx, 0],
                positions[p1_idx, 1],
                constraints[offset + 1],  # r1
            )
            positions[joint_idx, 0] = new_x
            positions[joint_idx, 1] = new_y

        elif joint_type == JOINT_FIXED:
            p0_idx = parent_indices[joint_idx, 0]
            p1_idx = parent_indices[joint_idx, 1]
            new_x, new_y = solve_fixed(
                positions[p0_idx, 0],
                positions[p0_idx, 1],
                positions[p1_idx, 0],
                positions[p1_idx, 1],
                constraints[offset],      # radius
                constraints[offset + 1],  # angle
            )
            positions[joint_idx, 0] = new_x
            positions[joint_idx, 1] = new_y

        elif joint_type == JOINT_LINEAR:
            p0_idx = parent_indices[joint_idx, 0]
            p1_idx = parent_indices[joint_idx, 1]
            p2_idx = parent_indices[joint_idx, 2]
            new_x, new_y = solve_linear(
                positions[joint_idx, 0],
                positions[joint_idx, 1],
                positions[p0_idx, 0],
                positions[p0_idx, 1],
                constraints[offset],  # radius
                positions[p1_idx, 0],
                positions[p1_idx, 1],
                positions[p2_idx, 0],
                positions[p2_idx, 1],
            )
            positions[joint_idx, 0] = new_x
            positions[joint_idx, 1] = new_y


@njit(cache=True)  # type: ignore[untyped-decorator]
def simulate(
    positions: np.ndarray,
    constraints: np.ndarray,
    joint_types: np.ndarray,
    parent_indices: np.ndarray,
    constraint_offsets: np.ndarray,
    solve_order: np.ndarray,
    iterations: int,
    dt: float,
) -> np.ndarray:
    """Run full simulation and return trajectory.

    This is the main entry point for numba-optimized simulation.
    All arrays are processed in a tight loop with no Python overhead.

    Args:
        positions: Initial joint positions, shape (n_joints, 2).
            Will be modified to contain final positions.
        constraints: Flat array of constraint values.
        joint_types: Joint type code for each joint, shape (n_joints,).
        parent_indices: Parent joint indices, shape (n_joints, max_parents).
        constraint_offsets: Start index in constraints for each joint.
        solve_order: Indices of joints to solve, in order.
        iterations: Number of simulation steps to run.
        dt: Time step for crank rotation.

    Returns:
        Trajectory array of shape (iterations, n_joints, 2) containing
        all joint positions at each step. If any position becomes NaN
        (unbuildable configuration), subsequent positions may also be NaN.
    """
    n_joints = positions.shape[0]
    trajectory = np.empty((iterations, n_joints, 2), dtype=np.float64)

    for step in range(iterations):
        step_single(
            positions,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            dt,
        )
        # Copy positions to trajectory
        for j in range(n_joints):
            trajectory[step, j, 0] = positions[j, 0]
            trajectory[step, j, 1] = positions[j, 1]

    return trajectory


@njit(cache=True)  # type: ignore[untyped-decorator]
def has_nan_positions(trajectory: np.ndarray) -> bool:
    """Check if trajectory contains any NaN positions.

    Args:
        trajectory: Array of shape (iterations, n_joints, 2).

    Returns:
        True if any position is NaN (indicating unbuildable configuration).
    """
    for i in range(trajectory.shape[0]):
        for j in range(trajectory.shape[1]):
            if np.isnan(trajectory[i, j, 0]) or np.isnan(trajectory[i, j, 1]):
                return True
    return False


@njit(cache=True)  # type: ignore[untyped-decorator]
def first_nan_step(trajectory: np.ndarray) -> int:
    """Find the first step with NaN positions.

    Args:
        trajectory: Array of shape (iterations, n_joints, 2).

    Returns:
        Index of first step with NaN, or -1 if no NaN found.
    """
    for i in range(trajectory.shape[0]):
        for j in range(trajectory.shape[1]):
            if np.isnan(trajectory[i, j, 0]) or np.isnan(trajectory[i, j, 1]):
                return i
    return -1
