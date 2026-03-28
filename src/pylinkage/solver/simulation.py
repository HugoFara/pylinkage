"""Pure-numba simulation loop for linkage mechanisms.

This module provides the core simulation functions that operate on
numeric arrays, avoiding Python object overhead for maximum performance.
"""

import numpy as np

from .._numba_compat import njit
from .acceleration import (
    solve_crank_acceleration,
    solve_fixed_acceleration,
    solve_prismatic_acceleration,
    solve_revolute_acceleration,
)
from .joints import solve_crank, solve_fixed, solve_linear, solve_revolute
from .types import (
    JOINT_CRANK,
    JOINT_FIXED,
    JOINT_LINEAR,
    JOINT_REVOLUTE,
    JOINT_STATIC,
)
from .velocity import (
    solve_crank_velocity,
    solve_fixed_velocity,
    solve_prismatic_velocity,
    solve_revolute_velocity,
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
                constraints[offset],  # radius
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
                constraints[offset],  # r0
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
                constraints[offset],  # radius
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


@njit(cache=True)  # type: ignore[untyped-decorator]
def step_single_velocity(
    positions: np.ndarray,
    velocities: np.ndarray,
    constraints: np.ndarray,
    joint_types: np.ndarray,
    parent_indices: np.ndarray,
    constraint_offsets: np.ndarray,
    solve_order: np.ndarray,
    omega_values: np.ndarray,
    crank_indices: np.ndarray,
) -> None:
    """Compute velocities for all joints after positions are solved.

    Must be called after step_single() has updated positions.

    Args:
        positions: Joint positions, shape (n_joints, 2).
        velocities: Output velocities, shape (n_joints, 2). Modified in-place.
        constraints: Flat array of constraint values.
        joint_types: Joint type code for each joint, shape (n_joints,).
        parent_indices: Parent joint indices, shape (n_joints, max_parents).
        constraint_offsets: Start index in constraints for each joint.
        solve_order: Indices of joints to solve, in order.
        omega_values: Angular velocities for crank joints (rad/s).
        crank_indices: Indices of crank joints in joints array.
    """
    # Build a mapping from joint index to crank omega index
    # For non-cranks, omega is not used
    crank_omega_map = np.full(len(joint_types), -1, dtype=np.int32)
    for i in range(len(crank_indices)):
        crank_omega_map[crank_indices[i]] = i

    for i in range(len(solve_order)):
        joint_idx = solve_order[i]
        joint_type = joint_types[joint_idx]
        offset = constraint_offsets[joint_idx]

        if joint_type == JOINT_STATIC:
            # Static joints have zero velocity
            velocities[joint_idx, 0] = 0.0
            velocities[joint_idx, 1] = 0.0

        elif joint_type == JOINT_CRANK:
            p0_idx = parent_indices[joint_idx, 0]
            omega_idx = crank_omega_map[joint_idx]
            omega = omega_values[omega_idx] if omega_idx >= 0 else 0.0
            vx, vy = solve_crank_velocity(
                positions[joint_idx, 0],
                positions[joint_idx, 1],
                positions[p0_idx, 0],
                positions[p0_idx, 1],
                velocities[p0_idx, 0],
                velocities[p0_idx, 1],
                constraints[offset],  # radius
                omega,
            )
            velocities[joint_idx, 0] = vx
            velocities[joint_idx, 1] = vy

        elif joint_type == JOINT_REVOLUTE:
            p0_idx = parent_indices[joint_idx, 0]
            p1_idx = parent_indices[joint_idx, 1]
            vx, vy = solve_revolute_velocity(
                positions[joint_idx, 0],
                positions[joint_idx, 1],
                positions[p0_idx, 0],
                positions[p0_idx, 1],
                velocities[p0_idx, 0],
                velocities[p0_idx, 1],
                positions[p1_idx, 0],
                positions[p1_idx, 1],
                velocities[p1_idx, 0],
                velocities[p1_idx, 1],
            )
            velocities[joint_idx, 0] = vx
            velocities[joint_idx, 1] = vy

        elif joint_type == JOINT_FIXED:
            p0_idx = parent_indices[joint_idx, 0]
            p1_idx = parent_indices[joint_idx, 1]
            vx, vy = solve_fixed_velocity(
                positions[joint_idx, 0],
                positions[joint_idx, 1],
                positions[p0_idx, 0],
                positions[p0_idx, 1],
                velocities[p0_idx, 0],
                velocities[p0_idx, 1],
                positions[p1_idx, 0],
                positions[p1_idx, 1],
                velocities[p1_idx, 0],
                velocities[p1_idx, 1],
                constraints[offset],  # radius
                constraints[offset + 1],  # angle
            )
            velocities[joint_idx, 0] = vx
            velocities[joint_idx, 1] = vy

        elif joint_type == JOINT_LINEAR:
            p0_idx = parent_indices[joint_idx, 0]
            p1_idx = parent_indices[joint_idx, 1]
            p2_idx = parent_indices[joint_idx, 2]
            vx, vy = solve_prismatic_velocity(
                positions[joint_idx, 0],
                positions[joint_idx, 1],
                positions[p0_idx, 0],
                positions[p0_idx, 1],
                velocities[p0_idx, 0],
                velocities[p0_idx, 1],
                constraints[offset],  # radius
                positions[p1_idx, 0],
                positions[p1_idx, 1],
                velocities[p1_idx, 0],
                velocities[p1_idx, 1],
                positions[p2_idx, 0],
                positions[p2_idx, 1],
                velocities[p2_idx, 0],
                velocities[p2_idx, 1],
            )
            velocities[joint_idx, 0] = vx
            velocities[joint_idx, 1] = vy


@njit(cache=True)  # type: ignore[untyped-decorator]
def step_single_acceleration(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    constraints: np.ndarray,
    joint_types: np.ndarray,
    parent_indices: np.ndarray,
    constraint_offsets: np.ndarray,
    solve_order: np.ndarray,
    omega_values: np.ndarray,
    alpha_values: np.ndarray,
    crank_indices: np.ndarray,
) -> None:
    """Compute accelerations for all joints after velocities are solved.

    Must be called after step_single_velocity() has updated velocities.

    Args:
        positions: Joint positions, shape (n_joints, 2).
        velocities: Joint velocities, shape (n_joints, 2).
        accelerations: Output accelerations, shape (n_joints, 2). Modified in-place.
        constraints: Flat array of constraint values.
        joint_types: Joint type code for each joint, shape (n_joints,).
        parent_indices: Parent joint indices, shape (n_joints, max_parents).
        constraint_offsets: Start index in constraints for each joint.
        solve_order: Indices of joints to solve, in order.
        omega_values: Angular velocities for crank joints (rad/s).
        alpha_values: Angular accelerations for crank joints (rad/s²).
        crank_indices: Indices of crank joints in joints array.
    """
    # Build a mapping from joint index to crank index
    crank_map = np.full(len(joint_types), -1, dtype=np.int32)
    for i in range(len(crank_indices)):
        crank_map[crank_indices[i]] = i

    for i in range(len(solve_order)):
        joint_idx = solve_order[i]
        joint_type = joint_types[joint_idx]
        offset = constraint_offsets[joint_idx]

        if joint_type == JOINT_STATIC:
            # Static joints have zero acceleration
            accelerations[joint_idx, 0] = 0.0
            accelerations[joint_idx, 1] = 0.0

        elif joint_type == JOINT_CRANK:
            p0_idx = parent_indices[joint_idx, 0]
            crank_idx = crank_map[joint_idx]
            omega = omega_values[crank_idx] if crank_idx >= 0 else 0.0
            alpha = alpha_values[crank_idx] if crank_idx >= 0 else 0.0
            ax, ay = solve_crank_acceleration(
                positions[joint_idx, 0],
                positions[joint_idx, 1],
                velocities[joint_idx, 0],
                velocities[joint_idx, 1],
                positions[p0_idx, 0],
                positions[p0_idx, 1],
                velocities[p0_idx, 0],
                velocities[p0_idx, 1],
                accelerations[p0_idx, 0],
                accelerations[p0_idx, 1],
                constraints[offset],  # radius
                omega,
                alpha,
            )
            accelerations[joint_idx, 0] = ax
            accelerations[joint_idx, 1] = ay

        elif joint_type == JOINT_REVOLUTE:
            p0_idx = parent_indices[joint_idx, 0]
            p1_idx = parent_indices[joint_idx, 1]
            ax, ay = solve_revolute_acceleration(
                positions[joint_idx, 0],
                positions[joint_idx, 1],
                velocities[joint_idx, 0],
                velocities[joint_idx, 1],
                positions[p0_idx, 0],
                positions[p0_idx, 1],
                velocities[p0_idx, 0],
                velocities[p0_idx, 1],
                accelerations[p0_idx, 0],
                accelerations[p0_idx, 1],
                positions[p1_idx, 0],
                positions[p1_idx, 1],
                velocities[p1_idx, 0],
                velocities[p1_idx, 1],
                accelerations[p1_idx, 0],
                accelerations[p1_idx, 1],
            )
            accelerations[joint_idx, 0] = ax
            accelerations[joint_idx, 1] = ay

        elif joint_type == JOINT_FIXED:
            p0_idx = parent_indices[joint_idx, 0]
            p1_idx = parent_indices[joint_idx, 1]
            ax, ay = solve_fixed_acceleration(
                positions[joint_idx, 0],
                positions[joint_idx, 1],
                velocities[joint_idx, 0],
                velocities[joint_idx, 1],
                positions[p0_idx, 0],
                positions[p0_idx, 1],
                velocities[p0_idx, 0],
                velocities[p0_idx, 1],
                accelerations[p0_idx, 0],
                accelerations[p0_idx, 1],
                positions[p1_idx, 0],
                positions[p1_idx, 1],
                velocities[p1_idx, 0],
                velocities[p1_idx, 1],
                accelerations[p1_idx, 0],
                accelerations[p1_idx, 1],
                constraints[offset],  # radius
                constraints[offset + 1],  # angle
            )
            accelerations[joint_idx, 0] = ax
            accelerations[joint_idx, 1] = ay

        elif joint_type == JOINT_LINEAR:
            p0_idx = parent_indices[joint_idx, 0]
            p1_idx = parent_indices[joint_idx, 1]
            p2_idx = parent_indices[joint_idx, 2]
            ax, ay = solve_prismatic_acceleration(
                positions[joint_idx, 0],
                positions[joint_idx, 1],
                velocities[joint_idx, 0],
                velocities[joint_idx, 1],
                positions[p0_idx, 0],
                positions[p0_idx, 1],
                velocities[p0_idx, 0],
                velocities[p0_idx, 1],
                accelerations[p0_idx, 0],
                accelerations[p0_idx, 1],
                constraints[offset],  # radius
                positions[p1_idx, 0],
                positions[p1_idx, 1],
                velocities[p1_idx, 0],
                velocities[p1_idx, 1],
                accelerations[p1_idx, 0],
                accelerations[p1_idx, 1],
                positions[p2_idx, 0],
                positions[p2_idx, 1],
                velocities[p2_idx, 0],
                velocities[p2_idx, 1],
                accelerations[p2_idx, 0],
                accelerations[p2_idx, 1],
            )
            accelerations[joint_idx, 0] = ax
            accelerations[joint_idx, 1] = ay


@njit(cache=True)  # type: ignore[untyped-decorator]
def simulate_with_kinematics(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    constraints: np.ndarray,
    joint_types: np.ndarray,
    parent_indices: np.ndarray,
    constraint_offsets: np.ndarray,
    solve_order: np.ndarray,
    omega_values: np.ndarray,
    alpha_values: np.ndarray,
    crank_indices: np.ndarray,
    iterations: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run full simulation with velocity and acceleration computation.

    This extends the basic simulate() function to also compute velocities
    and accelerations at each step.

    Args:
        positions: Initial joint positions, shape (n_joints, 2).
            Will be modified to contain final positions.
        velocities: Initial velocities, shape (n_joints, 2).
            Will be modified to contain final velocities.
        accelerations: Initial accelerations, shape (n_joints, 2).
            Will be modified to contain final accelerations.
        constraints: Flat array of constraint values.
        joint_types: Joint type code for each joint, shape (n_joints,).
        parent_indices: Parent joint indices, shape (n_joints, max_parents).
        constraint_offsets: Start index in constraints for each joint.
        solve_order: Indices of joints to solve, in order.
        omega_values: Angular velocities for crank joints (rad/s).
        alpha_values: Angular accelerations for crank joints (rad/s²).
        crank_indices: Indices of crank joints in the joints array.
        iterations: Number of simulation steps to run.
        dt: Time step for crank rotation.

    Returns:
        Tuple of (positions_trajectory, velocities_trajectory,
        accelerations_trajectory), each with shape (iterations, n_joints, 2).
    """
    n_joints = positions.shape[0]
    pos_trajectory = np.empty((iterations, n_joints, 2), dtype=np.float64)
    vel_trajectory = np.empty((iterations, n_joints, 2), dtype=np.float64)
    acc_trajectory = np.empty((iterations, n_joints, 2), dtype=np.float64)

    for step in range(iterations):
        # First compute positions
        step_single(
            positions,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            dt,
        )

        # Then compute velocities based on new positions
        step_single_velocity(
            positions,
            velocities,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            crank_indices,
        )

        # Then compute accelerations based on new velocities
        step_single_acceleration(
            positions,
            velocities,
            accelerations,
            constraints,
            joint_types,
            parent_indices,
            constraint_offsets,
            solve_order,
            omega_values,
            alpha_values,
            crank_indices,
        )

        # Copy to trajectory arrays
        for j in range(n_joints):
            pos_trajectory[step, j, 0] = positions[j, 0]
            pos_trajectory[step, j, 1] = positions[j, 1]
            vel_trajectory[step, j, 0] = velocities[j, 0]
            vel_trajectory[step, j, 1] = velocities[j, 1]
            acc_trajectory[step, j, 0] = accelerations[j, 0]
            acc_trajectory[step, j, 1] = accelerations[j, 1]

    return pos_trajectory, vel_trajectory, acc_trajectory
