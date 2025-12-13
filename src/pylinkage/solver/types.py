"""Numba-compatible data structures for the solver.

This module defines the numeric representation of linkages for use
in the pure-numba simulation loop.
"""


from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Joint type constants
JOINT_STATIC = 0
JOINT_CRANK = 1
JOINT_REVOLUTE = 2
JOINT_FIXED = 3
JOINT_LINEAR = 4  # Deprecated: use JOINT_PRISMATIC
JOINT_PRISMATIC = 4  # Alias for JOINT_LINEAR

# Maximum number of parent joints (Prismatic has 3: circle center + 2 line points)
MAX_PARENTS = 3


@dataclass
class SolverData:
    """Numba-compatible representation of a linkage.

    All data is stored in contiguous numpy arrays for efficient
    access from numba-compiled code.

    Attributes:
        positions: Joint positions, shape (n_joints, 2).
        constraints: Flat array of all constraint values.
        joint_types: Joint type code for each joint, shape (n_joints,).
        parent_indices: Parent joint indices, shape (n_joints, MAX_PARENTS).
            Unused slots contain -1.
        constraint_offsets: Start index in constraints array for each joint.
        constraint_counts: Number of constraints for each joint.
        solve_order: Indices of joints in solving order.
    """

    positions: NDArray[np.float64]
    constraints: NDArray[np.float64]
    joint_types: NDArray[np.int32]
    parent_indices: NDArray[np.int32]
    constraint_offsets: NDArray[np.int32]
    constraint_counts: NDArray[np.int32]
    solve_order: NDArray[np.int32]

    def copy(self) -> "SolverData":
        """Create a deep copy of the solver data."""
        return SolverData(
            positions=self.positions.copy(),
            constraints=self.constraints.copy(),
            joint_types=self.joint_types.copy(),
            parent_indices=self.parent_indices.copy(),
            constraint_offsets=self.constraint_offsets.copy(),
            constraint_counts=self.constraint_counts.copy(),
            solve_order=self.solve_order.copy(),
        )

    @property
    def n_joints(self) -> int:
        """Number of joints in the linkage."""
        return len(self.joint_types)

    @property
    def n_constraints(self) -> int:
        """Total number of constraints."""
        return len(self.constraints)
