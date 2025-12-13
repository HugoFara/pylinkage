"""Bridge module for converting between high-level and low-level representations.

This module provides conversion functions that bridge:
- Linkage (Python objects) ↔ SolverData (numpy arrays)

The bridge module sits at a higher abstraction level than both the solver
(which is pure numerics) and the linkage module (which is the user API).
This separation keeps the solver free of Python object dependencies for
maximum performance.

Typical usage:
    >>> from pylinkage import Linkage
    >>> from pylinkage.bridge import linkage_to_solver_data
    >>> from pylinkage.solver import simulate
    >>>
    >>> linkage = create_my_linkage()
    >>> data = linkage_to_solver_data(linkage)
    >>> trajectory = simulate(
    ...     data.positions, data.constraints, data.joint_types,
    ...     data.parent_indices, data.constraint_offsets,
    ...     data.solve_order, iterations=1000, dt=1.0
    ... )
"""

from .solver_conversion import (
    linkage_to_solver_data,
    solver_data_to_linkage,
    update_solver_constraints,
    update_solver_positions,
)

__all__ = [
    "linkage_to_solver_data",
    "solver_data_to_linkage",
    "update_solver_constraints",
    "update_solver_positions",
]
