"""Pure-numba simulation solver for linkage mechanisms.

This module provides a high-performance simulation backend that uses
numba JIT compilation to eliminate Python overhead in the hot loop.

The solver operates on numeric arrays rather than Python objects,
achieving significant speedups for repeated simulations (e.g., during
optimization).

Basic usage:
    >>> from pylinkage import Linkage
    >>> linkage = create_my_linkage()
    >>> trajectory = linkage.step_fast(iterations=1000)
    >>> # trajectory.shape == (1000, n_joints, 2)

For direct access to the solver:
    >>> from pylinkage.solver import linkage_to_solver_data, simulate
    >>> data = linkage_to_solver_data(linkage)
    >>> trajectory = simulate(
    ...     data.positions, data.constraints, data.joint_types,
    ...     data.parent_indices, data.constraint_offsets,
    ...     data.solve_order, iterations=1000, dt=1.0
    ... )
"""

# Pure numba components (no Python object dependencies)
from .joints import (
    solve_crank,
    solve_fixed,
    solve_linear,
    solve_revolute,
)
from .simulation import (
    first_nan_step,
    has_nan_positions,
    simulate,
    step_single,
)
from .types import (
    JOINT_CRANK,
    JOINT_FIXED,
    JOINT_LINEAR,
    JOINT_REVOLUTE,
    JOINT_STATIC,
    MAX_PARENTS,
    SolverData,
)

# Conversion functions are loaded lazily to avoid circular imports.
# They are implemented in pylinkage.bridge for architectural reasons
# (keeps solver pure, with no Python object dependencies).
# For new code, prefer importing from pylinkage.bridge directly.
_conversion_attrs = {
    "linkage_to_solver_data",
    "solver_data_to_linkage",
    "update_solver_constraints",
    "update_solver_positions",
}


def __getattr__(name: str) -> object:
    """Lazy import of conversion functions to avoid circular imports."""
    if name in _conversion_attrs:
        from . import conversion

        return getattr(conversion, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Types
    "SolverData",
    "JOINT_STATIC",
    "JOINT_CRANK",
    "JOINT_REVOLUTE",
    "JOINT_FIXED",
    "JOINT_LINEAR",
    "MAX_PARENTS",
    # Joint solvers
    "solve_crank",
    "solve_revolute",
    "solve_fixed",
    "solve_linear",
    # Simulation
    "step_single",
    "simulate",
    "has_nan_positions",
    "first_nan_step",
    # Conversion (lazy-loaded for backwards compatibility)
    "linkage_to_solver_data",
    "solver_data_to_linkage",
    "update_solver_constraints",
    "update_solver_positions",
]
