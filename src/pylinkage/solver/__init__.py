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

# Pure numba components (no Python object dependencies). The kernels stay
# importable from here but are not public: see tests/test_public_api.py.
from .acceleration import (
    solve_crank_acceleration,  # noqa: F401
    solve_fixed_acceleration,  # noqa: F401
    solve_prismatic_acceleration,  # noqa: F401
    solve_revolute_acceleration,  # noqa: F401
    solve_rigid_body_acceleration,  # noqa: F401
)

# Group solvers (standalone functions for Assur groups)
from .groups import (
    solve_rrp_dyad,  # noqa: F401
    solve_rrr_dyad,  # noqa: F401
)
from .joints import (
    solve_crank,  # noqa: F401
    solve_fixed,  # noqa: F401
    solve_linear,  # noqa: F401
    solve_revolute,  # noqa: F401
)
from .simulation import (
    first_nan_step,  # noqa: F401
    has_nan_positions,  # noqa: F401
    simulate,
    simulate_with_kinematics,
    step_single,  # noqa: F401
    step_single_acceleration,  # noqa: F401
    step_single_velocity,  # noqa: F401
)

# High-level solving API
from .solve import (
    solve_decomposition,  # noqa: F401
    solve_group,  # noqa: F401
)
from .types import (
    JOINT_CRANK,
    JOINT_FIXED,
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    JOINT_STATIC,
    MAX_PARENTS,
    SolverData,
)
from .velocity import (
    solve_crank_velocity,  # noqa: F401
    solve_fixed_velocity,  # noqa: F401
    solve_prismatic_velocity,  # noqa: F401
    solve_revolute_velocity,  # noqa: F401
    solve_rigid_body_velocity,  # noqa: F401
)

# Conversion functions are loaded lazily to avoid circular imports. They are
# implemented in pylinkage.bridge (which keeps this package free of Python
# object dependencies) but this package is their public home.
_conversion_attrs = {
    "linkage_to_solver_data",
    "solver_data_to_linkage",
    "update_solver_constraints",
    "update_solver_positions",
}


def __getattr__(name: str) -> object:
    """Lazy import of conversion functions to avoid circular imports."""
    if name in _conversion_attrs:
        from ..bridge import solver_conversion

        return getattr(solver_conversion, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Types
    "SolverData",
    "JOINT_STATIC",
    "JOINT_CRANK",
    "JOINT_REVOLUTE",
    "JOINT_FIXED",
    "JOINT_PRISMATIC",
    "MAX_PARENTS",
    # Simulation
    "simulate",
    "simulate_with_kinematics",
    # Conversion (lazy-loaded to avoid a circular import through pylinkage.bridge)
    "linkage_to_solver_data",
    "solver_data_to_linkage",
    "update_solver_constraints",
    "update_solver_positions",
]
