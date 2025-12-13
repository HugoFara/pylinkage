"""Conversion between Linkage objects and SolverData arrays.

This module re-exports conversion functions from the bridge module
for backwards compatibility. The actual implementation lives in
pylinkage.bridge.solver_conversion to keep the solver module pure
(only numpy/numba dependencies in the hot path).

For new code, prefer importing directly from pylinkage.bridge.
"""

# Re-export from bridge module for backwards compatibility
from ..bridge.solver_conversion import (
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
