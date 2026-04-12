"""Symbolic computation module for pylinkage.

This module provides symbolic computation capabilities using SymPy,
enabling:

- **Closed-form trajectory expressions**: Get algebraic expressions for
  joint positions as functions of the input angle and link lengths.

- **Symbolic constraint equations**: Express linkage constraints as
  symbolic equations for analysis and manipulation.

- **Gradient-based optimization**: Compute analytical gradients for
  efficient gradient-based optimization of linkage parameters.

Example usage::

    from pylinkage.symbolic import (
        SymStatic, SymCrank, SymRevolute, SymbolicLinkage,
        compute_trajectory_numeric, SymbolicOptimizer,
        fourbar_symbolic,
    )
    import numpy as np

    # Create a four-bar linkage symbolically
    linkage = fourbar_symbolic(
        ground_length=4,
        crank_length="L1",  # Symbolic parameter
        coupler_length="L2",
        rocker_length="L3",
    )

    # Get closed-form trajectory expression for coupler point C
    x_expr, y_expr = linkage.coupler_curve("C")
    print(f"C_x(theta) = {x_expr}")
    print(f"C_y(theta) = {y_expr}")

    # Compute numeric trajectory with specific parameter values
    params = {"L1": 1.0, "L2": 3.0, "L3": 3.0}
    theta_vals = np.linspace(0, 2 * np.pi, 360)
    trajectories = compute_trajectory_numeric(linkage, params, theta_vals)

    # Access (x, y) positions for joint C
    C_positions = trajectories["C"]  # Shape: (360, 2)

For optimization::

    import sympy as sp

    # Define objective function symbolically
    def objective(trajectories):
        x, y = trajectories["C"]
        # Minimize deviation from target y=2
        return (y - 2)**2

    optimizer = SymbolicOptimizer(linkage, objective)
    result = optimizer.optimize(
        initial_params={"L1": 1.0, "L2": 3.0, "L3": 3.0},
        bounds={"L1": (0.5, 2), "L2": (1, 5), "L3": (1, 5)},
    )

    if result.success:
        print(f"Optimal parameters: {result.params}")
        print(f"Objective value: {result.objective_value}")
"""

__all__ = [
    # Type definitions
    "SymCoord",
    "theta",
    # Joint classes
    "SymJoint",
    "SymStatic",
    "SymCrank",
    "SymRevolute",
    # Linkage
    "SymbolicLinkage",
    # Geometry functions
    "symbolic_circle_intersect",
    "symbolic_circle_line_intersect",
    "symbolic_cyl_to_cart",
    "symbolic_dist",
    "symbolic_sqr_dist",
    # Solver functions
    "solve_linkage_symbolically",
    "eliminate_theta",
    "compute_trajectory_numeric",
    "create_trajectory_functions",
    "check_buildability",
    # Optimization
    "SymbolicOptimizer",
    "OptimizationResult",
    "symbolic_gradient",
    "symbolic_hessian",
    "generate_symbolic_bounds",
    # Conversion functions
    "linkage_to_symbolic",
    "symbolic_to_linkage",
    "get_numeric_parameters",
    "fourbar_symbolic",
]

from ._types import SymCoord, theta
from .conversion import (
    fourbar_symbolic,
    get_numeric_parameters,
    linkage_to_symbolic,
    symbolic_to_linkage,
)
from .geometry import (
    symbolic_circle_intersect,
    symbolic_circle_line_intersect,
    symbolic_cyl_to_cart,
    symbolic_dist,
    symbolic_sqr_dist,
)
from .joints import SymCrank, SymJoint, SymRevolute, SymStatic
from .linkage import SymbolicLinkage
from .optimization import (
    OptimizationResult,
    SymbolicOptimizer,
    generate_symbolic_bounds,
    symbolic_gradient,
    symbolic_hessian,
)
from .solver import (
    check_buildability,
    compute_trajectory_numeric,
    create_trajectory_functions,
    eliminate_theta,
    solve_linkage_symbolically,
)
