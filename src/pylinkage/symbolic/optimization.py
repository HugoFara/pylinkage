"""Symbolic optimization with gradients and Jacobians.

This module provides gradient-based optimization for linkage parameters
using symbolic derivatives computed by SymPy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from ._types import SymCoord

if TYPE_CHECKING:
    from .linkage import SymbolicLinkage


def symbolic_gradient(
    objective: sp.Expr,
    parameters: list[sp.Symbol],
) -> list[sp.Expr]:
    """
    Compute the symbolic gradient of an objective function.

    :param objective: Symbolic objective expression.
    :param parameters: List of parameter symbols to differentiate with respect to.
    :returns: List of partial derivative expressions.
    """
    return [sp.diff(objective, p) for p in parameters]


def symbolic_hessian(
    objective: sp.Expr,
    parameters: list[sp.Symbol],
) -> sp.Matrix:
    """
    Compute the symbolic Hessian matrix of an objective function.

    :param objective: Symbolic objective expression.
    :param parameters: List of parameter symbols.
    :returns: SymPy Matrix of second partial derivatives.
    """
    return sp.hessian(objective, parameters)


@dataclass
class OptimizationResult:
    """Result of a symbolic optimization."""

    success: bool
    """Whether the optimization converged successfully."""

    params: dict[str, float]
    """Optimal parameter values."""

    objective_value: float
    """Final objective function value."""

    iterations: int
    """Number of iterations performed."""

    message: str = ""
    """Status message from the optimizer."""


class SymbolicOptimizer:
    """
    Gradient-based optimization using symbolic derivatives.

    This class pre-computes symbolic gradients and uses lambdify for
    fast numeric evaluation during optimization. The optimizer uses
    scipy.optimize.minimize with analytical gradients.

    Example:
        >>> def objective(trajectories):
        ...     x, y = trajectories["C"]
        ...     return (y - 2)**2  # Minimize deviation from y=2
        ...
        >>> optimizer = SymbolicOptimizer(linkage, objective)
        >>> result = optimizer.optimize(
        ...     initial_params={"r_AB": 1.0, "r_BC": 3.0, "r_CD": 3.0},
        ...     bounds={"r_AB": (0.5, 2), "r_BC": (1, 5), "r_CD": (1, 5)},
        ... )
    """

    def __init__(
        self,
        linkage: SymbolicLinkage,
        objective_func: Callable[[dict[str, SymCoord]], sp.Expr],
        theta_samples: int = 100,
    ) -> None:
        """
        Create a symbolic optimizer.

        :param linkage: The symbolic linkage to optimize.
        :param objective_func: Function that takes trajectory dict and returns
            a symbolic objective expression. The trajectories are
            {joint_name: (x_expr, y_expr)} where x_expr and y_expr are
            symbolic expressions in theta and parameters.
        :param theta_samples: Number of theta samples for averaging objectives
            that depend on the full trajectory. Default is 100.
        """
        self.linkage = linkage
        self.objective_func = objective_func
        self.theta_samples = theta_samples

        # Build symbolic objective and gradient
        self._setup_optimization()

    def _setup_optimization(self) -> None:
        """Pre-compute symbolic objective, gradient, and lambdified functions."""
        # Get trajectory expressions
        trajectories = self.linkage.get_trajectory_expressions()

        # Compute symbolic objective
        self.objective_expr = self.objective_func(trajectories)

        # Get parameter list (sorted for consistency)
        self.param_list = sorted(self.linkage.parameters.values(), key=str)
        self.param_names = [str(p) for p in self.param_list]

        # Compute symbolic gradient
        self.gradient_exprs = symbolic_gradient(self.objective_expr, self.param_list)

        # All symbols for lambdify: theta + params
        all_symbols = [self.linkage.theta] + self.param_list

        # Lambdify objective and gradient
        self._objective_func = sp.lambdify(
            all_symbols, self.objective_expr, modules=["numpy"]
        )
        self._gradient_funcs = [
            sp.lambdify(all_symbols, g, modules=["numpy"])
            for g in self.gradient_exprs
        ]

    def evaluate(
        self,
        param_values: dict[str, float],
        theta_samples: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> float:
        """
        Evaluate the objective function numerically.

        The objective is averaged over all theta samples.

        :param param_values: Dictionary mapping parameter names to values.
        :param theta_samples: Theta values to evaluate at. If None, uses
            equally spaced samples over [0, 2*pi].
        :returns: Mean objective value.
        """
        if theta_samples is None:
            theta_samples = np.linspace(0, 2 * np.pi, self.theta_samples)
        else:
            theta_samples = np.asarray(theta_samples)

        params = [param_values[n] for n in self.param_names]

        # Evaluate at all theta values and average
        # Suppress warnings for invalid sqrt (unbuildable configurations produce NaN)
        values = []
        with np.errstate(invalid="ignore"):
            for t in theta_samples:
                try:
                    val = float(self._objective_func(t, *params))
                    if np.isfinite(val):
                        values.append(val)
                except (ValueError, ZeroDivisionError, TypeError):
                    # Skip invalid points (e.g., unbuildable configurations)
                    pass

        if not values:
            return float("inf")

        return float(np.mean(values))

    def gradient(
        self,
        param_values: dict[str, float],
        theta_samples: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> NDArray[np.float64]:
        """
        Evaluate the gradient numerically.

        The gradient is averaged over all theta samples.

        :param param_values: Dictionary mapping parameter names to values.
        :param theta_samples: Theta values to evaluate at. If None, uses
            equally spaced samples over [0, 2*pi].
        :returns: Array of gradient values (one per parameter).
        """
        if theta_samples is None:
            theta_samples = np.linspace(0, 2 * np.pi, self.theta_samples)
        else:
            theta_samples = np.asarray(theta_samples)

        params = [param_values[n] for n in self.param_names]

        # Evaluate gradient at all theta values and average
        # Suppress warnings for invalid sqrt (unbuildable configurations produce NaN)
        grads: list[list[float]] = [[] for _ in self._gradient_funcs]

        with np.errstate(invalid="ignore"):
            for t in theta_samples:
                try:
                    for i, g_func in enumerate(self._gradient_funcs):
                        val = float(g_func(t, *params))
                        if np.isfinite(val):
                            grads[i].append(val)
                except (ValueError, ZeroDivisionError, TypeError):
                    pass

        result = []
        for g_vals in grads:
            if g_vals:
                result.append(float(np.mean(g_vals)))
            else:
                result.append(0.0)

        return np.array(result)

    def optimize(
        self,
        initial_params: dict[str, float],
        bounds: dict[str, tuple[float, float]] | None = None,
        theta_samples: Sequence[float] | NDArray[np.floating] | None = None,
        method: str = "L-BFGS-B",
        maxiter: int = 1000,
        tol: float = 1e-6,
    ) -> OptimizationResult:
        """
        Run gradient-based optimization.

        Uses scipy.optimize.minimize with analytical gradients.

        :param initial_params: Dictionary of initial parameter values.
        :param bounds: Dictionary mapping parameter names to (min, max) bounds.
            If None, parameters are unbounded.
        :param theta_samples: Theta values for objective evaluation.
        :param method: Scipy optimization method. Default is "L-BFGS-B".
        :param maxiter: Maximum iterations. Default is 1000.
        :param tol: Convergence tolerance. Default is 1e-6.
        :returns: OptimizationResult with optimal parameters.
        """
        from scipy.optimize import minimize

        if theta_samples is None:
            theta_samples = np.linspace(0, 2 * np.pi, self.theta_samples)
        else:
            theta_samples = np.asarray(theta_samples)

        # Initial point as array
        x0 = np.array([initial_params[n] for n in self.param_names])

        def objective(x: NDArray[np.float64]) -> float:
            pv = dict(zip(self.param_names, x, strict=True))
            return self.evaluate(pv, theta_samples)

        def gradient(x: NDArray[np.float64]) -> NDArray[np.float64]:
            pv = dict(zip(self.param_names, x, strict=True))
            return self.gradient(pv, theta_samples)

        # Format bounds for scipy
        scipy_bounds = None
        if bounds:
            scipy_bounds = [bounds.get(n, (None, None)) for n in self.param_names]

        # Run optimization
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method=method,
            bounds=scipy_bounds,
            options={"maxiter": maxiter},
            tol=tol,
        )

        return OptimizationResult(
            success=result.success,
            params=dict(zip(self.param_names, result.x, strict=True)),
            objective_value=float(result.fun),
            iterations=result.nit,
            message=result.message if hasattr(result, "message") else "",
        )


def generate_symbolic_bounds(
    linkage: SymbolicLinkage,
    center_values: dict[str, float],
    min_ratio: float = 5.0,
    max_factor: float = 5.0,
) -> dict[str, tuple[float, float]]:
    """
    Generate optimization bounds based on center values.

    Creates bounds as (center/min_ratio, center*max_factor) for each
    parameter. This is similar to the numeric generate_bounds function.

    :param linkage: The symbolic linkage.
    :param center_values: Dictionary of center values for each parameter.
    :param min_ratio: Divisor for lower bound. Default is 5.
    :param max_factor: Multiplier for upper bound. Default is 5.
    :returns: Dictionary mapping parameter names to (min, max) bounds.
    """
    bounds = {}
    for param_name in linkage.parameters:
        if param_name in center_values:
            center = center_values[param_name]
            bounds[param_name] = (center / min_ratio, center * max_factor)
    return bounds
