"""Symbolic solving and trajectory computation.

This module provides functions for solving linkage equations symbolically
and computing numeric trajectories from symbolic expressions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from ._types import SymCoord
from ._types import theta as default_theta

if TYPE_CHECKING:
    from .linkage import SymbolicLinkage


def solve_linkage_symbolically(
    linkage: SymbolicLinkage,
    output_joints: list[str] | None = None,
    simplify: bool = True,
) -> dict[str, SymCoord]:
    """
    Solve for joint positions as functions of theta.

    Returns (optionally simplified) closed-form expressions for
    trajectory curves of all or specified joints.

    Args:
        linkage: The symbolic linkage to solve.
        output_joints: List of joint names to include in output.
            If None, all joints are included.
        simplify: Whether to simplify the expressions. Default is True.

    Returns:
        Dictionary mapping joint name to (x(theta), y(theta)).
    """
    trajectories = linkage.get_trajectory_expressions()

    if output_joints is not None:
        trajectories = {k: v for k, v in trajectories.items() if k in output_joints}

    if simplify:
        simplified: dict[str, SymCoord] = {}
        for name, (x, y) in trajectories.items():
            simplified[name] = (sp.simplify(x), sp.simplify(y))
        return simplified

    return trajectories


def _rationalize(expr: sp.Expr) -> sp.Expr:
    """Replace every float by the rational it stands for, to 1e-10.

    The Groebner basis needs exact coefficients, and ``solve_linkage_symbolically``
    leaves rounding noise such as ``85.3312500000001`` on the ones it simplifies.
    """
    return sp.nsimplify(expr, rational=True, tolerance=1e-10)


def eliminate_theta(
    x_expr: sp.Expr,
    y_expr: sp.Expr,
    theta: sp.Symbol | None = None,
) -> sp.Expr | None:
    """
    Eliminate theta to get an implicit curve equation f(x, y) = 0.

    This converts parametric equations x(theta), y(theta) into an
    implicit algebraic equation in x and y only. This gives the
    algebraic equation of the coupler curve.

    Uses the substitution cos(theta) = c, sin(theta) = s with the
    constraint c^2 + s^2 = 1. Each square root ``sqrt(R)`` of the
    parametrization becomes a symbol ``r`` with the constraint
    ``r^2 = R``, so the system is polynomial; a Groebner basis then
    eliminates c, s and every r. The curve found is the full algebraic
    curve: both signs of every root, hence both assembly branches of the
    joint, lie on it.

    Args:
        x_expr: X coordinate as a function of theta.
        y_expr: Y coordinate as a function of theta.
        theta: The angle symbol to eliminate. Default uses global theta.

    Returns:
        Polynomial in x and y (and in the link lengths kept symbolic), or
        None if elimination fails. Floating-point coefficients are read as
        the rationals they stand for (to 1e-10), so the polynomial is exact.

    Note:
        The joint at the end of the first dyad has a single square root
        in its parametrization and its curve comes out in a fraction of a
        second. A joint built on that one has a root inside a root, and
        the Groebner basis computation may then run for minutes.

    Example:
        >>> theta = sp.Symbol('theta')
        >>> x = sp.cos(theta)
        >>> y = sp.sin(theta)
        >>> eliminate_theta(x, y, theta)
        x**2 + y**2 - 1
    """
    if theta is None:
        theta = default_theta

    x, y = sp.symbols("x y", real=True)
    c, s = sp.symbols("c s", real=True)

    # cos(theta) -> c, sin(theta) -> s; expand_trig first so that
    # cos(2*theta) and the like are polynomial in c and s too.
    trig = {sp.cos(theta): c, sp.sin(theta): s}
    x_sub = sp.expand_trig(_rationalize(x_expr)).subs(trig)
    y_sub = sp.expand_trig(_rationalize(y_expr)).subs(trig)
    if theta in x_sub.free_symbols or theta in y_sub.free_symbols:
        return None

    # sqrt(R) -> r with r**2 == R, innermost roots first
    radicals: dict[sp.Expr, sp.Symbol] = {}

    def is_root(expr: sp.Basic) -> bool:
        return bool(expr.is_Pow) and expr.exp.is_Rational and expr.exp.q == 2

    def as_symbol(expr: sp.Pow) -> sp.Expr:
        if expr.base not in radicals:
            radicals[expr.base] = sp.Symbol(f"_r{len(radicals)}", real=True)
        return radicals[expr.base] ** int(2 * expr.exp)

    x_sub = x_sub.replace(is_root, as_symbol)
    y_sub = y_sub.replace(is_root, as_symbol)

    equations = [
        sp.numer(sp.together(x - x_sub)),
        sp.numer(sp.together(y - y_sub)),
        c**2 + s**2 - 1,
        *(sp.numer(sp.together(r**2 - radicand)) for radicand, r in radicals.items()),
    ]
    roots = list(radicals.values())
    parameters = sorted(
        (x_sub.free_symbols | y_sub.free_symbols) - {c, s, x, y, *roots},
        key=str,
    )
    keep = {x, y, *parameters}

    try:
        # Lex order eliminates the leading generators: s and c first (the
        # parametrization is linear in s, so they go cheaply), then the
        # roots. x, y and the symbolic lengths are what remains.
        basis = sp.groebner(equations, s, c, *roots, x, y, *parameters, order="lex")
    except Exception:
        # Groebner basis computation can fail for complex expressions
        return None

    curves = [
        poly
        for poly in basis
        if poly.free_symbols <= keep and (x in poly.free_symbols or y in poly.free_symbols)
    ]
    if not curves:
        return None
    return min(curves, key=lambda poly: sp.Poly(poly, x, y).total_degree())


def compute_trajectory_numeric(
    linkage: SymbolicLinkage,
    param_values: dict[str, float],
    theta_values: Sequence[float] | NDArray[np.floating],
    output_joints: list[str] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """
    Compute numeric trajectory by evaluating symbolic expressions.

    Uses lambdify for efficient numeric evaluation of the symbolic
    trajectory expressions.

    Args:
        linkage: The symbolic linkage.
        param_values: Dictionary mapping parameter names to numeric values.
        theta_values: Sequence of theta values to evaluate at.
        output_joints: List of joint names to include. If None, all joints.

    Returns:
        Dictionary mapping joint name to Nx2 array of (x, y) positions.
    """
    theta_arr = np.asarray(theta_values, dtype=np.float64)

    # Get trajectories
    trajectories = linkage.get_trajectory_expressions()
    if output_joints is not None:
        trajectories = {k: v for k, v in trajectories.items() if k in output_joints}

    # Build parameter value list (sorted by name for consistency)
    param_symbols = sorted(linkage.parameters.values(), key=str)
    param_vals = [param_values[str(p)] for p in param_symbols]

    results: dict[str, NDArray[np.float64]] = {}

    for name, (x_expr, y_expr) in trajectories.items():
        # Create lambdified functions
        all_symbols = [linkage.theta] + param_symbols
        x_func = sp.lambdify(all_symbols, x_expr, modules=["numpy"])
        y_func = sp.lambdify(all_symbols, y_expr, modules=["numpy"])

        # Evaluate at all theta values
        # Suppress warnings for invalid sqrt (unbuildable configurations produce NaN)
        with np.errstate(invalid="ignore"):
            x_vals = x_func(theta_arr, *param_vals)
            y_vals = y_func(theta_arr, *param_vals)

        # Handle scalar results (for static joints)
        if np.isscalar(x_vals):
            x_vals = np.full_like(theta_arr, x_vals)
        if np.isscalar(y_vals):
            y_vals = np.full_like(theta_arr, y_vals)

        results[name] = np.column_stack([x_vals, y_vals])

    return results


def create_trajectory_functions(
    linkage: SymbolicLinkage,
    output_joints: list[str] | None = None,
) -> dict[str, tuple[object, object, list[sp.Symbol]]]:
    """
    Create lambdified functions for trajectory evaluation.

    This is useful when you need to evaluate trajectories many times
    with different parameter values, as it avoids repeated lambdify calls.

    Args:
        linkage: The symbolic linkage.
        output_joints: List of joint names to include. If None, all joints.

    Returns:
        Dictionary mapping joint name to (x_func, y_func, param_symbols).

    Example:
        >>> funcs = create_trajectory_functions(linkage)
        >>> x_func, y_func, params = funcs["C"]
        >>> x_vals = x_func(theta_array, *[param_values[str(p)] for p in params])
    """
    trajectories = linkage.get_trajectory_expressions()
    if output_joints is not None:
        trajectories = {k: v for k, v in trajectories.items() if k in output_joints}

    param_symbols = sorted(linkage.parameters.values(), key=str)
    all_symbols = [linkage.theta] + param_symbols

    results: dict[str, tuple[object, object, list[sp.Symbol]]] = {}

    for name, (x_expr, y_expr) in trajectories.items():
        x_func = sp.lambdify(all_symbols, x_expr, modules=["numpy"])
        y_func = sp.lambdify(all_symbols, y_expr, modules=["numpy"])
        results[name] = (x_func, y_func, param_symbols)

    return results


def check_buildability(
    linkage: SymbolicLinkage,
    param_values: dict[str, float],
    theta_value: float = 0.0,
) -> tuple[bool, str]:
    """
    Check if a linkage is buildable with given parameters.

    A linkage is unbuildable if the constraint equations cannot be
    satisfied (e.g., circles don't intersect). This function evaluates
    the positions at a specific theta and checks for complex (imaginary)
    results.

    Args:
        linkage: The symbolic linkage.
        param_values: Dictionary mapping parameter names to numeric values.
        theta_value: Theta value to check at. Default is 0.

    Returns:
        Tuple of (is_buildable, error_message).

    Example:
        >>> buildable, msg = check_buildability(linkage, {"r0": 1, "r1": 10})
        >>> if not buildable:
        ...     print(f"Linkage is unbuildable: {msg}")
    """
    # Substitute parameters
    param_subs = {
        linkage.parameters[k]: v for k, v in param_values.items() if k in linkage.parameters
    }
    theta_sub = {linkage.theta: theta_value}
    all_subs = {**param_subs, **theta_sub}

    for joint in linkage.joints:
        x_expr, y_expr = joint.position_expr()

        # Substitute and evaluate
        x_val = complex(x_expr.subs(all_subs).evalf())
        y_val = complex(y_expr.subs(all_subs).evalf())

        # Check for imaginary components
        if abs(x_val.imag) > 1e-10 or abs(y_val.imag) > 1e-10:
            return (
                False,
                f"Joint {joint.name!r} has complex position at theta={theta_value}: "
                f"({x_val}, {y_val}). Circles may not intersect.",
            )

    return (True, "")
