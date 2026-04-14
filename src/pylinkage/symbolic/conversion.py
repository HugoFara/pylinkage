"""Factory helpers and utilities for the symbolic linkage API.

Conversion helpers that bridged the legacy numeric ``Linkage`` (joint
API) and the symbolic linkage were removed alongside the
``pylinkage.joints`` module. Build symbolic linkages directly using
:class:`SymbolicLinkage` or :func:`fourbar_symbolic`.
"""

from __future__ import annotations

import sympy as sp

from .joints import SymCrank, SymRevolute, SymStatic
from .linkage import SymbolicLinkage


def get_numeric_parameters(sym_linkage: SymbolicLinkage) -> dict[str, float]:
    """Extract numeric parameter values stored in symbolic joints.

    Returns a ``{parameter_name: value}`` dict populated from
    ``_numeric_*`` attributes previously stashed on each joint.

    :param sym_linkage: The symbolic linkage.
    :returns: Dictionary mapping parameter names to numeric values.
    """
    params: dict[str, float] = {}

    for joint in sym_linkage.joints:
        if isinstance(joint, SymCrank):
            if hasattr(joint, "_numeric_r") and joint._numeric_r is not None:
                params[str(joint.r)] = joint._numeric_r

        elif isinstance(joint, SymRevolute):
            if hasattr(joint, "_numeric_r0") and joint._numeric_r0 is not None:
                params[str(joint.r0)] = joint._numeric_r0
            if hasattr(joint, "_numeric_r1") and joint._numeric_r1 is not None:
                params[str(joint.r1)] = joint._numeric_r1

    return params


def fourbar_symbolic(
    ground_length: float | sp.Expr | str = "L0",
    crank_length: float | sp.Expr | str = "L1",
    coupler_length: float | sp.Expr | str = "L2",
    rocker_length: float | sp.Expr | str = "L3",
    ground_x: float = 0.0,
    ground_y: float = 0.0,
) -> SymbolicLinkage:
    """Create a symbolic four-bar linkage.

    The four-bar consists of:

    - A (ground left): Static anchor at ``(ground_x, ground_y)``
    - D (ground right): Static anchor at ``(ground_x + ground_length, ground_y)``
    - B (crank end): Crank rotating around A with radius ``crank_length``
    - C (coupler/rocker connection): Revolute connecting B and D

    :param ground_length: Distance between ground anchors A and D.
    :param crank_length: Length of the input crank (A to B).
    :param coupler_length: Length of the coupler (B to C).
    :param rocker_length: Length of the output rocker (D to C).
    :param ground_x: X coordinate of ground anchor A. Default is 0.
    :param ground_y: Y coordinate of ground anchor A. Default is 0.
    :returns: A SymbolicLinkage representing the four-bar.
    """
    if isinstance(ground_length, str):
        L0 = sp.Symbol(ground_length, positive=True, real=True)
        d_x = ground_x + L0
    else:
        d_x = ground_x + sp.sympify(ground_length)

    A = SymStatic(x=ground_x, y=ground_y, name="A")
    D = SymStatic(x=d_x, y=ground_y, name="D")
    B = SymCrank(parent=A, radius=crank_length, name="B")
    C = SymRevolute(
        parent0=B,
        parent1=D,
        distance0=coupler_length,
        distance1=rocker_length,
        branch=1,
        name="C",
    )

    return SymbolicLinkage(joints=[A, D, B, C], name="fourbar")
