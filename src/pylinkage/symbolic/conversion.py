"""Conversion between symbolic and numeric linkage representations.

This module provides functions to convert between the symbolic linkage
representation (using SymPy) and the numeric linkage representation
(using the standard pylinkage classes).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp

from ._types import theta as default_theta
from .joints import SymCrank, SymJoint, SymRevolute, SymStatic
from .linkage import SymbolicLinkage

if TYPE_CHECKING:
    from ..joints.joint import Joint
    from ..linkage.linkage import Linkage


def linkage_to_symbolic(
    linkage: Linkage,
    param_prefix: str = "",
) -> SymbolicLinkage:
    """
    Convert a numeric Linkage to a SymbolicLinkage.

    Creates symbolic joints with parameter symbols for constraints.
    The original numeric values are stored as `_numeric_*` attributes
    for later retrieval.

    :param linkage: The numeric Linkage to convert.
    :param param_prefix: Prefix for parameter names. Default is empty.
    :returns: A SymbolicLinkage with symbolic parameters.

    Example:
        >>> from pylinkage import Linkage, Static, Crank, Revolute
        >>> # Create numeric linkage
        >>> A = Static(0, 0, name="A")
        >>> B = Crank(1, 0, joint0=A, distance=1, angle=0.1, name="B")
        >>> linkage = Linkage(joints=[A, B], order=[B])
        >>> # Convert to symbolic
        >>> sym_linkage = linkage_to_symbolic(linkage)
    """
    from ..joints.crank import Crank
    from ..joints.joint import Static
    from ..joints.revolute import Revolute

    sym_joints: list[SymJoint] = []
    joint_map: dict[int, SymJoint] = {}  # Maps numeric joint id to symbolic joint

    for joint in linkage.joints:
        if isinstance(joint, Crank):
            # Find parent in symbolic joints
            parent: SymJoint | tuple[float, float]
            if joint.joint0 is not None and id(joint.joint0) in joint_map:
                parent = joint_map[id(joint.joint0)]
            elif joint.joint0 is not None:
                parent = (
                    float(joint.joint0.x) if joint.joint0.x is not None else 0.0,
                    float(joint.joint0.y) if joint.joint0.y is not None else 0.0,
                )
            else:
                parent = (0.0, 0.0)

            # Create parameter name
            param_name = f"{param_prefix}r_{joint.name}" if param_prefix else f"r_{joint.name}"

            sym_crank = SymCrank(
                parent=parent,
                radius=param_name,
                name=joint.name,
            )
            # Store numeric value
            sym_crank._numeric_r = joint.r
            sym_joint: SymJoint = sym_crank

        elif isinstance(joint, Revolute):
            # Find parents in symbolic joints
            parent0 = joint_map.get(id(joint.joint0)) if joint.joint0 else None
            parent1 = joint_map.get(id(joint.joint1)) if joint.joint1 else None

            if parent0 is None or parent1 is None:
                raise ValueError(
                    f"Revolute joint {joint.name!r} has parents that were not "
                    "converted yet. Ensure joints are in topological order."
                )

            # Create parameter names
            param_name0 = f"{param_prefix}r0_{joint.name}" if param_prefix else f"r0_{joint.name}"
            param_name1 = f"{param_prefix}r1_{joint.name}" if param_prefix else f"r1_{joint.name}"

            # Determine branch from initial position
            branch = _determine_branch(joint, parent0, parent1)

            sym_rev = SymRevolute(
                parent0=parent0,
                parent1=parent1,
                distance0=param_name0,
                distance1=param_name1,
                branch=branch,
                name=joint.name,
            )
            # Store numeric values
            sym_rev._numeric_r0 = joint.r0
            sym_rev._numeric_r1 = joint.r1
            sym_joint = sym_rev

        elif isinstance(joint, Static):
            sym_joint = SymStatic(
                x=float(joint.x) if joint.x is not None else 0.0,
                y=float(joint.y) if joint.y is not None else 0.0,
                name=joint.name,
            )

        else:
            raise TypeError(
                f"Unsupported joint type {type(joint).__name__}. "
                "Only Static, Crank, and Revolute are supported."
            )

        sym_joints.append(sym_joint)
        joint_map[id(joint)] = sym_joint

    return SymbolicLinkage(
        joints=sym_joints,
        name=linkage.name,
    )


def _determine_branch(
    joint: Joint,
    sym_parent0: SymJoint,
    sym_parent1: SymJoint,
) -> int:
    """
    Determine which branch (+1 or -1) to use based on initial position.

    Evaluates both branches and picks the one closest to the joint's
    current position.
    """
    from .geometry import symbolic_circle_intersect

    # Get current position
    if joint.x is None or joint.y is None:
        return 1  # Default to +1 if no initial position

    # Get parent positions (assume they're static or already evaluated)
    p0_x, p0_y = sym_parent0.position_expr()
    p1_x, p1_y = sym_parent1.position_expr()

    # Get distances from original joint
    from ..joints.revolute import Revolute
    if isinstance(joint, Revolute):
        r0 = joint.r0 if joint.r0 is not None else 1.0
        r1 = joint.r1 if joint.r1 is not None else 1.0
    else:
        r0 = r1 = 1.0

    # Try both branches and see which is closer
    try:
        # Substitute with theta=0 for static evaluation
        subs = {default_theta: 0}

        x1, y1 = symbolic_circle_intersect(p0_x, p0_y, r0, p1_x, p1_y, r1, branch=1)
        x2, y2 = symbolic_circle_intersect(p0_x, p0_y, r0, p1_x, p1_y, r1, branch=-1)

        x1_val = float(x1.subs(subs).evalf())
        y1_val = float(y1.subs(subs).evalf())
        x2_val = float(x2.subs(subs).evalf())
        y2_val = float(y2.subs(subs).evalf())

        dist1 = (joint.x - x1_val) ** 2 + (joint.y - y1_val) ** 2
        dist2 = (joint.x - x2_val) ** 2 + (joint.y - y2_val) ** 2

        return 1 if dist1 <= dist2 else -1

    except Exception:
        return 1  # Default to +1 on error


def symbolic_to_linkage(
    sym_linkage: SymbolicLinkage,
    param_values: dict[str, float],
    initial_theta: float = 0.0,
) -> Linkage:
    """
    Convert a SymbolicLinkage to a numeric Linkage.

    Substitutes parameter values and evaluates at initial_theta to get
    initial positions, then creates the corresponding numeric joints.

    :param sym_linkage: The symbolic linkage to convert.
    :param param_values: Dictionary mapping parameter names to numeric values.
    :param initial_theta: Theta value to evaluate initial positions at.
        Default is 0.
    :returns: A numeric Linkage.

    Example:
        >>> params = {"r_B": 1.0, "r0_C": 3.0, "r1_C": 3.0}
        >>> linkage = symbolic_to_linkage(sym_linkage, params)
    """
    from ..joints.crank import Crank
    from ..joints.joint import Static
    from ..joints.revolute import Revolute
    from ..linkage.linkage import Linkage

    joints: list[Joint] = []
    joint_map: dict[str, Joint] = {}  # Maps joint name to numeric joint
    order: list[Joint] = []  # Joints that need to be simulated

    # Build substitution dictionary
    param_subs = {
        sym_linkage.parameters[k]: v
        for k, v in param_values.items()
        if k in sym_linkage.parameters
    }
    theta_sub = {sym_linkage.theta: initial_theta}
    all_subs = {**param_subs, **theta_sub}

    for sym_joint in sym_linkage.joints:
        # Evaluate position at initial_theta
        x_expr, y_expr = sym_joint.position_expr()
        x_val = float(x_expr.subs(all_subs).evalf())
        y_val = float(y_expr.subs(all_subs).evalf())

        # Handle complex results (unbuildable)
        if math.isnan(x_val) or math.isnan(y_val):
            raise ValueError(
                f"Joint {sym_joint.name!r} has invalid position. "
                "The linkage may be unbuildable with these parameters."
            )

        if isinstance(sym_joint, SymStatic):
            joint: Joint = Static(x=x_val, y=y_val, name=sym_joint.name)

        elif isinstance(sym_joint, SymCrank):
            # Get parent
            parent: Joint | None = None
            if sym_joint.parent0 is not None:
                parent = joint_map.get(sym_joint.parent0.name)

            # Get radius value
            r_val: float
            if isinstance(sym_joint.r, sp.Symbol):
                r_val = param_values.get(str(sym_joint.r), 1.0)
            else:
                r_val = float(sym_joint.r.subs(all_subs).evalf())

            # Compute angle step (default to 2*pi/100 for 100 steps per rotation)
            angle_step = 2 * np.pi / 100

            joint = Crank(
                x=x_val,
                y=y_val,
                joint0=parent,
                distance=r_val,
                angle=angle_step,
                name=sym_joint.name,
            )
            order.append(joint)

        elif isinstance(sym_joint, SymRevolute):
            # Get parents
            parent0 = joint_map.get(sym_joint.parent0.name) if sym_joint.parent0 else None
            parent1 = joint_map.get(sym_joint.parent1.name) if sym_joint.parent1 else None

            # Get distance values
            r0_val: float
            r1_val: float
            if isinstance(sym_joint.r0, sp.Symbol):
                r0_val = param_values.get(str(sym_joint.r0), 1.0)
            else:
                r0_val = float(sym_joint.r0.subs(all_subs).evalf())
            if isinstance(sym_joint.r1, sp.Symbol):
                r1_val = param_values.get(str(sym_joint.r1), 1.0)
            else:
                r1_val = float(sym_joint.r1.subs(all_subs).evalf())

            joint = Revolute(
                x=x_val,
                y=y_val,
                joint0=parent0,
                joint1=parent1,
                distance0=r0_val,
                distance1=r1_val,
                name=sym_joint.name,
            )
            order.append(joint)

        else:
            raise TypeError(f"Unsupported symbolic joint type: {type(sym_joint)}")

        joints.append(joint)
        joint_map[sym_joint.name] = joint

    return Linkage(
        joints=joints,
        order=order if order else None,
        name=sym_linkage.name,
    )


def get_numeric_parameters(sym_linkage: SymbolicLinkage) -> dict[str, float]:
    """
    Extract numeric parameter values stored in symbolic joints.

    When a SymbolicLinkage is created from a numeric Linkage using
    linkage_to_symbolic(), the original numeric values are stored
    as `_numeric_*` attributes. This function retrieves them.

    :param sym_linkage: The symbolic linkage.
    :returns: Dictionary mapping parameter names to numeric values.

    Example:
        >>> sym_linkage = linkage_to_symbolic(linkage)
        >>> params = get_numeric_parameters(sym_linkage)
        >>> # params = {"r_B": 1.0, "r0_C": 3.0, "r1_C": 3.0}
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
    """
    Create a symbolic four-bar linkage.

    This is a convenience function for creating the most common linkage type.

    The four-bar consists of:
    - A (ground left): Static anchor at (ground_x, ground_y)
    - D (ground right): Static anchor at (ground_x + ground_length, ground_y)
    - B (crank end): Crank rotating around A with radius crank_length
    - C (coupler/rocker connection): Revolute connecting B and D

    :param ground_length: Distance between ground anchors A and D.
    :param crank_length: Length of the input crank (A to B).
    :param coupler_length: Length of the coupler (B to C).
    :param rocker_length: Length of the output rocker (D to C).
    :param ground_x: X coordinate of ground anchor A. Default is 0.
    :param ground_y: Y coordinate of ground anchor A. Default is 0.
    :returns: A SymbolicLinkage representing the four-bar.

    Example:
        >>> linkage = fourbar_symbolic(
        ...     ground_length=4,
        ...     crank_length=1,
        ...     coupler_length=3,
        ...     rocker_length=3,
        ... )
        >>> x_expr, y_expr = linkage.coupler_curve("C")
    """
    # Handle ground length for D position
    if isinstance(ground_length, str):
        L0 = sp.Symbol(ground_length, positive=True, real=True)
        d_x = ground_x + L0
    else:
        d_x = ground_x + sp.sympify(ground_length)

    # Create joints
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
