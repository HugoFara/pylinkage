"""Symbolic linkage representation.

This module provides the SymbolicLinkage class for building and analyzing
linkages with symbolic (SymPy) expressions for joint positions.
"""

from __future__ import annotations

from collections.abc import Iterable

import sympy as sp

from ._types import SymCoord
from ._types import theta as default_theta
from .joints import SymJoint


class SymbolicLinkage:
    """
    A linkage with symbolic position expressions.

    The SymbolicLinkage class holds a collection of symbolic joints and
    provides methods for:
    - Computing closed-form trajectory expressions
    - Extracting constraint equations
    - Computing Jacobians for optimization
    - Simplifying expressions
    """

    __slots__ = ("joints", "name", "theta", "_parameters")

    joints: tuple[SymJoint, ...]
    name: str
    theta: sp.Symbol
    _parameters: dict[str, sp.Symbol] | None

    def __init__(
        self,
        joints: Iterable[SymJoint],
        theta: sp.Symbol | None = None,
        name: str | None = None,
    ) -> None:
        """
        Create a symbolic linkage from a collection of joints.

        :param joints: Iterable of SymJoint objects.
        :param theta: Input angle symbol. Default uses the global theta.
        :param name: Human-readable name for the linkage.
        """
        self.joints = tuple(joints)
        self.theta = theta if theta is not None else default_theta
        self.name = name if name is not None else str(id(self))
        self._parameters = None  # Lazy initialization

    def __repr__(self) -> str:
        """Return a string representation of the linkage."""
        joint_names = [j.name for j in self.joints]
        return f"SymbolicLinkage(name={self.name!r}, joints={joint_names})"

    @property
    def parameters(self) -> dict[str, sp.Symbol]:
        """
        Return all symbolic parameters in the linkage.

        Collects parameters from all joints and returns them as a
        dictionary mapping parameter names to symbols.

        :returns: Dictionary of parameter name to symbol.
        """
        if self._parameters is None:
            self._parameters = {}
            for joint in self.joints:
                for sym in joint.parameters:
                    self._parameters[str(sym)] = sym
        return self._parameters

    def get_joint(self, name: str) -> SymJoint:
        """
        Get a joint by name.

        :param name: Name of the joint.
        :returns: The joint with the given name.
        :raises ValueError: If no joint with the given name exists.
        """
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise ValueError(f"Joint {name!r} not found in linkage")

    def get_trajectory_expressions(self) -> dict[str, SymCoord]:
        """
        Return symbolic trajectory expressions for all joints.

        :returns: Dictionary mapping joint name to (x(theta), y(theta)).
        """
        return {joint.name: joint.position_expr() for joint in self.joints}

    def get_constraint_equations(self) -> list[sp.Eq]:
        """
        Return all constraint equations for the linkage.

        :returns: List of SymPy Eq objects representing all constraints.
        """
        equations: list[sp.Eq] = []
        for joint in self.joints:
            equations.extend(joint.constraint_equations())
        return equations

    def coupler_curve(self, joint_name: str) -> SymCoord:
        """
        Get the trajectory expression for a specific joint.

        :param joint_name: Name of the joint.
        :returns: Tuple of (x(theta), y(theta)) expressions.
        :raises ValueError: If no joint with the given name exists.
        """
        joint = self.get_joint(joint_name)
        return joint.position_expr()

    def jacobian(
        self,
        joint_names: list[str] | None = None,
    ) -> sp.Matrix:
        """
        Compute the Jacobian matrix of positions with respect to parameters.

        The Jacobian has rows for (x1, y1, x2, y2, ...) and columns for
        each parameter. This is useful for sensitivity analysis and
        gradient-based optimization.

        :param joint_names: List of joint names to include. If None, all
            joints are included.
        :returns: SymPy Matrix of shape (2*n_joints, n_params).
        """
        if joint_names is None:
            joints_to_include = self.joints
        else:
            joints_to_include = tuple(self.get_joint(n) for n in joint_names)

        # Build position vector
        positions: list[sp.Expr] = []
        for joint in joints_to_include:
            x, y = joint.position_expr()
            positions.extend([x, y])

        # Get parameter list (sorted for consistency)
        param_list = sorted(self.parameters.values(), key=str)

        # Compute Jacobian
        pos_matrix = sp.Matrix(positions)
        return pos_matrix.jacobian(param_list)

    def jacobian_theta(
        self,
        joint_names: list[str] | None = None,
    ) -> sp.Matrix:
        """
        Compute the Jacobian of positions with respect to theta.

        This gives the velocity direction for each joint.

        :param joint_names: List of joint names to include. If None, all
            joints are included.
        :returns: SymPy Matrix of shape (2*n_joints, 1).
        """
        if joint_names is None:
            joints_to_include = self.joints
        else:
            joints_to_include = tuple(self.get_joint(n) for n in joint_names)

        # Build position vector
        positions: list[sp.Expr] = []
        for joint in joints_to_include:
            x, y = joint.position_expr()
            positions.extend([x, y])

        # Compute Jacobian with respect to theta
        pos_matrix = sp.Matrix(positions)
        return pos_matrix.jacobian([self.theta])

    def simplify(self) -> SymbolicLinkage:
        """
        Return a new linkage with simplified expressions.

        Note: This creates new joint objects with simplified expressions,
        but the linkage structure is preserved.

        :returns: New SymbolicLinkage with simplified position expressions.
        """
        from .joints import SymCrank, SymRevolute, SymStatic

        new_joints: list[SymJoint] = []
        joint_map: dict[str, SymJoint] = {}

        for joint in self.joints:
            if isinstance(joint, SymStatic):
                x, y = joint.position_expr()
                new_joint: SymJoint = SymStatic(
                    x=sp.simplify(x),
                    y=sp.simplify(y),
                    name=joint.name,
                )
            elif isinstance(joint, SymCrank):
                parent: SymJoint | SymCoord
                if joint.parent0 is not None:
                    parent = joint_map[joint.parent0.name]
                elif joint._parent_coord is not None:
                    parent = joint._parent_coord
                else:
                    parent = (sp.Integer(0), sp.Integer(0))
                new_joint = SymCrank(
                    parent=parent,
                    radius=joint.r,
                    theta=joint.theta,
                    name=joint.name,
                )
            elif isinstance(joint, SymRevolute):
                assert joint.parent0 is not None and joint.parent1 is not None
                new_joint = SymRevolute(
                    parent0=joint_map[joint.parent0.name],
                    parent1=joint_map[joint.parent1.name],
                    distance0=joint.r0,
                    distance1=joint.r1,
                    branch=joint.branch,
                    name=joint.name,
                )
            else:
                raise TypeError(f"Unknown joint type: {type(joint)}")

            new_joints.append(new_joint)
            joint_map[joint.name] = new_joint

        return SymbolicLinkage(
            joints=new_joints,
            theta=self.theta,
            name=self.name,
        )

    def substitute(
        self,
        param_values: dict[str, float | sp.Expr],
    ) -> SymbolicLinkage:
        """
        Create a new linkage with parameters substituted.

        This is useful for partial evaluation or for substituting
        symbolic parameters with numeric values.

        :param param_values: Dictionary mapping parameter names to values.
        :returns: New SymbolicLinkage with substituted parameters.
        """
        from .joints import SymCrank, SymRevolute, SymStatic

        # Build substitution dict with Symbol keys
        subs = {self.parameters[k]: v for k, v in param_values.items()
                if k in self.parameters}

        new_joints: list[SymJoint] = []
        joint_map: dict[str, SymJoint] = {}

        for joint in self.joints:
            if isinstance(joint, SymStatic):
                x, y = joint.position_expr()
                new_joint: SymJoint = SymStatic(
                    x=x.subs(subs),
                    y=y.subs(subs),
                    name=joint.name,
                )
            elif isinstance(joint, SymCrank):
                parent: SymJoint | SymCoord
                if joint.parent0 is not None:
                    parent = joint_map[joint.parent0.name]
                elif joint._parent_coord is not None:
                    parent = joint._parent_coord
                else:
                    parent = (sp.Integer(0), sp.Integer(0))
                new_r = joint.r.subs(subs) if hasattr(joint.r, "subs") else joint.r
                new_joint = SymCrank(
                    parent=parent,
                    radius=new_r,
                    theta=joint.theta,
                    name=joint.name,
                )
            elif isinstance(joint, SymRevolute):
                assert joint.parent0 is not None and joint.parent1 is not None
                new_r0 = joint.r0.subs(subs) if hasattr(joint.r0, "subs") else joint.r0
                new_r1 = joint.r1.subs(subs) if hasattr(joint.r1, "subs") else joint.r1
                new_joint = SymRevolute(
                    parent0=joint_map[joint.parent0.name],
                    parent1=joint_map[joint.parent1.name],
                    distance0=new_r0,
                    distance1=new_r1,
                    branch=joint.branch,
                    name=joint.name,
                )
            else:
                raise TypeError(f"Unknown joint type: {type(joint)}")

            new_joints.append(new_joint)
            joint_map[joint.name] = new_joint

        return SymbolicLinkage(
            joints=new_joints,
            theta=self.theta,
            name=self.name,
        )
