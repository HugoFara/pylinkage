"""Symbolic joint classes for linkage analysis.

This module provides symbolic versions of the joint classes from
pylinkage.joints, enabling closed-form expressions for linkage trajectories.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import sympy as sp

from ._types import SymCoord
from ._types import theta as default_theta
from .geometry import symbolic_circle_intersect, symbolic_cyl_to_cart

if TYPE_CHECKING:
    from collections.abc import Set


class SymJoint(abc.ABC):
    """
    Abstract base class for symbolic joints.

    A symbolic joint represents a kinematic joint with its position
    expressed as symbolic SymPy expressions. This enables:
    - Closed-form trajectory analysis
    - Symbolic differentiation for optimization
    - Algebraic manipulation of linkage equations
    """

    __slots__ = ("name", "_x", "_y", "parent0", "parent1")

    name: str
    _x: sp.Expr | None
    _y: sp.Expr | None
    parent0: SymJoint | None
    parent1: SymJoint | None

    def __init__(
        self,
        x: float | sp.Expr | None = None,
        y: float | sp.Expr | None = None,
        parent0: SymJoint | None = None,
        parent1: SymJoint | None = None,
        name: str | None = None,
    ) -> None:
        """
        Create a symbolic joint.

        :param x: Initial or symbolic x position.
        :param y: Initial or symbolic y position.
        :param parent0: First parent joint (for kinematic constraints).
        :param parent1: Second parent joint (for kinematic constraints).
        :param name: Human-readable name for the joint.
        """
        self._x = sp.sympify(x) if x is not None else None
        self._y = sp.sympify(y) if y is not None else None
        self.parent0 = parent0
        self.parent1 = parent1
        self.name = name if name is not None else str(id(self))

    def __repr__(self) -> str:
        """Return a string representation of the joint."""
        return f"{self.__class__.__name__}(name={self.name!r})"

    @abc.abstractmethod
    def position_expr(self) -> SymCoord:
        """
        Return the symbolic (x, y) position expressions.

        :returns: Tuple of (x_expr, y_expr) SymPy expressions.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def constraint_equations(self) -> list[sp.Eq]:
        """
        Return the constraint equations this joint satisfies.

        :returns: List of SymPy Eq objects representing constraints.
        """
        raise NotImplementedError

    @property
    def parameters(self) -> Set[sp.Symbol]:
        """
        Return all symbolic parameters in this joint's position expressions.

        Excludes the input angle theta.

        :returns: Set of SymPy symbols that are parameters.
        """
        x_expr, y_expr = self.position_expr()
        all_symbols = x_expr.free_symbols | y_expr.free_symbols
        # Exclude theta (the input angle)
        result: Set[sp.Symbol] = set(all_symbols) - {default_theta}
        return result


class SymStatic(SymJoint):
    """
    A symbolic static joint with fixed position.

    This represents a ground point or fixed anchor in the linkage.
    The position can be numeric constants or symbolic parameters.
    """

    __slots__ = ()

    def __init__(
        self,
        x: float | sp.Expr = 0,
        y: float | sp.Expr = 0,
        name: str | None = None,
    ) -> None:
        """
        Create a static joint at a fixed position.

        :param x: X coordinate (numeric or symbolic). Default is 0.
        :param y: Y coordinate (numeric or symbolic). Default is 0.
        :param name: Human-readable name for the joint.
        """
        super().__init__(x=x, y=y, name=name)

    def position_expr(self) -> SymCoord:
        """Return the constant position expressions."""
        assert self._x is not None and self._y is not None
        return (self._x, self._y)

    def constraint_equations(self) -> list[sp.Eq]:
        """Static joints have no constraints."""
        return []


class SymCrank(SymJoint):
    """
    A symbolic crank (motor) joint.

    The crank rotates around its parent at a constant radius.
    Its position is parameterized by the input angle theta.
    """

    __slots__ = ("r", "theta", "_parent_coord", "_numeric_r")

    r: sp.Symbol | sp.Expr
    theta: sp.Symbol
    _parent_coord: SymCoord | None
    _numeric_r: float | None

    def __init__(
        self,
        parent: SymJoint | SymCoord,
        radius: float | sp.Expr | str = "r",
        theta: sp.Symbol | None = None,
        name: str | None = None,
    ) -> None:
        """
        Create a crank joint rotating around a parent.

        :param parent: Parent joint or fixed coordinate (x, y).
        :param radius: Distance from parent. Can be numeric, symbolic, or
            a string to create a new symbol. Default is "r".
        :param theta: Input angle symbol. Default uses the global theta.
        :param name: Human-readable name for the joint.
        """
        if isinstance(parent, SymJoint):
            super().__init__(parent0=parent, name=name)
            self._parent_coord = None
        else:
            super().__init__(name=name)
            self._parent_coord = (sp.sympify(parent[0]), sp.sympify(parent[1]))

        # Handle radius parameter
        if isinstance(radius, str):
            self.r = sp.Symbol(radius, positive=True, real=True)
        else:
            self.r = sp.sympify(radius)

        # Use provided theta or default
        self.theta = theta if theta is not None else default_theta

        # Initialize numeric value storage
        self._numeric_r = None

    def position_expr(self) -> SymCoord:
        """
        Return the crank position as a function of theta.

        Position: (x0 + r*cos(theta), y0 + r*sin(theta))
        """
        if self.parent0 is not None:
            x0, y0 = self.parent0.position_expr()
        elif self._parent_coord is not None:
            x0, y0 = self._parent_coord
        else:
            x0, y0 = sp.Integer(0), sp.Integer(0)

        return symbolic_cyl_to_cart(self.r, self.theta, x0, y0)

    def constraint_equations(self) -> list[sp.Eq]:
        """
        Return the distance constraint from crank to parent.

        Constraint: dist(self, parent) = r
        """
        x, y = self.position_expr()
        if self.parent0 is not None:
            x0, y0 = self.parent0.position_expr()
        elif self._parent_coord is not None:
            x0, y0 = self._parent_coord
        else:
            x0, y0 = sp.Integer(0), sp.Integer(0)

        return [sp.Eq((x - x0) ** 2 + (y - y0) ** 2, self.r**2)]

    @property
    def parameters(self) -> Set[sp.Symbol]:
        """Return parameters, including radius if symbolic."""
        params = super().parameters
        if isinstance(self.r, sp.Symbol):
            params = params | {self.r}
        return params


class SymRevolute(SymJoint):
    """
    A symbolic revolute (pin) joint connecting two parent joints.

    The revolute joint is constrained to be at fixed distances from
    two parent joints, which geometrically defines the intersection
    of two circles. The branch parameter selects which intersection.
    """

    __slots__ = ("r0", "r1", "branch", "_numeric_r0", "_numeric_r1")

    r0: sp.Symbol | sp.Expr
    r1: sp.Symbol | sp.Expr
    branch: int
    _numeric_r0: float | None
    _numeric_r1: float | None

    def __init__(
        self,
        parent0: SymJoint,
        parent1: SymJoint,
        distance0: float | sp.Expr | str = "r0",
        distance1: float | sp.Expr | str = "r1",
        branch: int = 1,
        name: str | None = None,
    ) -> None:
        """
        Create a revolute joint connecting two parents.

        :param parent0: First parent joint.
        :param parent1: Second parent joint.
        :param distance0: Distance to parent0. Can be numeric, symbolic, or
            a string to create a new symbol. Default is "r0".
        :param distance1: Distance to parent1. Can be numeric, symbolic, or
            a string to create a new symbol. Default is "r1".
        :param branch: +1 or -1 to select which circle intersection.
            Default is +1.
        :param name: Human-readable name for the joint.
        """
        super().__init__(parent0=parent0, parent1=parent1, name=name)

        # Handle distance parameters
        if isinstance(distance0, str):
            self.r0 = sp.Symbol(distance0, positive=True, real=True)
        else:
            self.r0 = sp.sympify(distance0)

        if isinstance(distance1, str):
            self.r1 = sp.Symbol(distance1, positive=True, real=True)
        else:
            self.r1 = sp.sympify(distance1)

        self.branch = branch

        # Initialize numeric value storage
        self._numeric_r0 = None
        self._numeric_r1 = None

    def position_expr(self) -> SymCoord:
        """
        Return the revolute position from circle-circle intersection.

        The position is at the intersection of:
        - Circle centered at parent0 with radius r0
        - Circle centered at parent1 with radius r1
        """
        assert self.parent0 is not None and self.parent1 is not None

        x1, y1 = self.parent0.position_expr()
        x2, y2 = self.parent1.position_expr()

        return symbolic_circle_intersect(
            x1,
            y1,
            self.r0,
            x2,
            y2,
            self.r1,
            self.branch,
        )

    def constraint_equations(self) -> list[sp.Eq]:
        """
        Return the distance constraints to both parents.

        Constraints:
        - dist(self, parent0) = r0
        - dist(self, parent1) = r1
        """
        assert self.parent0 is not None and self.parent1 is not None

        x, y = self.position_expr()
        x1, y1 = self.parent0.position_expr()
        x2, y2 = self.parent1.position_expr()

        return [
            sp.Eq((x - x1) ** 2 + (y - y1) ** 2, self.r0**2),
            sp.Eq((x - x2) ** 2 + (y - y2) ** 2, self.r1**2),
        ]

    @property
    def parameters(self) -> Set[sp.Symbol]:
        """Return parameters, including distances if symbolic."""
        params = super().parameters
        if isinstance(self.r0, sp.Symbol):
            params = params | {self.r0}
        if isinstance(self.r1, sp.Symbol):
            params = params | {self.r1}
        return params
