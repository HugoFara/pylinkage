"""Type definitions for the symbolic module."""

from typing import TypeAlias

import sympy as sp

# Symbolic coordinate: a tuple of two SymPy expressions (x, y)
SymCoord: TypeAlias = tuple[sp.Expr, sp.Expr]

# Symbolic constraints: tuple of SymPy expressions
SymConstraints: TypeAlias = tuple[sp.Expr, ...]

# Standard symbols used throughout the module
theta = sp.Symbol("theta", real=True)  # Input angle (crank rotation)
