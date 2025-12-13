"""
Type definitions for pylinkage.

This module defines type aliases used throughout the pylinkage library to ensure
consistent typing and improve code readability. These types represent the core
geometric and mathematical concepts used in linkage simulation and optimization.

Type Categories:
    Coordinates: Coord, MaybeCoord, Coord3 - represent points and circles in 2D space
    Paths: Locus, Loci - represent trajectories traced by joints during motion
    Constraints: Constraints, Bounds - represent geometric constraints and optimization bounds
    Geometry: Circle, Line, BoundingBox - represent geometric primitives
    Functions: FitnessFunc - callable types for optimization
    Positions: JointPositions - represent joint state during simulation
"""

from collections.abc import Callable, Sequence
from typing import TypeAlias

# Basic coordinate types
Coord: TypeAlias = tuple[float, float]
"""A 2D coordinate as (x, y)."""

# Coordinate that may contain None (for uninitialized joints)
MaybeCoord: TypeAlias = tuple[float | None, float | None]
"""A 2D coordinate that may contain None values."""

Coord3: TypeAlias = tuple[float, float, float]
"""A 3-element tuple, typically used for circles (x, y, radius)."""

# Locus types (path traced by a joint)
Locus: TypeAlias = tuple[Coord, ...]
"""A sequence of coordinates representing a path traced by a single joint."""

Loci: TypeAlias = tuple[Locus, ...]
"""Multiple loci, one per joint in a linkage."""

# Constraint types - may contain None for uninitialized constraints
Constraints: TypeAlias = tuple[float | None, ...]
"""A sequence of geometric constraints (distances, angles), may contain None."""

Bounds: TypeAlias = tuple[Sequence[float], Sequence[float]]
"""Optimization bounds as (lower_bounds, upper_bounds)."""

# Function types
FitnessFunc: TypeAlias = Callable[..., float]
"""A fitness function that returns a score."""

# Joint position types - may contain None for uninitialized positions
JointPositions: TypeAlias = Sequence[MaybeCoord]
"""A sequence of joint coordinates, may contain None values."""

# Circle type for geometry operations
Circle: TypeAlias = tuple[float, float, float]
"""A circle defined as (center_x, center_y, radius)."""

# Line type for geometry operations (cartesian equation ax + by + c = 0)
Line: TypeAlias = tuple[float, float, float]
"""A line in cartesian form (a, b, c) representing ax + by + c = 0."""

# Bounding box type
BoundingBox: TypeAlias = tuple[float, float, float, float]
"""Bounding box as (y_min, x_max, y_max, x_min)."""
