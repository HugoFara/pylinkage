"""Type definitions for pylinkage."""

import sys
from collections.abc import Callable, Sequence

# TypeAlias is available in typing from Python 3.10+
# For 3.9 compatibility, we use typing_extensions
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

# Basic coordinate types
Coord: TypeAlias = tuple[float, float]
"""A 2D coordinate as (x, y)."""

Coord3: TypeAlias = tuple[float, float, float]
"""A 3-element tuple, typically used for circles (x, y, radius)."""

# Locus types (path traced by a joint)
Locus: TypeAlias = tuple[Coord, ...]
"""A sequence of coordinates representing a path traced by a single joint."""

Loci: TypeAlias = tuple[Locus, ...]
"""Multiple loci, one per joint in a linkage."""

# Constraint types
Constraints: TypeAlias = tuple[float, ...]
"""A sequence of geometric constraints (distances, angles)."""

Bounds: TypeAlias = tuple[Sequence[float], Sequence[float]]
"""Optimization bounds as (lower_bounds, upper_bounds)."""

# Function types
FitnessFunc: TypeAlias = Callable[..., float]
"""A fitness function that returns a score."""

# Joint position types
JointPositions: TypeAlias = Sequence[Coord]
"""A sequence of joint coordinates."""

# Circle type for geometry operations
Circle: TypeAlias = tuple[float, float, float]
"""A circle defined as (center_x, center_y, radius)."""

# Line type for geometry operations (cartesian equation ax + by + c = 0)
Line: TypeAlias = tuple[float, float, float]
"""A line in cartesian form (a, b, c) representing ax + by + c = 0."""

# Bounding box type
BoundingBox: TypeAlias = tuple[float, float, float, float]
"""Bounding box as (y_min, x_max, y_max, x_min)."""
