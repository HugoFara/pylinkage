"""Pydantic schemas matching pylinkage serialization format.

These schemas match the JSON format produced by:
- src/pylinkage/linkage/serialization.py
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# Joint reference types
class JointRef(BaseModel):
    """Reference to another joint by name."""

    ref: str


class InlineStatic(BaseModel):
    """Inline static joint definition."""

    inline: Literal[True] = True
    type: Literal["Static"] = "Static"
    x: float
    y: float
    name: str | None = None


JointReference = JointRef | InlineStatic | None


# Joint schemas matching serialization.py format
class JointDict(BaseModel):
    """Joint dictionary matching pylinkage serialization format."""

    type: Literal["Static", "Crank", "Fixed", "Revolute", "Prismatic", "Linear"]
    name: str
    x: float | None = None
    y: float | None = None

    # Parent references (for Crank, Fixed, Revolute, Prismatic)
    joint0: dict[str, Any] | None = None
    joint1: dict[str, Any] | None = None
    joint2: dict[str, Any] | None = None  # For Prismatic

    # Type-specific attributes
    distance: float | None = None  # For Crank, Fixed
    angle: float | None = None  # For Crank, Fixed
    distance0: float | None = None  # For Revolute
    distance1: float | None = None  # For Revolute
    revolute_radius: float | None = None  # For Prismatic


# Linkage schemas
class LinkageCreate(BaseModel):
    """Schema for creating a new linkage."""

    name: str | None = None
    joints: list[JointDict]
    solve_order: list[str] | None = None


class LinkageUpdate(BaseModel):
    """Schema for updating an existing linkage."""

    name: str | None = None
    joints: list[JointDict] | None = None
    solve_order: list[str] | None = None


class LinkageResponse(BaseModel):
    """Schema for linkage API responses."""

    id: str
    name: str
    joints: list[JointDict]
    solve_order: list[str] | None = None
    created_at: datetime
    updated_at: datetime
    is_buildable: bool = True
    rotation_period: int | None = None
    error: str | None = None


class LinkageListItem(BaseModel):
    """Compact linkage info for list endpoints."""

    id: str
    name: str
    joint_count: int
    created_at: datetime
    is_buildable: bool


# Simulation schemas
class SimulationRequest(BaseModel):
    """Request for running a simulation."""

    iterations: int | None = None  # None = use rotation_period
    dt: float = 1.0


class Position(BaseModel):
    """2D position."""

    x: float
    y: float


class SimulationFrame(BaseModel):
    """Single frame of simulation results."""

    step: int
    positions: list[Position]


class SimulationResponse(BaseModel):
    """Complete simulation results."""

    linkage_id: str
    iterations: int
    frames: list[SimulationFrame]
    joint_names: list[str]
    is_complete: bool = True
    error: str | None = None


class TrajectoryResponse(BaseModel):
    """Compact trajectory format (array-based for efficiency)."""

    linkage_id: str
    iterations: int
    # Shape conceptually: (iterations, n_joints, 2)
    # Stored as list of frames, each frame is list of [x, y] pairs
    positions: list[list[list[float]]]
    joint_names: list[str]


# Example schemas
class ExampleInfo(BaseModel):
    """Information about a prebuilt example."""

    name: str
    description: str
    joint_count: int
