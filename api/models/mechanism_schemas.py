"""Pydantic schemas matching mechanism module serialization format.

These schemas match the JSON format produced by:
- src/pylinkage/mechanism/serialization.py
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


# Joint schemas
class GroundJointSchema(BaseModel):
    """Ground joint (fixed to frame)."""

    id: str
    type: Literal["ground"] = "ground"
    position: list[float | None]
    name: str | None = None


class RevoluteJointSchema(BaseModel):
    """Revolute joint (pin joint)."""

    id: str
    type: Literal["revolute"] = "revolute"
    position: list[float | None]
    name: str | None = None


class PrismaticJointSchema(BaseModel):
    """Prismatic joint (slider)."""

    id: str
    type: Literal["prismatic"] = "prismatic"
    position: list[float | None]
    name: str | None = None
    axis: list[float] = [1.0, 0.0]
    slide_distance: float = 0.0


# Link schemas
class GroundLinkSchema(BaseModel):
    """Ground link (stationary frame)."""

    id: str
    type: Literal["ground"] = "ground"
    joints: list[str]
    name: str | None = None


class DriverLinkSchema(BaseModel):
    """Driver link (motor-driven crank)."""

    id: str
    type: Literal["driver"] = "driver"
    joints: list[str]
    motor_joint: str
    angular_velocity: float = 0.1
    initial_angle: float = 0.0
    name: str | None = None


class RegularLinkSchema(BaseModel):
    """Regular rigid link."""

    id: str
    type: Literal["link"] = "link"
    joints: list[str]
    name: str | None = None


# Mechanism schemas
class MechanismCreate(BaseModel):
    """Schema for creating a new mechanism."""

    name: str | None = None
    joints: list[dict]
    links: list[dict]
    ground: str | None = None


class MechanismUpdate(BaseModel):
    """Schema for updating an existing mechanism."""

    name: str | None = None
    joints: list[dict] | None = None
    links: list[dict] | None = None
    ground: str | None = None


class MechanismResponse(BaseModel):
    """Schema for mechanism API responses."""

    id: str
    name: str
    joints: list[dict]
    links: list[dict]
    ground: str | None = None
    created_at: datetime
    updated_at: datetime
    is_buildable: bool = True
    rotation_period: int | None = None
    error: str | None = None


class MechanismListItem(BaseModel):
    """Compact mechanism info for list endpoints."""

    id: str
    name: str
    joint_count: int
    link_count: int
    created_at: datetime
    is_buildable: bool


# Simulation schemas
class SimulationRequest(BaseModel):
    """Request for running a simulation."""

    iterations: int | None = None
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

    mechanism_id: str
    iterations: int
    frames: list[SimulationFrame]
    joint_names: list[str]
    is_complete: bool = True
    error: str | None = None


class TrajectoryResponse(BaseModel):
    """Compact trajectory format (array-based for efficiency)."""

    mechanism_id: str
    iterations: int
    positions: list[list[list[float]]]
    joint_names: list[str]


# Example schemas
class ExampleInfo(BaseModel):
    """Information about a prebuilt example."""

    name: str
    description: str
    joint_count: int
    link_count: int
