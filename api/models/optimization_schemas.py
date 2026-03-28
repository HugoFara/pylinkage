"""Pydantic schemas for optimization endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# --- Objective definitions ---


class PathLengthObjective(BaseModel):
    """Maximize or minimize the total path length traced by a joint."""

    type: Literal["path_length"] = "path_length"
    joint_index: int = Field(..., ge=0, description="Index of the joint to measure")


class BoundingBoxAreaObjective(BaseModel):
    """Minimize the bounding box area of a joint's path."""

    type: Literal["bounding_box_area"] = "bounding_box_area"
    joint_index: int = Field(..., ge=0, description="Index of the joint to measure")


class XExtentObjective(BaseModel):
    """Maximize horizontal extent (stride) of a joint's path."""

    type: Literal["x_extent"] = "x_extent"
    joint_index: int = Field(..., ge=0, description="Index of the joint to measure")


class YExtentObjective(BaseModel):
    """Maximize vertical extent of a joint's path."""

    type: Literal["y_extent"] = "y_extent"
    joint_index: int = Field(..., ge=0, description="Index of the joint to measure")


class TargetPathObjective(BaseModel):
    """Minimize distance between a joint's path and target points."""

    type: Literal["target_path"] = "target_path"
    joint_index: int = Field(..., ge=0, description="Index of the joint to measure")
    target_points: list[list[float]] = Field(
        ..., min_length=2, description="Target points [[x, y], ...]"
    )


ObjectiveSpec = (
    PathLengthObjective
    | BoundingBoxAreaObjective
    | XExtentObjective
    | YExtentObjective
    | TargetPathObjective
)


# --- Algorithm parameters ---


class PSOParams(BaseModel):
    """Parameters for Particle Swarm Optimization."""

    algorithm: Literal["pso"] = "pso"
    n_particles: int = Field(default=30, ge=1, le=500)
    iterations: int = Field(default=100, ge=1, le=5000)
    inertia: float = Field(default=0.6, ge=0.0, le=2.0)
    leader: float = Field(default=3.0, ge=0.0, le=10.0)
    follower: float = Field(default=0.1, ge=0.0, le=10.0)
    neighbors: int = Field(default=17, ge=1)


class DifferentialEvolutionParams(BaseModel):
    """Parameters for Differential Evolution."""

    algorithm: Literal["differential_evolution"] = "differential_evolution"
    max_iterations: int = Field(default=200, ge=1, le=5000)
    population_size: int = Field(default=15, ge=1, le=100)
    tolerance: float = Field(default=0.01, gt=0.0)
    mutation: list[float] = Field(default=[0.5, 1.0])
    recombination: float = Field(default=0.7, ge=0.0, le=1.0)
    strategy: str = "best1bin"
    seed: int | None = None


class NelderMeadParams(BaseModel):
    """Parameters for Nelder-Mead local optimization."""

    algorithm: Literal["nelder_mead"] = "nelder_mead"
    max_iterations: int = Field(default=1000, ge=1, le=50000)
    tolerance: float | None = None


class GridSearchParams(BaseModel):
    """Parameters for grid search (trials and errors)."""

    algorithm: Literal["grid_search"] = "grid_search"
    divisions: int = Field(default=5, ge=2, le=20)
    n_results: int = Field(default=5, ge=1, le=50)


AlgorithmParams = PSOParams | DifferentialEvolutionParams | NelderMeadParams | GridSearchParams


# --- Requests ---


class OptimizationRequest(BaseModel):
    """Request for running optimization on a mechanism."""

    mechanism: dict[str, Any] = Field(
        ..., description="Mechanism dict (same format as MechanismCreate)"
    )
    objective: ObjectiveSpec = Field(
        ..., discriminator="type", description="Objective function specification"
    )
    algorithm: AlgorithmParams = Field(
        default_factory=PSOParams, discriminator="algorithm",
        description="Optimization algorithm and parameters"
    )
    minimize: bool = Field(
        default=False,
        description="If true, minimize the objective. Otherwise maximize.",
    )
    bounds_factor: float = Field(
        default=5.0, gt=0.0,
        description="Factor for auto-generating constraint bounds (center * factor).",
    )


# --- Responses ---


class OptimizationResultDTO(BaseModel):
    """Single optimization result."""

    score: float
    constraints: list[float]
    mechanism_dict: dict[str, Any] | None = None


class OptimizationResponse(BaseModel):
    """Response from optimization endpoint."""

    results: list[OptimizationResultDTO]
    best_score: float | None = None
    constraint_names: list[str]
    warnings: list[str] = Field(default_factory=list)
