"""Pydantic schemas for synthesis endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PointInput(BaseModel):
    """A 2D point."""

    x: float
    y: float


class AnglePairInput(BaseModel):
    """Input/output angle pair for function generation (radians)."""

    theta_in: float
    theta_out: float


class PoseInput(BaseModel):
    """Pose for motion generation: position + orientation angle (radians)."""

    x: float
    y: float
    angle: float


# --- Requests ---


class PathGenerationRequest(BaseModel):
    """Request for path generation synthesis."""

    precision_points: list[PointInput] = Field(..., min_length=3, max_length=10)
    max_solutions: int = Field(default=10, ge=1, le=50)
    require_grashof: bool = True
    require_crank_rocker: bool = False


class FunctionGenerationRequest(BaseModel):
    """Request for function generation synthesis."""

    angle_pairs: list[AnglePairInput] = Field(..., min_length=3, max_length=10)
    ground_length: float = Field(default=100.0, gt=0)
    require_grashof: bool = True
    require_crank_rocker: bool = False


class MotionGenerationRequest(BaseModel):
    """Request for motion generation synthesis."""

    poses: list[PoseInput] = Field(..., min_length=3, max_length=5)
    max_solutions: int = Field(default=10, ge=1, le=50)
    require_grashof: bool = True
    require_crank_rocker: bool = False


# --- Responses ---


class FourBarSolutionDTO(BaseModel):
    """Serialized four-bar solution geometry."""

    ground_pivot_a: list[float]
    ground_pivot_d: list[float]
    crank_pivot_b: list[float]
    coupler_pivot_c: list[float]
    crank_length: float
    coupler_length: float
    rocker_length: float
    ground_length: float
    coupler_point: list[float] | None = None
    grashof_type: str | None = None


class SynthesisResponse(BaseModel):
    """Response from any synthesis endpoint."""

    solutions: list[FourBarSolutionDTO]
    mechanism_dicts: list[dict[str, Any]]
    warnings: list[str]
    solution_count: int


# --- Topology-aware synthesis ---


class TopologyGenerationRequest(BaseModel):
    """Request for multi-topology synthesis (4-bar through 8-bar)."""

    precision_points: list[PointInput] = Field(..., min_length=3, max_length=10)
    max_links: int = Field(default=6, ge=4, le=8)
    max_solutions: int = Field(default=20, ge=1, le=50)
    max_solutions_per_topology: int = Field(default=5, ge=1, le=20)


class QualityMetricsDTO(BaseModel):
    """Quality metrics for a topology synthesis solution."""

    path_accuracy: float
    min_transmission_angle: float
    link_ratio: float
    compactness: float
    num_links: int
    is_grashof: bool
    overall_score: float


class TopologySolutionDTO(BaseModel):
    """A single solution from multi-topology synthesis."""

    topology_id: str
    topology_name: str
    family: str
    num_links: int
    metrics: QualityMetricsDTO
    mechanism_dict: dict[str, Any]


class TopologySynthesisResponse(BaseModel):
    """Response from topology-aware synthesis."""

    solutions: list[TopologySolutionDTO]
    warnings: list[str]
    solution_count: int
