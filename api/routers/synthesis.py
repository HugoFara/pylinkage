"""Synthesis endpoints for generating four-bar linkages from requirements."""

from fastapi import APIRouter, HTTPException

from ..models.synthesis_schemas import (
    FunctionGenerationRequest,
    MotionGenerationRequest,
    PathGenerationRequest,
    SynthesisResponse,
    TopologyGenerationRequest,
    TopologySynthesisResponse,
)
from ..services import synthesis_service

router = APIRouter(prefix="/synthesis", tags=["synthesis"])


@router.post("/path-generation", response_model=SynthesisResponse)
def path_generation(request: PathGenerationRequest) -> SynthesisResponse:
    """Generate four-bar linkages where coupler passes through precision points."""
    try:
        return synthesis_service.run_path_generation(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/function-generation", response_model=SynthesisResponse)
def function_generation(
    request: FunctionGenerationRequest,
) -> SynthesisResponse:
    """Generate four-bar linkages matching input/output angle relationships."""
    try:
        return synthesis_service.run_function_generation(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/motion-generation", response_model=SynthesisResponse)
def motion_generation(request: MotionGenerationRequest) -> SynthesisResponse:
    """Generate four-bar linkages guiding a body through specified poses."""
    try:
        return synthesis_service.run_motion_generation(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/topology-generation", response_model=TopologySynthesisResponse)
def topology_generation(
    request: TopologyGenerationRequest,
) -> TopologySynthesisResponse:
    """Generate linkages across multiple topologies (4-bar through 8-bar)."""
    try:
        return synthesis_service.run_topology_generation(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
