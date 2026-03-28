"""Optimization endpoints for linkage parameter optimization."""

from fastapi import APIRouter, HTTPException

from ..models.optimization_schemas import (
    OptimizationRequest,
    OptimizationResponse,
)
from ..services import optimization_service

router = APIRouter(prefix="/optimization", tags=["optimization"])


@router.post("", response_model=OptimizationResponse)
def optimize(request: OptimizationRequest) -> OptimizationResponse:
    """Optimize a mechanism's constraints to maximize/minimize an objective.

    Supported algorithms: pso, differential_evolution, nelder_mead, grid_search.
    Supported objectives: path_length, bounding_box_area, x_extent, y_extent, target_path.
    """
    try:
        return optimization_service.run_optimization(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
