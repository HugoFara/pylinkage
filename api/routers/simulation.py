"""Simulation endpoints."""

from fastapi import APIRouter, HTTPException

from ..models.schemas import (
    Position,
    SimulationFrame,
    SimulationRequest,
    SimulationResponse,
    TrajectoryResponse,
)
from ..services import linkage_service
from ..storage.memory import storage

router = APIRouter(prefix="/linkages", tags=["simulation"])


@router.post("/{linkage_id}/simulate", response_model=SimulationResponse)
def simulate_linkage(linkage_id: str, request: SimulationRequest) -> SimulationResponse:
    """Run simulation on a linkage."""
    stored = storage.get(linkage_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Linkage not found")

    # Build the linkage
    linkage_dict = {
        "name": stored.get("name"),
        "joints": stored.get("joints", []),
        "solve_order": stored.get("solve_order"),
    }
    linkage, is_buildable, error = linkage_service.validate_and_build(linkage_dict)

    if not is_buildable or linkage is None:
        return SimulationResponse(
            linkage_id=linkage_id,
            iterations=0,
            frames=[],
            joint_names=[],
            is_complete=False,
            error=error,
        )

    # Run simulation
    try:
        frames_data = linkage_service.run_simulation(
            linkage,
            iterations=request.iterations,
            dt=request.dt,
        )
        joint_names = linkage_service.get_joint_names(linkage)

        frames = [
            SimulationFrame(
                step=i,
                positions=[Position(x=pos[0], y=pos[1]) for pos in frame],
            )
            for i, frame in enumerate(frames_data)
        ]

        return SimulationResponse(
            linkage_id=linkage_id,
            iterations=len(frames),
            frames=frames,
            joint_names=joint_names,
            is_complete=True,
        )
    except Exception as e:
        return SimulationResponse(
            linkage_id=linkage_id,
            iterations=0,
            frames=[],
            joint_names=[],
            is_complete=False,
            error=str(e),
        )


@router.post("/{linkage_id}/trajectory", response_model=TrajectoryResponse)
def get_trajectory(linkage_id: str, request: SimulationRequest) -> TrajectoryResponse:
    """Get trajectory as compact array format (more efficient for large simulations)."""
    stored = storage.get(linkage_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Linkage not found")

    # Build the linkage
    linkage_dict = {
        "name": stored.get("name"),
        "joints": stored.get("joints", []),
        "solve_order": stored.get("solve_order"),
    }
    linkage, is_buildable, error = linkage_service.validate_and_build(linkage_dict)

    if not is_buildable or linkage is None:
        raise HTTPException(status_code=400, detail=error)

    # Run simulation
    frames_data = linkage_service.run_simulation(
        linkage,
        iterations=request.iterations,
        dt=request.dt,
    )
    joint_names = linkage_service.get_joint_names(linkage)

    # Convert to compact array format
    positions = [[[pos[0], pos[1]] for pos in frame] for frame in frames_data]

    return TrajectoryResponse(
        linkage_id=linkage_id,
        iterations=len(positions),
        positions=positions,
        joint_names=joint_names,
    )


@router.get("/{linkage_id}/rotation-period")
def get_rotation_period(linkage_id: str) -> dict[str, int | str | None]:
    """Get the rotation period for a linkage."""
    stored = storage.get(linkage_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Linkage not found")

    # Build the linkage
    linkage_dict = {
        "name": stored.get("name"),
        "joints": stored.get("joints", []),
        "solve_order": stored.get("solve_order"),
    }
    linkage, is_buildable, error = linkage_service.validate_and_build(linkage_dict)

    if not is_buildable or linkage is None:
        return {"rotation_period": None, "error": error}

    return {
        "rotation_period": linkage_service.get_rotation_period(linkage),
        "error": None,
    }
