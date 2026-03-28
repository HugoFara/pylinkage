"""Simulation endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..models.mechanism_schemas import (
    MechanismCreate,
    Position,
    SimulationFrame,
    SimulationRequest,
    SimulationResponse,
    TrajectoryResponse,
)
from ..services import mechanism_service
from ..storage.memory import storage

router = APIRouter(prefix="/mechanisms", tags=["simulation"])


class DirectSimulationRequest(BaseModel):
    """Request for direct simulation with mechanism data."""

    mechanism: MechanismCreate
    iterations: int | None = None
    dt: float = 1.0


@router.post("/simulate", response_model=SimulationResponse)
def simulate_direct(request: DirectSimulationRequest) -> SimulationResponse:
    """Run simulation directly on mechanism data (no storage required)."""
    mechanism_dict = {
        "name": request.mechanism.name,
        "joints": request.mechanism.joints,  # Already dicts
        "links": request.mechanism.links,  # Already dicts
        "ground": request.mechanism.ground,
    }
    mechanism, is_buildable, error = mechanism_service.validate_and_build(mechanism_dict)

    if not is_buildable or mechanism is None:
        return SimulationResponse(
            mechanism_id="direct",
            iterations=0,
            frames=[],
            joint_names=[],
            is_complete=False,
            error=error,
        )

    # Run simulation
    try:
        frames_data = mechanism_service.run_simulation(
            mechanism,
            iterations=request.iterations,
            dt=request.dt,
        )
        joint_names = mechanism_service.get_joint_names(mechanism)

        frames = [
            SimulationFrame(
                step=i,
                positions=[Position(x=pos[0], y=pos[1]) for pos in frame],
            )
            for i, frame in enumerate(frames_data)
        ]

        return SimulationResponse(
            mechanism_id="direct",
            iterations=len(frames),
            frames=frames,
            joint_names=joint_names,
            is_complete=True,
        )
    except Exception as e:
        return SimulationResponse(
            mechanism_id="direct",
            iterations=0,
            frames=[],
            joint_names=[],
            is_complete=False,
            error=str(e),
        )


@router.post("/{mechanism_id}/simulate", response_model=SimulationResponse)
def simulate_mechanism(mechanism_id: str, request: SimulationRequest) -> SimulationResponse:
    """Run simulation on a mechanism."""
    stored = storage.get(mechanism_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Mechanism not found")

    # Build the mechanism
    mechanism_dict = {
        "name": stored.get("name"),
        "joints": stored.get("joints", []),
        "links": stored.get("links", []),
        "ground": stored.get("ground"),
    }
    mechanism, is_buildable, error = mechanism_service.validate_and_build(mechanism_dict)

    if not is_buildable or mechanism is None:
        return SimulationResponse(
            mechanism_id=mechanism_id,
            iterations=0,
            frames=[],
            joint_names=[],
            is_complete=False,
            error=error,
        )

    # Run simulation
    try:
        frames_data = mechanism_service.run_simulation(
            mechanism,
            iterations=request.iterations,
            dt=request.dt,
        )
        joint_names = mechanism_service.get_joint_names(mechanism)

        frames = [
            SimulationFrame(
                step=i,
                positions=[Position(x=pos[0], y=pos[1]) for pos in frame],
            )
            for i, frame in enumerate(frames_data)
        ]

        return SimulationResponse(
            mechanism_id=mechanism_id,
            iterations=len(frames),
            frames=frames,
            joint_names=joint_names,
            is_complete=True,
        )
    except Exception as e:
        return SimulationResponse(
            mechanism_id=mechanism_id,
            iterations=0,
            frames=[],
            joint_names=[],
            is_complete=False,
            error=str(e),
        )


@router.post("/{mechanism_id}/trajectory", response_model=TrajectoryResponse)
def get_trajectory(mechanism_id: str, request: SimulationRequest) -> TrajectoryResponse:
    """Get trajectory as compact array format (more efficient for large simulations)."""
    stored = storage.get(mechanism_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Mechanism not found")

    # Build the mechanism
    mechanism_dict = {
        "name": stored.get("name"),
        "joints": stored.get("joints", []),
        "links": stored.get("links", []),
        "ground": stored.get("ground"),
    }
    mechanism, is_buildable, error = mechanism_service.validate_and_build(mechanism_dict)

    if not is_buildable or mechanism is None:
        raise HTTPException(status_code=400, detail=error)

    # Run simulation
    frames_data = mechanism_service.run_simulation(
        mechanism,
        iterations=request.iterations,
        dt=request.dt,
    )
    joint_names = mechanism_service.get_joint_names(mechanism)

    # Convert to compact array format
    positions = [[[pos[0], pos[1]] for pos in frame] for frame in frames_data]

    return TrajectoryResponse(
        mechanism_id=mechanism_id,
        iterations=len(positions),
        positions=positions,
        joint_names=joint_names,
    )


@router.get("/{mechanism_id}/rotation-period")
def get_rotation_period(mechanism_id: str) -> dict[str, int | str | None]:
    """Get the rotation period for a mechanism."""
    stored = storage.get(mechanism_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Mechanism not found")

    # Build the mechanism
    mechanism_dict = {
        "name": stored.get("name"),
        "joints": stored.get("joints", []),
        "links": stored.get("links", []),
        "ground": stored.get("ground"),
    }
    mechanism, is_buildable, error = mechanism_service.validate_and_build(mechanism_dict)

    if not is_buildable or mechanism is None:
        return {"rotation_period": None, "error": error}

    return {
        "rotation_period": mechanism_service.get_rotation_period(mechanism),
        "error": None,
    }
