"""Mechanism CRUD endpoints."""

from fastapi import APIRouter, HTTPException

from ..models.mechanism_schemas import (
    MechanismCreate,
    MechanismListItem,
    MechanismResponse,
    MechanismUpdate,
)
from ..services import mechanism_service
from ..storage.memory import storage

router = APIRouter(prefix="/mechanisms", tags=["mechanisms"])


@router.post("/", response_model=MechanismResponse, status_code=201)
def create_mechanism(data: MechanismCreate) -> MechanismResponse:
    """Create a new mechanism."""
    mechanism_dict = data.model_dump(exclude_none=True)

    # Validate buildability
    mechanism, is_buildable, error = mechanism_service.validate_and_build(mechanism_dict)

    # Store the mechanism
    mechanism_id = storage.create(mechanism_dict)
    stored = storage.get(mechanism_id)

    if stored is None:
        raise HTTPException(status_code=500, detail="Failed to create mechanism")

    response_data = mechanism_service.mechanism_to_response_dict(
        mechanism_id, stored, mechanism, is_buildable, error
    )
    return MechanismResponse(**response_data)


@router.get("", response_model=list[MechanismListItem])
def list_mechanisms(skip: int = 0, limit: int = 100) -> list[MechanismListItem]:
    """List all mechanisms."""
    items = storage.list_all(skip=skip, limit=limit)
    return [
        MechanismListItem(
            id=item["id"],
            name=item.get("name", "Unnamed"),
            joint_count=len(item.get("joints", [])),
            link_count=len(item.get("links", [])),
            created_at=item["created_at"],
            is_buildable=item.get("is_buildable", True),
        )
        for item in items
    ]


@router.get("/{mechanism_id}", response_model=MechanismResponse)
def get_mechanism(mechanism_id: str) -> MechanismResponse:
    """Get a mechanism by ID."""
    stored = storage.get(mechanism_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Mechanism not found")

    # Try to build to get current state
    mechanism_dict = {
        "name": stored.get("name"),
        "joints": stored.get("joints", []),
        "links": stored.get("links", []),
        "ground": stored.get("ground"),
    }
    mechanism, is_buildable, error = mechanism_service.validate_and_build(mechanism_dict)

    response_data = mechanism_service.mechanism_to_response_dict(
        mechanism_id, stored, mechanism, is_buildable, error
    )
    return MechanismResponse(**response_data)


@router.put("/{mechanism_id}", response_model=MechanismResponse)
def update_mechanism(mechanism_id: str, data: MechanismUpdate) -> MechanismResponse:
    """Update an existing mechanism."""
    existing = storage.get(mechanism_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Mechanism not found")

    # Merge update data
    update_data = data.model_dump(exclude_none=True)

    if update_data:
        storage.update(mechanism_id, update_data)

    stored = storage.get(mechanism_id)
    if stored is None:
        raise HTTPException(status_code=500, detail="Failed to update mechanism")

    # Build the updated mechanism
    mechanism_dict = {
        "name": stored.get("name"),
        "joints": stored.get("joints", []),
        "links": stored.get("links", []),
        "ground": stored.get("ground"),
    }
    mechanism, is_buildable, error = mechanism_service.validate_and_build(mechanism_dict)

    response_data = mechanism_service.mechanism_to_response_dict(
        mechanism_id, stored, mechanism, is_buildable, error
    )
    return MechanismResponse(**response_data)


@router.delete("/{mechanism_id}", status_code=204)
def delete_mechanism(mechanism_id: str) -> None:
    """Delete a mechanism."""
    if not storage.delete(mechanism_id):
        raise HTTPException(status_code=404, detail="Mechanism not found")
