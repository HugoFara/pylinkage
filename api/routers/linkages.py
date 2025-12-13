"""Linkage CRUD endpoints."""

from fastapi import APIRouter, HTTPException

from ..models.schemas import (
    LinkageCreate,
    LinkageListItem,
    LinkageResponse,
    LinkageUpdate,
)
from ..services import linkage_service
from ..storage.memory import storage

router = APIRouter(prefix="/linkages", tags=["linkages"])


@router.post("/", response_model=LinkageResponse, status_code=201)
def create_linkage(data: LinkageCreate) -> LinkageResponse:
    """Create a new linkage."""
    linkage_dict = data.model_dump(exclude_none=True)

    # Validate buildability
    linkage, is_buildable, error = linkage_service.validate_and_build(linkage_dict)

    # Store the linkage
    linkage_id = storage.create(linkage_dict)
    stored = storage.get(linkage_id)

    if stored is None:
        raise HTTPException(status_code=500, detail="Failed to create linkage")

    response_data = linkage_service.linkage_to_response_dict(
        linkage_id, stored, linkage, is_buildable, error
    )
    return LinkageResponse(**response_data)


@router.get("/", response_model=list[LinkageListItem])
def list_linkages(skip: int = 0, limit: int = 100) -> list[LinkageListItem]:
    """List all linkages."""
    items = storage.list_all(skip=skip, limit=limit)
    return [
        LinkageListItem(
            id=item["id"],
            name=item.get("name", "Unnamed"),
            joint_count=len(item.get("joints", [])),
            created_at=item["created_at"],
            is_buildable=item.get("is_buildable", True),
        )
        for item in items
    ]


@router.get("/{linkage_id}", response_model=LinkageResponse)
def get_linkage(linkage_id: str) -> LinkageResponse:
    """Get a linkage by ID."""
    stored = storage.get(linkage_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Linkage not found")

    # Try to build to get current state
    linkage_dict = {
        "name": stored.get("name"),
        "joints": stored.get("joints", []),
        "solve_order": stored.get("solve_order"),
    }
    linkage, is_buildable, error = linkage_service.validate_and_build(linkage_dict)

    response_data = linkage_service.linkage_to_response_dict(
        linkage_id, stored, linkage, is_buildable, error
    )
    return LinkageResponse(**response_data)


@router.put("/{linkage_id}", response_model=LinkageResponse)
def update_linkage(linkage_id: str, data: LinkageUpdate) -> LinkageResponse:
    """Update an existing linkage."""
    existing = storage.get(linkage_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Linkage not found")

    # Merge update data
    update_data = data.model_dump(exclude_none=True)

    # If joints changed, update and revalidate
    if update_data:
        storage.update(linkage_id, update_data)

    stored = storage.get(linkage_id)
    if stored is None:
        raise HTTPException(status_code=500, detail="Failed to update linkage")

    # Build the updated linkage
    linkage_dict = {
        "name": stored.get("name"),
        "joints": stored.get("joints", []),
        "solve_order": stored.get("solve_order"),
    }
    linkage, is_buildable, error = linkage_service.validate_and_build(linkage_dict)

    response_data = linkage_service.linkage_to_response_dict(
        linkage_id, stored, linkage, is_buildable, error
    )
    return LinkageResponse(**response_data)


@router.delete("/{linkage_id}", status_code=204)
def delete_linkage(linkage_id: str) -> None:
    """Delete a linkage."""
    if not storage.delete(linkage_id):
        raise HTTPException(status_code=404, detail="Linkage not found")
