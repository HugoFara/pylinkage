"""In-memory storage backend for linkages."""

import uuid
from datetime import datetime, timezone
from typing import Any


class MemoryStorage:
    """In-memory storage for linkages."""

    def __init__(self) -> None:
        self._linkages: dict[str, dict[str, Any]] = {}

    def create(self, linkage_data: dict[str, Any]) -> str:
        """Create a new linkage and return its ID."""
        linkage_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        self._linkages[linkage_id] = {
            "id": linkage_id,
            "created_at": now,
            "updated_at": now,
            **linkage_data,
        }
        return linkage_id

    def get(self, linkage_id: str) -> dict[str, Any] | None:
        """Get a linkage by ID."""
        return self._linkages.get(linkage_id)

    def update(self, linkage_id: str, data: dict[str, Any]) -> bool:
        """Update a linkage. Returns True if successful."""
        if linkage_id not in self._linkages:
            return False
        self._linkages[linkage_id].update(data)
        self._linkages[linkage_id]["updated_at"] = datetime.now(timezone.utc)
        return True

    def delete(self, linkage_id: str) -> bool:
        """Delete a linkage. Returns True if successful."""
        if linkage_id not in self._linkages:
            return False
        del self._linkages[linkage_id]
        return True

    def list_all(self, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        """List all linkages with pagination."""
        items = list(self._linkages.values())
        # Sort by created_at descending (newest first)
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return items[skip : skip + limit]

    def clear(self) -> None:
        """Clear all linkages."""
        self._linkages.clear()


# Global storage instance
storage = MemoryStorage()
