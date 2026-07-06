"""Base domain models."""

from pydantic import BaseModel
from typing import Any


class DomainModel(BaseModel):
    """Base class for all domain models."""

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return self.model_dump()
