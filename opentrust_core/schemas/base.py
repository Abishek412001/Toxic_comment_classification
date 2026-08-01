"""
Base Schema Model using Pydantic v2.
"""

from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """Abstract base Pydantic v2 schema for all OpenTrust requests and responses."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )
