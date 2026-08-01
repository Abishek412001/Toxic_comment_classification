"""
Standardized API Response Envelopes for OpenTrust AI Platform.
"""

from typing import Generic, TypeVar, Optional, List, Any, Dict
from datetime import datetime
from pydantic import Field
from opentrust_core.schemas.base import BaseSchema

T = TypeVar("T")


class APIResponse(BaseSchema, Generic[T]):
    """Standardized API Response envelope returned by all OpenTrust services."""

    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[T] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class PaginatedResponse(APIResponse[List[T]], Generic[T]):
    """Paginated list response envelope."""

    total_items: int = 0
    page: int = 1
    page_size: int = 20
    total_pages: int = 0


class ErrorDetail(BaseSchema):
    """Detailed error object."""

    code: str = "INTERNAL_ERROR"
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseSchema):
    """Standardized Error response envelope."""

    success: bool = False
    message: str = "An error occurred"
    error: ErrorDetail
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
