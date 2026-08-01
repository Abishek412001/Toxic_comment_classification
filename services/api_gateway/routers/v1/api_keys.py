"""
API Key Management Endpoints.
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
from fastapi import APIRouter, Depends
from opentrust_core.auth.models import ApiKeyCreate, ApiKeyRead, ApiKeyCreatedResponse, UserRead, RoleEnum
from opentrust_core.auth.dependencies import get_current_user, require_role
from opentrust_core.security import generate_api_key, hash_secret
from opentrust_core.schemas.response import APIResponse

router = APIRouter(prefix="/api-keys", tags=["API Key Management"])

API_KEY_DATABASE: Dict[str, Dict[str, Any]] = {}


@router.post("", response_model=APIResponse[ApiKeyCreatedResponse])
async def create_api_key(
    key_in: ApiKeyCreate,
    current_user: UserRead = Depends(require_role([RoleEnum.ADMIN, RoleEnum.DEVELOPER])),
):
    """Generates a new API Key for tenant microservice integration."""
    raw_key = generate_api_key()
    key_hash = hash_secret(raw_key)
    key_id = f"key_{uuid.uuid4().hex[:8]}"

    expires_at = datetime.utcnow() + timedelta(days=key_in.expires_in_days) if key_in.expires_in_days else None

    key_record = {
        "id": key_id,
        "name": key_in.name,
        "prefix": raw_key[:12],
        "key_hash": key_hash,
        "organization_id": key_in.organization_id,
        "is_active": True,
        "expires_at": expires_at,
    }
    API_KEY_DATABASE[key_id] = key_record

    res = ApiKeyCreatedResponse(
        id=key_id,
        name=key_in.name,
        prefix=raw_key[:12],
        organization_id=key_in.organization_id,
        is_active=True,
        expires_at=expires_at,
        raw_api_key=raw_key,
    )
    return APIResponse[ApiKeyCreatedResponse](
        data=res,
        message="API Key created successfully. Store the raw key securely; it will not be displayed again.",
    )


@router.get("", response_model=APIResponse[List[ApiKeyRead]])
async def list_api_keys(current_user: UserRead = Depends(get_current_user)):
    """Lists registered API keys for organization."""
    keys = [
        ApiKeyRead(
            id=d["id"],
            name=d["name"],
            prefix=d["prefix"],
            organization_id=d["organization_id"],
            is_active=d["is_active"],
            expires_at=d["expires_at"],
        )
        for d in API_KEY_DATABASE.values()
    ]
    return APIResponse[List[ApiKeyRead]](data=keys)
