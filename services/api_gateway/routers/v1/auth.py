"""
Authentication Endpoints: Signup, Login, Profile, Token Refresh.
"""

import uuid
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from opentrust_core.auth.passwords import hash_password, verify_password
from opentrust_core.auth.jwt import create_access_token
from opentrust_core.auth.models import UserCreate, UserRead, RoleEnum
from opentrust_core.auth.dependencies import get_current_user
from opentrust_core.schemas.response import APIResponse
from opentrust_core.exceptions import AuthenticationError

router = APIRouter(prefix="/auth", tags=["Authentication & Access"])

# In-memory User Store for Demonstration & Verification
USER_DATABASE: Dict[str, Dict[str, Any]] = {
    "admin@opentrust.ai": {
        "id": "usr_admin_001",
        "email": "admin@opentrust.ai",
        "full_name": "OpenTrust System Admin",
        "password_hash": hash_password("AdminSecure2026!"),
        "role": "admin",
        "org_id": "org_opentrust_root",
        "ws_id": "ws_default",
    }
}


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400
    user: UserRead


@router.post("/signup", response_model=APIResponse[UserRead], status_code=status.HTTP_201_CREATED)
async def signup(user_in: UserCreate):
    """Registers a new user account."""
    if user_in.email in USER_DATABASE:
        raise HTTPException(status_code=400, detail="User with this email already exists.")

    user_id = f"usr_{uuid.uuid4().hex[:8]}"
    pwd_hash = hash_password(user_in.password)

    user_record = {
        "id": user_id,
        "email": user_in.email,
        "full_name": user_in.full_name,
        "password_hash": pwd_hash,
        "role": user_in.role.value,
        "org_id": f"org_{uuid.uuid4().hex[:8]}",
        "ws_id": f"ws_{uuid.uuid4().hex[:8]}",
    }
    USER_DATABASE[user_in.email] = user_record

    user_read = UserRead(
        id=user_id,
        email=user_in.email,
        full_name=user_in.full_name,
        role=user_in.role,
        organization_id=user_record["org_id"],
        workspace_id=user_record["ws_id"],
    )
    return APIResponse[UserRead](data=user_read, message="User account created successfully.")


@router.post("/login", response_model=APIResponse[TokenResponse])
async def login(credentials: LoginRequest):
    """Authenticates user and returns JWT access token."""
    user = USER_DATABASE.get(credentials.email)
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise AuthenticationError("Invalid email or password.")

    token_payload = {
        "sub": user["id"],
        "email": user["email"],
        "name": user["full_name"],
        "role": user["role"],
        "org_id": user["org_id"],
        "ws_id": user["ws_id"],
    }
    access_token = create_access_token(token_payload)

    user_read = UserRead(
        id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        role=RoleEnum(user["role"]),
        organization_id=user["org_id"],
        workspace_id=user["ws_id"],
    )

    return APIResponse[TokenResponse](
        data=TokenResponse(access_token=access_token, user=user_read),
        message="Authentication successful.",
    )


@router.get("/me", response_model=APIResponse[UserRead])
async def get_me(current_user: UserRead = Depends(get_current_user)):
    """Returns currently authenticated user profile."""
    return APIResponse[UserRead](data=current_user)
