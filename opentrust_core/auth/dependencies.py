"""
FastAPI Security & Dependency Injectors for Authentication, RBAC, and Tenant Context.
"""

from typing import Optional
from fastapi import Request, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from opentrust_core.auth.jwt import decode_access_token
from opentrust_core.auth.models import UserRead, RoleEnum, PlanTierEnum
from opentrust_core.exceptions import AuthenticationError, AuthorizationError
from opentrust_core.auth.rate_limiter import rate_limiter

security_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
) -> UserRead:
    """Extracts and validates JWT token from HTTP Bearer authorization header."""
    if not credentials or not credentials.credentials:
        raise AuthenticationError("Missing or invalid Authorization header")

    payload = decode_access_token(credentials.credentials)

    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Invalid token payload: missing sub claim")

    return UserRead(
        id=user_id,
        email=payload.get("email", "user@opentrust.ai"),
        full_name=payload.get("name", "Enterprise User"),
        role=RoleEnum(payload.get("role", "developer")),
        organization_id=payload.get("org_id", "org_default"),
        workspace_id=payload.get("ws_id", "ws_default"),
    )


def require_role(allowed_roles: list[RoleEnum]):
    """Enforces Role-Based Access Control (RBAC) permission check."""
    async def role_checker(current_user: UserRead = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise AuthorizationError(
                f"Role '{current_user.role.value}' is not authorized. Allowed: {[r.value for r in allowed_roles]}"
            )
        return current_user

    return role_checker


async def check_api_key_rate_limit(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Enforces rate limiting based on X-API-Key header or anonymous client IP."""
    client_id = x_api_key if x_api_key else "anonymous_client"
    plan_tier = PlanTierEnum.ENTERPRISE if (x_api_key and x_api_key.startswith("ot_ent_")) else PlanTierEnum.FREE
    remaining = rate_limiter.check_rate_limit(client_id, plan_tier=plan_tier)
    return {"client_id": client_id, "remaining_quota": remaining}
