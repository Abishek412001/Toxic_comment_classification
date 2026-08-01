"""
Organization Management Endpoints.
"""

import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, Depends
from opentrust_core.auth.models import OrganizationCreate, OrganizationRead, UserRead, RoleEnum, PlanTierEnum
from opentrust_core.auth.dependencies import get_current_user, require_role
from opentrust_core.schemas.response import APIResponse

router = APIRouter(prefix="/organizations", tags=["Organizations & Tenants"])

ORGANIZATION_DATABASE: Dict[str, Dict[str, Any]] = {
    "org_opentrust_root": {
        "id": "org_opentrust_root",
        "name": "OpenTrust AI Enterprise",
        "slug": "opentrust-root",
        "owner_id": "usr_admin_001",
        "plan_tier": "enterprise",
    }
}


@router.post("", response_model=APIResponse[OrganizationRead])
async def create_organization(
    org_in: OrganizationCreate,
    current_user: UserRead = Depends(require_role([RoleEnum.ADMIN, RoleEnum.DEVELOPER])),
):
    """Creates a new organization tenant."""
    org_id = f"org_{uuid.uuid4().hex[:8]}"
    org_record = {
        "id": org_id,
        "name": org_in.name,
        "slug": org_in.slug,
        "owner_id": current_user.id,
        "plan_tier": org_in.plan_tier.value,
    }
    ORGANIZATION_DATABASE[org_id] = org_record

    org_read = OrganizationRead(
        id=org_id,
        name=org_in.name,
        slug=org_in.slug,
        owner_id=current_user.id,
        plan_tier=org_in.plan_tier,
    )
    return APIResponse[OrganizationRead](data=org_read, message="Organization created successfully.")


@router.get("", response_model=APIResponse[List[OrganizationRead]])
async def list_organizations(current_user: UserRead = Depends(get_current_user)):
    """Lists organizations accessible to the current user."""
    orgs = [
        OrganizationRead(
            id=data["id"],
            name=data["name"],
            slug=data["slug"],
            owner_id=data["owner_id"],
            plan_tier=PlanTierEnum(data["plan_tier"]),
        )
        for data in ORGANIZATION_DATABASE.values()
    ]
    return APIResponse[List[OrganizationRead]](data=orgs)
