"""
Pydantic v2 Models for Multi-Tenant Auth, Organizations, Workspaces, and API Keys.
"""

from enum import Enum
from typing import Optional, List
from datetime import datetime
from pydantic import Field, EmailStr
from opentrust_core.schemas.base import BaseSchema


class RoleEnum(str, Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    VIEWER = "viewer"


class PlanTierEnum(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


# User Schemas
class UserBase(BaseSchema):
    email: EmailStr
    full_name: str
    is_active: bool = True
    role: RoleEnum = RoleEnum.DEVELOPER


class UserCreate(UserBase):
    password: str = Field(min_length=8)


class UserRead(UserBase):
    id: str
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Organization Schemas
class OrganizationBase(BaseSchema):
    name: str
    slug: str
    plan_tier: PlanTierEnum = PlanTierEnum.FREE


class OrganizationCreate(OrganizationBase):
    pass


class OrganizationRead(OrganizationBase):
    id: str
    owner_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Workspace Schemas
class WorkspaceBase(BaseSchema):
    name: str
    slug: str


class WorkspaceCreate(WorkspaceBase):
    organization_id: str


class WorkspaceRead(WorkspaceBase):
    id: str
    organization_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


# API Key Schemas
class ApiKeyCreate(BaseSchema):
    name: str
    organization_id: str
    expires_in_days: Optional[int] = 30


class ApiKeyRead(BaseSchema):
    id: str
    name: str
    prefix: str
    organization_id: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class ApiKeyCreatedResponse(ApiKeyRead):
    raw_api_key: str  # Displayed only once upon creation
