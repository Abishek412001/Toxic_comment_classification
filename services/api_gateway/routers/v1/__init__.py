"""
API Gateway Router V1 Package.
"""

from fastapi import APIRouter
from services.api_gateway.routers.v1.auth import router as auth_router
from services.api_gateway.routers.v1.organizations import router as org_router
from services.api_gateway.routers.v1.api_keys import router as api_key_router
from services.api_gateway.routers.v1.moderation import router as moderation_router
from services.api_gateway.routers.v1.sentiment import router as sentiment_router
from services.api_gateway.routers.v1.emotion import router as emotion_router
from services.api_gateway.routers.v1.xai import router as xai_router
from services.api_gateway.routers.v1.decision import router as decision_router
from services.api_gateway.routers.v1.guardrails import router as guardrails_router
from services.api_gateway.routers.v1.mlops import router as mlops_router
from services.api_gateway.routers.v1.saas import router as saas_router
from services.api_gateway.routers.v1.integrations import router as integrations_router
from services.api_gateway.routers.v1.studio import router as studio_router

router_v1 = APIRouter()

router_v1.include_router(auth_router)
router_v1.include_router(org_router)
router_v1.include_router(api_key_router)
router_v1.include_router(moderation_router)
router_v1.include_router(sentiment_router)
router_v1.include_router(emotion_router)
router_v1.include_router(xai_router)
router_v1.include_router(decision_router)
router_v1.include_router(guardrails_router)
router_v1.include_router(mlops_router)
router_v1.include_router(saas_router)
router_v1.include_router(integrations_router)
router_v1.include_router(studio_router)


@router_v1.get("/", summary="API V1 Index")
async def v1_index():
    return {
        "status": "online",
        "service": "OpenTrust AI Platform Gateway",
        "version": "1.0.0",
        "endpoints": [
            "/api/v1/auth/signup",
            "/api/v1/auth/login",
            "/api/v1/auth/me",
            "/api/v1/organizations",
            "/api/v1/api-keys",
            "/api/v1/moderation/predict",
            "/api/v1/moderation/batch",
            "/api/v1/moderation/policies",
            "/api/v1/sentiment/analyze",
            "/api/v1/sentiment/batch",
            "/api/v1/sentiment/engines",
            "/api/v1/emotion/detect",
            "/api/v1/emotion/batch",
            "/api/v1/emotion/lexicon",
            "/api/v1/xai/explain",
            "/api/v1/xai/batch",
            "/api/v1/decision/evaluate",
            "/api/v1/decision/audit-trail",
            "/api/v1/guardrails/prompt/inspect",
            "/api/v1/guardrails/prompt/mask-pii",
            "/api/v1/guardrails/response/inspect",
            "/api/v1/mlops/models/register",
            "/api/v1/mlops/models/promote",
            "/api/v1/mlops/models/rollback",
            "/api/v1/mlops/drift/detect",
            "/api/v1/mlops/retrain/trigger",
            "/api/v1/mlops/observability/metrics",
            "/api/v1/saas/subscriptions/subscribe",
            "/api/v1/saas/subscriptions/current",
            "/api/v1/saas/billing/usage",
            "/api/v1/saas/billing/invoices",
            "/api/v1/saas/notifications/webhook",
            "/api/v1/integrations/connectors",
            "/api/v1/integrations/connectors/dispatch",
            "/api/v1/integrations/marketplace/items",
            "/api/v1/integrations/marketplace/install",
            "/api/v1/studio/prompts/save",
            "/api/v1/studio/prompts/evaluate",
            "/api/v1/studio/workflows/execute",
            "/api/v1/studio/agents/run",
            "/api/v1/studio/rag/query",
        ],
    }
