"""
OpenTrust AI API Gateway Main FastAPI Application.
"""

import os
import sys

# Ensure root project directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from opentrust_core.config import settings
from opentrust_core.exceptions import OpenTrustException
from opentrust_core.logging import get_logger
from opentrust_core.schemas.response import ErrorResponse, ErrorDetail
from services.api_gateway.routers.health import router as health_router
from services.api_gateway.routers.v1 import router_v1

logger = get_logger("api_gateway")

app = FastAPI(
    title=f"{settings.PROJECT_NAME} - Enterprise API Gateway",
    description="Production-grade API Gateway for Content Moderation, LLM Safety, Sentiment & Emotion Mining, and Explainable AI.",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(OpenTrustException)
async def opentrust_exception_handler(request: Request, exc: OpenTrustException):
    """Global Exception Handler for OpenTrust Platform Errors."""
    logger.error(f"OpenTrust Exception on {request.url.path}: {exc.message} (Code: {exc.code})")
    error_res = ErrorResponse(
        success=False,
        message=exc.message,
        error=ErrorDetail(
            code=exc.code,
            message=exc.message,
            details=exc.details,
        ),
    )
    return JSONResponse(status_code=exc.status_code, content=error_res.model_dump(mode="json"))


# Include Routers
app.include_router(health_router)
app.include_router(router_v1, prefix=settings.API_PREFIX)


@app.get("/", summary="Root Endpoint Redirect")
async def root():
    return {
        "platform": settings.PROJECT_NAME,
        "status": "online",
        "docs": "/docs",
        "health": "/health/liveness",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
