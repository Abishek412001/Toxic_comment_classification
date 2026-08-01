"""
LLM Guardrails & AI Safety API Gateway Router.
"""

from fastapi import APIRouter, Depends
from services.guardrail_service.schemas import (
    PromptGuardrailRequest,
    PromptGuardrailResponse,
    ResponseGuardrailRequest,
    ResponseGuardrailResponse,
    PIIMaskingRequest,
    PIIMaskingResponse,
)
from services.guardrail_service.engine import guardrail_engine
from opentrust_core.auth.dependencies import check_api_key_rate_limit
from opentrust_core.schemas.response import APIResponse

router = APIRouter(prefix="/guardrails", tags=["LLM Guardrails & AI Safety"])


@router.post("/prompt/inspect", response_model=APIResponse[PromptGuardrailResponse])
async def inspect_prompt(
    request: PromptGuardrailRequest,
    rate_info: dict = Depends(check_api_key_rate_limit),
):
    """Inspects input prompt for injections, jailbreak attacks, and PII."""
    res = guardrail_engine.inspect_prompt(request)
    return APIResponse[PromptGuardrailResponse](
        data=res,
        message="Prompt guardrail inspection completed.",
    )


@router.post("/prompt/mask-pii", response_model=APIResponse[PIIMaskingResponse])
async def mask_prompt_pii(
    request: PIIMaskingRequest,
    rate_info: dict = Depends(check_api_key_rate_limit),
):
    """Detects and redacts PII entities in text."""
    res = guardrail_engine.mask_pii_only(request)
    return APIResponse[PIIMaskingResponse](
        data=res,
        message=f"PII masking completed ({res.pii_count} entities redacted).",
    )


@router.post("/response/inspect", response_model=APIResponse[ResponseGuardrailResponse])
async def inspect_response(
    request: ResponseGuardrailRequest,
    rate_info: dict = Depends(check_api_key_rate_limit),
):
    """Inspects generated LLM response for PII leakage and hallucinations."""
    res = guardrail_engine.inspect_response(request)
    return APIResponse[ResponseGuardrailResponse](
        data=res,
        message="Response guardrail inspection completed.",
    )
