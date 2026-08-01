"""
Pydantic v2 Schemas for Prompt & Response Guardrails, PII Redaction, and Safety Inspections.
"""

from typing import List, Dict, Optional, Any
from pydantic import Field
from opentrust_core.schemas.base import BaseSchema


class PIIEntity(BaseSchema):
    entity_type: str  # EMAIL, PHONE, CREDIT_CARD, IP_ADDRESS, SSN_PAN
    value: str
    start_char: int
    end_char: int


class PromptGuardrailRequest(BaseSchema):
    prompt: str = Field(min_length=1, max_length=20000, description="Input prompt text to evaluate")
    detect_pii: bool = True
    detect_injection: bool = True
    detect_jailbreak: bool = True


class PromptGuardrailResponse(BaseSchema):
    prompt: str
    is_safe: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    contains_injection: bool
    injection_type: Optional[str] = None
    contains_jailbreak: bool
    jailbreak_type: Optional[str] = None
    contains_pii: bool
    pii_entities: List[PIIEntity]
    sanitized_prompt: str
    recommended_action: str  # ALLOW, WARN, MASK, BLOCK
    latency_ms: float


class PIIMaskingRequest(BaseSchema):
    text: str = Field(min_length=1, max_length=20000)
    mask_char: str = "[REDACTED]"


class PIIMaskingResponse(BaseSchema):
    original_text: str
    masked_text: str
    pii_count: int
    entities_found: List[PIIEntity]


class ResponseGuardrailRequest(BaseSchema):
    response_text: str = Field(min_length=1, max_length=50000)
    original_prompt: Optional[str] = None


class ResponseGuardrailResponse(BaseSchema):
    response_text: str
    is_safe: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    contains_hallucination_warning: bool
    contains_pii_leakage: bool
    flagged_reasons: List[str]
    sanitized_response: str
    action: str  # PASS, REWRITE, BLOCK
    latency_ms: float
