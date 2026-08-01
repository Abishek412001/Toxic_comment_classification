"""
Enterprise LLM Guardrail Orchestration Engine (Prompt Guard, Response Guard & Masking).
"""

import time
from services.guardrail_service.schemas import (
    PromptGuardrailRequest,
    PromptGuardrailResponse,
    ResponseGuardrailRequest,
    ResponseGuardrailResponse,
    PIIMaskingRequest,
    PIIMaskingResponse,
)
from services.guardrail_service.pii_detector import pii_detector
from services.guardrail_service.injection_detector import injection_detector

HALLUCINATION_KEYWORDS = ["as an ai i cannot verify", "unconfirmed source", "hallucinated detail"]


class GuardrailEngine:
    """Enterprise LLM Guardrail & Safety Engine."""

    def inspect_prompt(self, request: PromptGuardrailRequest) -> PromptGuardrailResponse:
        """Inspects prompt for prompt injection, jailbreak attacks, and PII."""
        start_time = time.perf_counter()

        has_inj, inj_type, has_jb, jb_type = False, None, False, None
        if request.detect_injection or request.detect_jailbreak:
            has_inj, inj_type, has_jb, jb_type = injection_detector.detect_injection_or_jailbreak(request.prompt)

        sanitized_prompt, pii_entities = request.prompt, []
        if request.detect_pii:
            sanitized_prompt, pii_entities = pii_detector.mask_pii(request.prompt)

        contains_pii = len(pii_entities) > 0
        is_safe = not (has_inj or has_jb)

        if has_inj or has_jb:
            risk_score = 0.95
            recommended_action = "BLOCK"
        elif contains_pii:
            risk_score = 0.45
            recommended_action = "MASK"
        else:
            risk_score = 0.05
            recommended_action = "ALLOW"

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return PromptGuardrailResponse(
            prompt=request.prompt,
            is_safe=is_safe,
            risk_score=risk_score,
            contains_injection=has_inj,
            injection_type=inj_type,
            contains_jailbreak=has_jb,
            jailbreak_type=jb_type,
            contains_pii=contains_pii,
            pii_entities=pii_entities,
            sanitized_prompt=sanitized_prompt,
            recommended_action=recommended_action,
            latency_ms=round(latency_ms, 3),
        )

    def mask_pii_only(self, request: PIIMaskingRequest) -> PIIMaskingResponse:
        """Masks PII entities in a raw text string."""
        masked_text, entities = pii_detector.mask_pii(request.text, mask_label=request.mask_char)
        return PIIMaskingResponse(
            original_text=request.text,
            masked_text=masked_text,
            pii_count=len(entities),
            entities_found=entities,
        )

    def inspect_response(self, request: ResponseGuardrailRequest) -> ResponseGuardrailResponse:
        """Inspects generated LLM response for PII leakage, hallucination indicators, and safety."""
        start_time = time.perf_counter()
        resp_lower = request.response_text.lower()

        sanitized_response, pii_entities = pii_detector.mask_pii(request.response_text)
        contains_pii_leakage = len(pii_entities) > 0

        contains_hallucination = any(kw in resp_lower for kw in HALLUCINATION_KEYWORDS)

        flagged_reasons = []
        if contains_pii_leakage:
            flagged_reasons.append("Detected unredacted PII in LLM response.")
        if contains_hallucination:
            flagged_reasons.append("Detected potential hallucination indicator.")

        is_safe = len(flagged_reasons) == 0
        risk_score = 0.85 if contains_pii_leakage else (0.40 if contains_hallucination else 0.05)
        action = "REWRITE" if contains_pii_leakage else ("WARN" if contains_hallucination else "PASS")

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return ResponseGuardrailResponse(
            response_text=request.response_text,
            is_safe=is_safe,
            risk_score=risk_score,
            contains_hallucination_warning=contains_hallucination,
            contains_pii_leakage=contains_pii_leakage,
            flagged_reasons=flagged_reasons,
            sanitized_response=sanitized_response,
            action=action,
            latency_ms=round(latency_ms, 3),
        )


guardrail_engine = GuardrailEngine()
