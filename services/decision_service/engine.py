"""
Enterprise Decision Intelligence Engine evaluating risk thresholds, HITL triggers, and Audit Trails.
"""

import uuid
from typing import List, Dict, Any
from services.decision_service.schemas import (
    DecisionEvaluationRequest,
    DecisionEvaluationResponse,
    AuditLogEntry,
)

# Immutable in-memory Audit Log Store
AUDIT_TRAIL_STORE: List[AuditLogEntry] = []


class DecisionEngine:
    """Enterprise Decision Intelligence & Audit Trail Engine."""

    def evaluate_decision(self, request: DecisionEvaluationRequest) -> DecisionEvaluationResponse:
        """Evaluates automated AI decisions against confidence thresholds and logs audit trail."""
        decision_id = f"dec_{uuid.uuid4().hex[:10]}"

        # Risk level categorization
        if request.risk_score >= 0.85:
            risk_level = "CRITICAL"
        elif request.risk_score >= 0.50:
            risk_level = "HIGH"
        elif request.risk_score >= 0.25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # HITL Review Trigger policy: Trigger human review if low confidence (<0.70) or high risk
        trigger_hitl = False
        hitl_reason = None

        if request.confidence_score < 0.70:
            trigger_hitl = True
            hitl_reason = f"Low model prediction confidence ({round(request.confidence_score, 2)} < 0.70 threshold)."
        elif request.moderation_action == "FLAG" or (risk_level in ["HIGH", "CRITICAL"] and request.moderation_action == "PASS"):
            trigger_hitl = True
            hitl_reason = f"Flagged high-risk prediction ({risk_level}) requires manual human review."

        final_action = request.moderation_action
        recommendation = (
            f"Action '{final_action}' confirmed with {risk_level} risk."
            if not trigger_hitl
            else f"Routed to HITL review queue due to: {hitl_reason}"
        )

        # Record to immutable audit log
        audit_entry = AuditLogEntry(
            decision_id=decision_id,
            text_snippet=request.text[:50] + ("..." if len(request.text) > 50 else ""),
            final_action=final_action,
            risk_score=request.risk_score,
            confidence_score=request.confidence_score,
            trigger_hitl_review=trigger_hitl,
        )
        AUDIT_TRAIL_STORE.append(audit_entry)

        return DecisionEvaluationResponse(
            decision_id=decision_id,
            final_action=final_action,
            trigger_hitl_review=trigger_hitl,
            hitl_reason=hitl_reason,
            risk_level=risk_level,
            recommendation=recommendation,
            audit_logged=True,
        )

    def get_audit_trail(self, limit: int = 50) -> List[AuditLogEntry]:
        """Returns recent decision audit trail entries."""
        return list(reversed(AUDIT_TRAIL_STORE))[:limit]


decision_engine = DecisionEngine()
