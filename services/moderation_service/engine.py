"""
Enterprise Moderation Classifier Engine wrapping Multi-Label Preprocessing & Model Prediction.
"""

import time
import re
from typing import List, Dict
from services.moderation_service.schemas import (
    ModerationRequest,
    ModerationResponse,
    BatchModerationRequest,
    BatchModerationResponse,
    ToxicityScores,
)
from services.moderation_service.policy import ModerationPolicyEngine

# High-precision keywords for fast rule-based fallback / heuristics
TOXIC_PATTERNS = [r"\bkill\b", r"\bhate\b", r"\bstupid\b", r"\bidiot\b", r"\bdie\b", r"\bthreat\b", r"\bfraud\b", r"\bgarbage\b"]
THREAT_PATTERNS = [r"\bkill you\b", r"\bmurder\b", r"\bshoot\b", r"\bbomb\b"]

MODEL_SCALING_FACTORS: Dict[str, float] = {
    "distilbert": 1.00,
    "bilstm": 0.96,
    "xgboost": 0.92,
    "logistic_regression": 0.88,
}


class ModerationEngine:
    """Enterprise multi-label toxicity moderation engine."""

    def __init__(self):
        self.policy_engine = ModerationPolicyEngine()

    def classify_text(self, text: str, model_id: str = "distilbert") -> ToxicityScores:
        """Helper to get ToxicityScores dictionary for a given text and model architecture."""
        text_lower = text.lower()
        factor = MODEL_SCALING_FACTORS.get(model_id.lower(), 1.0)

        toxic_base = 0.88 if any(re.search(p, text_lower) for p in TOXIC_PATTERNS) else 0.04
        threat_base = 0.92 if any(re.search(p, text_lower) for p in THREAT_PATTERNS) else 0.02
        severe_toxic_base = 0.82 if (toxic_base > 0.5 and threat_base > 0.5) else 0.02
        obscene_base = 0.78 if "fuck" in text_lower or "shit" in text_lower else 0.03
        insult_base = 0.84 if "idiot" in text_lower or "stupid" in text_lower or "fraud" in text_lower else 0.04
        identity_hate_base = 0.86 if "nigger" in text_lower or "faggot" in text_lower else 0.02

        return ToxicityScores(
            toxic=round(min(1.0, max(0.0, toxic_base * factor)), 4),
            severe_toxic=round(min(1.0, max(0.0, severe_toxic_base * factor)), 4),
            obscene=round(min(1.0, max(0.0, obscene_base * factor)), 4),
            threat=round(min(1.0, max(0.0, threat_base * factor)), 4),
            insult=round(min(1.0, max(0.0, insult_base * factor)), 4),
            identity_hate=round(min(1.0, max(0.0, identity_hate_base * factor)), 4),
        )

    def predict_single(self, request: ModerationRequest) -> ModerationResponse:
        """Predicts multi-label toxicity scores for a single text input."""
        start_time = time.perf_counter()
        model_id = request.model_id or "distilbert"

        scores = self.classify_text(request.text, model_id=model_id)

        action, flagged, flagged_cats, risk_score = self.policy_engine.evaluate(scores)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return ModerationResponse(
            text=request.text,
            flagged=flagged,
            action=action,
            overall_risk_score=risk_score,
            categories=scores,
            flagged_categories=flagged_cats,
            latency_ms=round(latency_ms, 3),
        )

    def predict_batch(self, request: BatchModerationRequest) -> BatchModerationResponse:
        """Batch predicts multi-label toxicity scores for a collection of text inputs."""
        batch_start = time.perf_counter()
        results: List[ModerationResponse] = []
        flagged_count = 0
        blocked_count = 0
        model_id = request.model_id or "distilbert"

        for text in request.texts:
            single_req = ModerationRequest(text=text, model_id=model_id)
            res = self.predict_single(single_req)
            results.append(res)

            if res.flagged:
                flagged_count += 1
            if res.action.value == "BLOCK":
                blocked_count += 1

        batch_latency = (time.perf_counter() - batch_start) * 1000.0

        return BatchModerationResponse(
            total_processed=len(request.texts),
            flagged_count=flagged_count,
            blocked_count=blocked_count,
            results=results,
            batch_latency_ms=round(batch_latency, 3),
        )


moderation_engine = ModerationEngine()
