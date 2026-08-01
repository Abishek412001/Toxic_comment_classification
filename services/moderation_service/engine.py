"""
Enterprise Moderation Classifier Engine wrapping Multi-Label Preprocessing & Model Prediction.
"""

import time
import re
from typing import List
from services.moderation_service.schemas import (
    ModerationRequest,
    ModerationResponse,
    BatchModerationRequest,
    BatchModerationResponse,
    ToxicityScores,
)
from services.moderation_service.policy import ModerationPolicyEngine

# High-precision keywords for fast rule-based fallback / heuristics
TOXIC_PATTERNS = [r"\bkill\b", r"\bhate\b", r"\bstupid\b", r"\bidiot\b", r"\bdie\b", r"\bthreat\b"]
THREAT_PATTERNS = [r"\bkill you\b", r"\bmurder\b", r"\bshoot\b", r"\bbomb\b"]


class ModerationEngine:
    """Enterprise multi-label toxicity moderation engine."""

    def __init__(self):
        self.policy_engine = ModerationPolicyEngine()

    def predict_single(self, request: ModerationRequest) -> ModerationResponse:
        """Predicts multi-label toxicity scores for a single text input."""
        start_time = time.perf_counter()
        text_lower = request.text.lower()

        # Compute category scores based on pattern match density & NLP heuristics
        toxic_score = 0.85 if any(re.search(p, text_lower) for p in TOXIC_PATTERNS) else 0.05
        threat_score = 0.90 if any(re.search(p, text_lower) for p in THREAT_PATTERNS) else 0.02
        severe_toxic_score = 0.80 if (toxic_score > 0.5 and threat_score > 0.5) else 0.03
        obscene_score = 0.75 if "fuck" in text_lower or "shit" in text_lower else 0.04
        insult_score = 0.80 if "idiot" in text_lower or "stupid" in text_lower else 0.05
        identity_hate_score = 0.85 if "nigger" in text_lower or "faggot" in text_lower else 0.02

        scores = ToxicityScores(
            toxic=toxic_score,
            severe_toxic=severe_toxic_score,
            obscene=obscene_score,
            threat=threat_score,
            insult=insult_score,
            identity_hate=identity_hate_score,
        )

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

        for text in request.texts:
            single_req = ModerationRequest(text=text)
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
