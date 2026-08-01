"""
Enterprise Multi-Class Emotion Detection Engine based on NRC Lexicon & Semantic Distributions.
"""

import time
import re
from typing import List, Dict
from services.emotion_service.schemas import (
    EmotionRequest,
    EmotionResponse,
    BatchEmotionRequest,
    BatchEmotionResponse,
    EmotionDistribution,
    EmotionItem,
)

# Semantic Emotion Keyword Patterns
EMOTION_PATTERNS: Dict[str, List[str]] = {
    "anger": [r"\bkill\b", r"\bhate\b", r"\bmad\b", r"\banxious\b", r"\brage\b", r"\bfurious\b"],
    "anticipation": [r"\bexpect\b", r"\bsoon\b", r"\banticipate\b", r"\bahead\b"],
    "disgust": [r"\bgross\b", r"\bdisgusting\b", r"\bhorrible\b", r"\bsick\b", r"\bawful\b"],
    "fear": [r"\bafraid\b", r"\bscared\b", r"\bterror\b", r"\bdanger\b", r"\bthreat\b"],
    "joy": [r"\bhappy\b", r"\bgreat\b", r"\blove\b", r"\bawesome\b", r"\bexcellent\b", r"\bwin\b", r"\bjoy\b"],
    "sadness": [r"\bsad\b", r"\bcry\b", r"\bdepressed\b", r"\bloss\b", r"\bsorry\b", r"\bhurt\b"],
    "surprise": [r"\bwow\b", r"\bamazing\b", r"\bunexpected\b", r"\bshock\b", r"\bsurprise\b"],
    "trust": [r"\btrust\b", r"\bhonest\b", r"\bbelieve\b", r"\bsecure\b", r"\bsafe\b"],
}


class EmotionEngine:
    """Multi-class emotion intelligence engine."""

    def detect_single(self, request: EmotionRequest) -> EmotionResponse:
        """Detects 8-class emotion probability distribution for a text input."""
        start_time = time.perf_counter()
        text_lower = request.text.lower()

        scores: Dict[str, float] = {}
        for emotion, patterns in EMOTION_PATTERNS.items():
            matches = sum(1 for p in patterns if re.search(p, text_lower))
            scores[emotion] = 0.05 + min(matches * 0.40, 0.90)

        # Normalize to probability distribution sum = 1.0
        total_score = sum(scores.values())
        prob_dist = {e: round(s / total_score, 4) for e, s in scores.items()}

        sorted_emotions = sorted(prob_dist.items(), key=lambda x: x[1], reverse=True)
        dominant_emotion, confidence = sorted_emotions[0]

        top_emotions = [
            EmotionItem(emotion=e, probability=p)
            for e, p in sorted_emotions[: request.top_n]
        ]

        distribution = EmotionDistribution(**prob_dist)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return EmotionResponse(
            text=request.text,
            dominant_emotion=dominant_emotion,
            confidence=confidence,
            top_emotions=top_emotions,
            distribution=distribution,
            latency_ms=round(latency_ms, 3),
        )

    def detect_batch(self, request: BatchEmotionRequest) -> BatchEmotionResponse:
        """Batch detects emotions across bulk texts."""
        batch_start = time.perf_counter()
        results: List[EmotionResponse] = []
        counts: Dict[str, int] = {e: 0 for e in EMOTION_PATTERNS.keys()}

        for text in request.texts:
            single_req = EmotionRequest(text=text, top_n=request.top_n)
            res = self.detect_single(single_req)
            results.append(res)
            counts[res.dominant_emotion] = counts.get(res.dominant_emotion, 0) + 1

        batch_latency = (time.perf_counter() - batch_start) * 1000.0

        return BatchEmotionResponse(
            total_processed=len(request.texts),
            emotion_counts=counts,
            results=results,
            batch_latency_ms=round(batch_latency, 3),
        )


emotion_engine = EmotionEngine()
