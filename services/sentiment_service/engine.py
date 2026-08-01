"""
Enterprise Sentiment Classifier Engine integrating VADER, TextBlob, and Ensemble Analysis.
"""

import time
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from services.sentiment_service.schemas import (
    SentimentRequest,
    SentimentResponse,
    BatchSentimentRequest,
    BatchSentimentResponse,
    SentimentLabelEnum,
    EngineTypeEnum,
)


class SentimentEngine:
    """Multi-engine sentiment analysis classifier."""

    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def _analyze_vader(self, text: str) -> SentimentResponse:
        start_time = time.perf_counter()
        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = SentimentLabelEnum.POSITIVE
        elif compound <= -0.05:
            label = SentimentLabelEnum.NEGATIVE
        else:
            label = SentimentLabelEnum.NEUTRAL

        confidence = abs(compound) if compound != 0.0 else 0.50
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return SentimentResponse(
            text=text,
            label=label,
            compound_score=round(compound, 4),
            polarity=round(compound, 4),
            subjectivity=0.50,
            confidence=round(confidence, 4),
            engine_used=EngineTypeEnum.VADER,
            latency_ms=round(latency_ms, 3),
        )

    def _analyze_textblob(self, text: str) -> SentimentResponse:
        start_time = time.perf_counter()
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0.1:
            label = SentimentLabelEnum.POSITIVE
        elif polarity < -0.1:
            label = SentimentLabelEnum.NEGATIVE
        else:
            label = SentimentLabelEnum.NEUTRAL

        confidence = abs(polarity) if polarity != 0.0 else 0.50
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return SentimentResponse(
            text=text,
            label=label,
            compound_score=round(polarity, 4),
            polarity=round(polarity, 4),
            subjectivity=round(subjectivity, 4),
            confidence=round(confidence, 4),
            engine_used=EngineTypeEnum.TEXTBLOB,
            latency_ms=round(latency_ms, 3),
        )

    def analyze_single(self, request: SentimentRequest) -> SentimentResponse:
        """Analyzes sentiment of a single text input."""
        if request.engine == EngineTypeEnum.TEXTBLOB:
            return self._analyze_textblob(request.text)
        elif request.engine == EngineTypeEnum.VADER:
            return self._analyze_vader(request.text)
        else:
            # Ensemble: Weighted combination of VADER & TextBlob
            vader_res = self._analyze_vader(request.text)
            tb_res = self._analyze_textblob(request.text)

            ensemble_compound = round(0.6 * vader_res.compound_score + 0.4 * tb_res.compound_score, 4)

            if ensemble_compound >= 0.05:
                label = SentimentLabelEnum.POSITIVE
            elif ensemble_compound <= -0.05:
                label = SentimentLabelEnum.NEGATIVE
            else:
                label = SentimentLabelEnum.NEUTRAL

            return SentimentResponse(
                text=request.text,
                label=label,
                compound_score=ensemble_compound,
                polarity=ensemble_compound,
                subjectivity=tb_res.subjectivity,
                confidence=round(max(vader_res.confidence, tb_res.confidence), 4),
                engine_used=EngineTypeEnum.ENSEMBLE,
                latency_ms=round(vader_res.latency_ms + tb_res.latency_ms, 3),
            )

    def analyze_batch(self, request: BatchSentimentRequest) -> BatchSentimentResponse:
        """Batch analyzes sentiment across bulk texts."""
        batch_start = time.perf_counter()
        results: List[SentimentResponse] = []
        pos_cnt, neu_cnt, neg_cnt = 0, 0, 0

        for text in request.texts:
            single_req = SentimentRequest(text=text, engine=request.engine)
            res = self.analyze_single(single_req)
            results.append(res)

            if res.label == SentimentLabelEnum.POSITIVE:
                pos_cnt += 1
            elif res.label == SentimentLabelEnum.NEUTRAL:
                neu_cnt += 1
            else:
                neg_cnt += 1

        batch_latency = (time.perf_counter() - batch_start) * 1000.0

        return BatchSentimentResponse(
            total_processed=len(request.texts),
            positive_count=pos_cnt,
            neutral_count=neu_cnt,
            negative_count=neg_cnt,
            results=results,
            batch_latency_ms=round(batch_latency, 3),
        )


sentiment_engine = SentimentEngine()
