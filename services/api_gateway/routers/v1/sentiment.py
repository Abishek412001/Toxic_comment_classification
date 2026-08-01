"""
Sentiment Service API Gateway Router.
"""

from fastapi import APIRouter, Depends
from services.sentiment_service.schemas import (
    SentimentRequest,
    SentimentResponse,
    BatchSentimentRequest,
    BatchSentimentResponse,
)
from services.sentiment_service.engine import sentiment_engine
from opentrust_core.auth.dependencies import check_api_key_rate_limit
from opentrust_core.schemas.response import APIResponse

router = APIRouter(prefix="/sentiment", tags=["Sentiment Intelligence"])


@router.post("/analyze", response_model=APIResponse[SentimentResponse])
async def analyze_sentiment(
    request: SentimentRequest,
    rate_info: dict = Depends(check_api_key_rate_limit),
):
    """Analyzes sentiment polarity, subjectivity, and compound scores for single text."""
    res = sentiment_engine.analyze_single(request)
    return APIResponse[SentimentResponse](
        data=res,
        message="Sentiment analysis completed successfully.",
    )


@router.post("/batch", response_model=APIResponse[BatchSentimentResponse])
async def analyze_sentiment_batch(
    request: BatchSentimentRequest,
    rate_info: dict = Depends(check_api_key_rate_limit),
):
    """Batch analyzes sentiment across bulk texts."""
    res = sentiment_engine.analyze_batch(request)
    return APIResponse[BatchSentimentResponse](
        data=res,
        message=f"Batch sentiment analysis completed for {res.total_processed} items.",
    )


@router.get("/engines", response_model=APIResponse[dict])
async def list_sentiment_engines():
    """Lists supported sentiment engines and description."""
    return APIResponse[dict](
        data={
            "vader": "VADER Lexicon Rule-Based Sentiment Analyzer",
            "textblob": "TextBlob Polarity & Subjectivity Analyzer",
            "ensemble": "Weighted Ensemble Analyzer (VADER + TextBlob)",
        }
    )
