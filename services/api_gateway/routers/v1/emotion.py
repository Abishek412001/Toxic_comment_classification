"""
Emotion Service API Gateway Router.
"""

from fastapi import APIRouter, Depends
from services.emotion_service.schemas import (
    EmotionRequest,
    EmotionResponse,
    BatchEmotionRequest,
    BatchEmotionResponse,
)
from services.emotion_service.engine import emotion_engine
from opentrust_core.auth.dependencies import check_api_key_rate_limit
from opentrust_core.schemas.response import APIResponse

router = APIRouter(prefix="/emotion", tags=["Emotion Intelligence"])


@router.post("/detect", response_model=APIResponse[EmotionResponse])
async def detect_emotion(
    request: EmotionRequest,
    rate_info: dict = Depends(check_api_key_rate_limit),
):
    """Detects multi-class emotion distribution and top-N emotions for single text."""
    res = emotion_engine.detect_single(request)
    return APIResponse[EmotionResponse](
        data=res,
        message="Emotion detection completed successfully.",
    )


@router.post("/batch", response_model=APIResponse[BatchEmotionResponse])
async def detect_emotion_batch(
    request: BatchEmotionRequest,
    rate_info: dict = Depends(check_api_key_rate_limit),
):
    """Batch detects multi-class emotions across bulk texts."""
    res = emotion_engine.detect_batch(request)
    return APIResponse[BatchEmotionResponse](
        data=res,
        message=f"Batch emotion detection completed for {res.total_processed} items.",
    )


@router.get("/lexicon", response_model=APIResponse[dict])
async def get_emotion_lexicon_info():
    """Returns details on the 8 NRC emotion categories."""
    return APIResponse[dict](
        data={
            "emotions": [
                "anger",
                "anticipation",
                "disgust",
                "fear",
                "joy",
                "sadness",
                "surprise",
                "trust",
            ],
            "framework": "NRC Emotion Lexicon & Semantic Probability Modeling",
        }
    )
