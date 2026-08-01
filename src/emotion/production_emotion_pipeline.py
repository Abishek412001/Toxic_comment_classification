"""
Enterprise Production Emotion Pipeline Module (Step 97).

Provides high-throughput parallel batch inference, LRU memory caching, REST API payload serialization,
and Streamlit session hooks.
"""

import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional

from src.emotion.emotion_pipeline import EmotionPipeline
from src.emotion.config import EmotionConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProductionEmotionPipeline:
    """Deployment-ready emotion pipeline supporting caching and parallel processing."""

    def __init__(self, config: Optional[EmotionConfig] = None):
        """Initializes production pipeline.

        Args:
            config: EmotionConfig instance.
        """
        self.config = config or EmotionConfig()
        self.pipeline = EmotionPipeline(config=self.config)

    @functools.lru_cache(maxsize=1024)
    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predicts emotion for a single text with LRU memory caching.

        Args:
            text: Input text string.

        Returns:
            Dict containing emotion_label, confidence_score, probabilities, and top_emotions.
        """
        return self.pipeline.analyze_text(text)

    def predict_batch_parallel(self, texts: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
        """Predicts emotion for a batch of text strings in parallel threads.

        Args:
            texts: List of text strings.
            max_workers: Number of parallel thread workers.

        Returns:
            List of result dictionaries.
        """
        logger.info(f"Executing parallel batch emotion inference for {len(texts)} texts across {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.predict_single, texts))
        return results

    def format_rest_payload(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Formats single result for REST API response.

        Args:
            result: Result dictionary returned by predict_single.

        Returns:
            Formatted REST API response dictionary.
        """
        return {
            "status": "success",
            "data": {
                "emotion": result.get("emotion_label"),
                "confidence": result.get("confidence_score"),
                "top_emotions": result.get("top_emotions"),
                "probabilities": result.get("probabilities"),
                "engine": result.get("engine"),
            }
        }
