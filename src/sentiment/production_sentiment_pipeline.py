"""
Enterprise Production Sentiment Pipeline Module (Step 88).

Provides high-throughput parallel batch inference, LRU memory caching, REST API payload serialization,
and Streamlit session hooks.
"""

import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional

from src.sentiment.sentiment_pipeline import SentimentPipeline
from src.sentiment.config import SentimentConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProductionSentimentPipeline:
    """Deployment-ready sentiment pipeline supporting caching and parallel processing."""

    def __init__(self, config: Optional[SentimentConfig] = None):
        """Initializes production pipeline.

        Args:
            config: SentimentConfig instance.
        """
        self.config = config or SentimentConfig()
        self.pipeline = SentimentPipeline(config=self.config)

    @functools.lru_cache(maxsize=1024)
    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predicts sentiment for a single text with LRU memory caching.

        Args:
            text: Input text string.

        Returns:
            Dict containing sentiment label, confidence, compound score, and probabilities.
        """
        return self.pipeline.analyze_text(text)

    def predict_batch_parallel(self, texts: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
        """Predicts sentiment for a batch of text strings in parallel threads.

        Args:
            texts: List of text strings.
            max_workers: Number of parallel thread workers.

        Returns:
            List of result dictionaries.
        """
        logger.info(f"Executing parallel batch inference for {len(texts)} texts across {max_workers} workers...")
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
                "sentiment": result.get("sentiment_label"),
                "confidence": result.get("confidence_score"),
                "compound_score": result.get("compound_score"),
                "probabilities": result.get("probabilities"),
                "engine": result.get("engine"),
            }
        }
