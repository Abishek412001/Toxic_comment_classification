"""
Sentiment Pipeline Module.

Master pipeline coordinating validation, engine execution, confidence score normalization, and batch inference.
"""

import logging
from typing import Dict, Any, List, Optional
from src.sentiment.base_sentiment import BaseSentimentAnalyzer
from src.sentiment.sentiment_factory import SentimentFactory
from src.sentiment.config import SentimentConfig
from src.sentiment.validator import SentimentValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SentimentPipeline:
    """Master Sentiment Analysis Pipeline."""

    def __init__(self, analyzer: Optional[BaseSentimentAnalyzer] = None, config: Optional[SentimentConfig] = None):
        """Initializes pipeline.

        Args:
            analyzer: Instantiated BaseSentimentAnalyzer subclass or None to instantiate via factory.
            config: SentimentConfig instance.
        """
        self.config = config or SentimentConfig()
        self.analyzer = analyzer or SentimentFactory.create(self.config)

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Validates and analyzes a single text.

        Args:
            text: Input string.

        Returns:
            Dict containing sentiment label, confidence, score, and probabilities.
        """
        val_text = SentimentValidator.validate_text(text)
        return self.analyzer.analyze(val_text)

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Validates and analyzes a batch of text strings.

        Args:
            texts: List of input strings.

        Returns:
            List of result dictionaries.
        """
        val_texts = SentimentValidator.validate_batch(texts)
        return self.analyzer.analyze_batch(val_texts)
