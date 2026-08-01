"""
Emotion Pipeline Module.

Master pipeline coordinating validation, engine execution, top-3 emotion extraction, and batch inference.
"""

import logging
from typing import Dict, Any, List, Optional
from src.emotion.base_emotion import BaseEmotionAnalyzer
from src.emotion.emotion_factory import EmotionFactory
from src.emotion.config import EmotionConfig
from src.emotion.validator import EmotionValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EmotionPipeline:
    """Master Emotion Mining Pipeline."""

    def __init__(self, analyzer: Optional[BaseEmotionAnalyzer] = None, config: Optional[EmotionConfig] = None):
        """Initializes pipeline.

        Args:
            analyzer: Instantiated BaseEmotionAnalyzer subclass or None to instantiate via factory.
            config: EmotionConfig instance.
        """
        self.config = config or EmotionConfig()
        self.analyzer = analyzer or EmotionFactory.create(self.config)

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Validates and analyzes a single text.

        Args:
            text: Input string.

        Returns:
            Dict containing emotion_label, confidence, probabilities, and top_emotions.
        """
        val_text = EmotionValidator.validate_text(text)
        return self.analyzer.analyze(val_text)

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Validates and analyzes a batch of text strings.

        Args:
            texts: List of input strings.

        Returns:
            List of result dictionaries.
        """
        val_texts = EmotionValidator.validate_batch(texts)
        return self.analyzer.analyze_batch(val_texts)
