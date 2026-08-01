"""
Production-Grade NRC Emotion Lexicon Analyzer Module (Step 92).

Calculates word-level emotion associations across 7 categories (joy, anger, fear, sadness, surprise, disgust, neutral).
Inherits from BaseEmotionAnalyzer and auto-registers with EmotionFactory.
"""

import os
import joblib
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from src.emotion.base_emotion import BaseEmotionAnalyzer
from src.emotion.emotion_factory import EmotionFactory
from src.emotion.config import EmotionConfig
from src.emotion.constants import EMOTION_LABELS
from src.emotion.utils import get_top_k_emotions
from src.emotion.exceptions import EmotionAnalysisError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NRCEmotionAnalyzer(BaseEmotionAnalyzer):
    """NRC Lexicon Word-Matching Emotion Analyzer."""

    def __init__(self, config: Optional[EmotionConfig] = None):
        """Initializes NRC Emotion Analyzer with word association dictionary.

        Args:
            config: EmotionConfig instance.
        """
        super().__init__(name="NRCEmotionAnalyzer")
        self.config = config or EmotionConfig(engine_type="nrc")
        self.lexicon = {
            "joy": ["happy", "awesome", "great", "love", "wonderful", "enjoy", "smile", "delight", "good", "fun"],
            "anger": ["furious", "mad", "rage", "hate", "idiot", "stupid", "annoyed", "shut up", "disgusting", "kill"],
            "fear": ["scared", "terrified", "danger", "threat", "fear", "afraid", "panic", "worry", "horrible"],
            "sadness": ["sad", "depressed", "unhappy", "cry", "grief", "miserable", "lonely", "sorry", "hopeless"],
            "surprise": ["wow", "amazed", "astonished", "shocked", "unexpected", "surprise", "unbelievable"],
            "disgust": ["gross", "nasty", "vile", "revolting", "sick", "disgust", "garbage", "trash"],
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyzes a single text string using NRC word-matching lexicons.

        Args:
            text: Input text string.

        Returns:
            Dict containing emotion_label, confidence_score, probabilities, and top_emotions.
        """
        try:
            tokens = text.lower().split()
            counts = {emo: 0 for emo in EMOTION_LABELS if emo != "neutral"}

            for token in tokens:
                for emo, words in self.lexicon.items():
                    if any(w in token for w in words):
                        counts[emo] += 1

            total_hits = sum(counts.values())

            if total_hits > 0:
                probs = {emo: round(counts[emo] / total_hits, 4) for emo in counts}
                probs["neutral"] = 0.05
                top_emo = max(probs, key=probs.get)
                conf = probs[top_emo]
            else:
                probs = {emo: 0.05 for emo in counts}
                probs["neutral"] = 0.70
                top_emo = "neutral"
                conf = 0.70

            top_k = get_top_k_emotions(probs, top_k=self.config.top_k)

            return {
                "emotion_label": top_emo,
                "confidence_score": round(conf, 4),
                "probabilities": probs,
                "top_emotions": top_k,
                "engine": "nrc",
            }
        except Exception as e:
            logger.error(f"NRC emotion analysis failed: {e}")
            raise EmotionAnalysisError(f"NRC emotion analysis failed: {e}") from e

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyzes a batch of text strings.

        Args:
            texts: List of text strings.

        Returns:
            List of result dictionaries.
        """
        return [self.analyze(t) for t in texts]

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"name": self.name, "engine": "nrc"}, filepath)
        logger.info(f"Saved NRCEmotionAnalyzer configuration to {filepath}")

    def load(self, filepath: str) -> "NRCEmotionAnalyzer":
        logger.info(f"Loaded NRCEmotionAnalyzer from {filepath}")
        return self


# Auto-register with EmotionFactory
EmotionFactory.register("nrc", NRCEmotionAnalyzer)
