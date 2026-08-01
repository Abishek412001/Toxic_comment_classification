"""
Stopword Removal Module (Step 37).

Strips standard English and custom domain-specific stopwords.
Supports NLTK, spaCy, and custom stopword sets.
"""

import re
import logging
from typing import List, Set, Optional
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.exceptions import CleaningError
from src.preprocessing.constants import DEFAULT_DOMAIN_STOPWORDS

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    try:
        NLTK_STOPS = set(nltk_stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        NLTK_STOPS = set(nltk_stopwords.words("english"))
    HAS_NLTK = True
except Exception:
    NLTK_STOPS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "this", "that",
    }
    HAS_NLTK = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class StopwordRemover(TextCleaner):
    """Transformer for filtering stopwords from text."""

    def __init__(self, include_domain_stopwords: bool = True, custom_stopwords: Optional[Set[str]] = None):
        super().__init__(name="StopwordRemover")
        self.stopwords = NLTK_STOPS.copy()
        if include_domain_stopwords:
            self.stopwords.update(DEFAULT_DOMAIN_STOPWORDS)
        if custom_stopwords:
            self.stopwords.update({w.lower() for w in custom_stopwords})

    def transform(self, text: str) -> str:
        """Strips stopwords from text.

        Args:
            text: Input string.

        Returns:
            Processed string with stopwords removed.
        """
        validated_text = self.validate_input(text, allow_empty=True)
        if not validated_text:
            return ""

        try:
            tokens = validated_text.split()
            filtered = [t for t in tokens if t.lower() not in self.stopwords]
            return " ".join(filtered)
        except Exception as e:
            logger.error(f"Error in StopwordRemover: {e}")
            raise CleaningError(f"Stopword removal failed: {e}") from e

    def batch_remove_stopwords(self, texts: List[str]) -> List[str]:
        """Strips stopwords across a batch of strings."""
        return self.transform_batch(texts)


def remove_stopwords(text: str, custom_stopwords: Optional[Set[str]] = None) -> str:
    """Functional wrapper for stopword removal."""
    return StopwordRemover(custom_stopwords=custom_stopwords).transform(text)


def batch_remove_stopwords(texts: List[str], custom_stopwords: Optional[Set[str]] = None) -> List[str]:
    """Functional wrapper for batch stopword removal."""
    return StopwordRemover(custom_stopwords=custom_stopwords).transform_batch(texts)
