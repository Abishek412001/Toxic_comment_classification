"""
Contraction Expansion Module (Step 28).

Expands English contractions (e.g. don't -> do not, I'm -> I am).
Uses contractions library with a comprehensive dictionary fallback.
"""

import re
import logging
from typing import List, Dict, Optional
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.exceptions import CleaningError

try:
    import contractions as contractions_lib
    HAS_CONTRACTIONS_LIB = True
except ImportError:
    HAS_CONTRACTIONS_LIB = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fallback English Contraction Map
DEFAULT_CONTRACTION_MAP: Dict[str, str] = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "doesn't": "does not",
    "didn't": "did not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "mustn't": "must not",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "what's": "what is",
    "that's": "that is",
    "who's": "who is",
    "where's": "where is",
    "how's": "how is",
}


class ContractionExpander(TextCleaner):
    """Transformer for expanding English contractions into full canonical words."""

    def __init__(self, custom_map: Optional[Dict[str, str]] = None):
        super().__init__(name="ContractionExpander")
        self.contraction_map = DEFAULT_CONTRACTION_MAP.copy()
        if custom_map:
            self.contraction_map.update({k.lower(): v.lower() for k, v in custom_map.items()})

        # Pre-compile regex for dictionary fallback
        pattern = r"\b(" + "|".join(re.escape(key) for key in self.contraction_map.keys()) + r")\b"
        self.regex_pattern = re.compile(pattern, re.IGNORECASE)

    def transform(self, text: str) -> str:
        """Expands contractions in text.

        Args:
            text: Input string.

        Returns:
            Expanded text string.
        """
        validated_text = self.validate_input(text, allow_empty=True)
        if not validated_text:
            return ""

        try:
            if HAS_CONTRACTIONS_LIB:
                return contractions_lib.fix(validated_text)
            else:
                def replace_match(match):
                    word = match.group(0)
                    lower_word = word.lower()
                    replacement = self.contraction_map.get(lower_word, word)
                    if word.isupper():
                        return replacement.upper()
                    elif word[0].isupper():
                        return replacement.capitalize()
                    return replacement

                return self.regex_pattern.sub(replace_match, validated_text)
        except Exception as e:
            logger.error(f"Error in ContractionExpander: {e}")
            raise CleaningError(f"Contraction expansion failed: {e}") from e

    def batch_expand(self, texts: List[str]) -> List[str]:
        """Expands contractions across a batch of strings."""
        return self.transform_batch(texts)


def expand_contractions(text: str) -> str:
    """Functional wrapper for contraction expansion."""
    return ContractionExpander().transform(text)


def batch_expand_contractions(texts: List[str]) -> List[str]:
    """Functional wrapper for batch contraction expansion."""
    return ContractionExpander().transform_batch(texts)
