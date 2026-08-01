"""
Emoji Removal & Conversion Module (Step 32).

Strips Unicode emojis or converts them to textual descriptions (e.g. 🤬 -> :angry_face:).
"""

import re
import logging
from typing import List
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.exceptions import CleaningError

try:
    import emoji
    HAS_EMOJI_LIB = True
except ImportError:
    HAS_EMOJI_LIB = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fallback Unicode Emoji Regex Pattern
EMOJI_REGEX = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags (iOS)
    "\U0001f900-\U0001f9ff"  # supplemental symbols & pictographs
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)


class EmojiCleaner(TextCleaner):
    """Transformer for removing or demoting emojis to text representations."""

    def __init__(self, demoji_to_text: bool = True):
        super().__init__(name="EmojiCleaner")
        self.demoji_to_text = demoji_to_text

    def transform(self, text: str) -> str:
        """Strips or converts emojis in text.

        Args:
            text: Input string.

        Returns:
            Processed string.
        """
        validated_text = self.validate_input(text, allow_empty=True)
        if not validated_text:
            return ""

        try:
            if HAS_EMOJI_LIB:
                if self.demoji_to_text:
                    cleaned = emoji.demojize(validated_text, delimiters=(" :", ": "))
                else:
                    cleaned = emoji.replace_emoji(validated_text, replace="")
            else:
                if self.demoji_to_text:
                    cleaned = EMOJI_REGEX.sub(" :emoji: ", validated_text)
                else:
                    cleaned = EMOJI_REGEX.sub(" ", validated_text)

            return re.sub(r"\s+", " ", cleaned).strip()
        except Exception as e:
            logger.error(f"Error in EmojiCleaner: {e}")
            raise CleaningError(f"Emoji processing failed: {e}") from e

    def batch_remove_emojis(self, texts: List[str]) -> List[str]:
        """Strips or converts emojis across a batch of strings."""
        return self.transform_batch(texts)


def remove_emojis(text: str, demoji_to_text: bool = True) -> str:
    """Functional wrapper for emoji processing."""
    return EmojiCleaner(demoji_to_text=demoji_to_text).transform(text)


def batch_remove_emojis(texts: List[str], demoji_to_text: bool = True) -> List[str]:
    """Functional wrapper for batch emoji processing."""
    return EmojiCleaner(demoji_to_text=demoji_to_text).transform_batch(texts)
