"""
Tokenizer Wrapper Module.

Provides configurable tokenization methods (whitespace, regex, subword).
"""

import re
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Tokenizer:
    """Tokenizer class supporting whitespace and regex-based tokenization."""

    def __init__(self, mode: str = "word"):
        """Initializes tokenizer.

        Args:
            mode: Tokenization mode ("word", "whitespace", "char").
        """
        self.mode = mode

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into a list of string tokens.

        Args:
            text: Input string.

        Returns:
            List of tokens.
        """
        if not text or not isinstance(text, str):
            return []

        if self.mode == "whitespace":
            return text.split()
        elif self.mode == "char":
            return list(text)
        else:
            # Default word regex
            return re.findall(r"\b\w+\b", text)

    def detokenize(self, tokens: List[str]) -> str:
        """Joins tokens back into a single string.

        Args:
            tokens: List of token strings.

        Returns:
            Joined text string.
        """
        return " ".join(tokens)
