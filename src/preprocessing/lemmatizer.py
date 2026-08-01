"""
Lemmatization Module (Step 38).

Reduces words to canonical dictionary lemmas using spaCy or NLTK WordNet backends.
"""

import logging
from typing import List, Optional
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.exceptions import CleaningError

try:
    import spacy
    try:
        NLP_SPACY = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        HAS_SPACY = True
    except Exception:
        NLP_SPACY = None
        HAS_SPACY = False
except ImportError:
    NLP_SPACY = None
    HAS_SPACY = False

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    try:
        WORDNET_LEMMATIZER = WordNetLemmatizer()
        WORDNET_LEMMATIZER.lemmatize("testing")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        WORDNET_LEMMATIZER = WordNetLemmatizer()
    HAS_WORDNET = True
except Exception:
    WORDNET_LEMMATIZER = None
    HAS_WORDNET = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Lemmatizer(TextCleaner):
    """Transformer for lemmatizing text into canonical base forms."""

    def __init__(self, backend: str = "spacy"):
        super().__init__(name="Lemmatizer")
        self.backend = backend.lower()

    def transform(self, text: str) -> str:
        """Lemmatizes text into canonical dictionary forms.

        Args:
            text: Input string.

        Returns:
            Lemmatized text string.
        """
        validated_text = self.validate_input(text, allow_empty=True)
        if not validated_text:
            return ""

        try:
            if self.backend == "spacy" and HAS_SPACY and NLP_SPACY is not None:
                doc = NLP_SPACY(validated_text)
                return " ".join([token.lemma_ for token in doc])
            elif HAS_WORDNET and WORDNET_LEMMATIZER is not None:
                tokens = validated_text.split()
                lemmas = [WORDNET_LEMMATIZER.lemmatize(t) for t in tokens]
                return " ".join(lemmas)
            else:
                # Rule-based fallback if no external backends are present
                return validated_text
        except Exception as e:
            logger.error(f"Error in Lemmatizer: {e}")
            raise CleaningError(f"Lemmatization failed: {e}") from e

    def batch_lemmatize(self, texts: List[str]) -> List[str]:
        """Lemmatizes across a batch of strings."""
        return self.transform_batch(texts)


def lemmatize_text(text: str, backend: str = "spacy") -> str:
    """Functional wrapper for text lemmatization."""
    return Lemmatizer(backend=backend).transform(text)


def batch_lemmatize_text(texts: List[str], backend: str = "spacy") -> List[str]:
    """Functional wrapper for batch text lemmatization."""
    return Lemmatizer(backend=backend).transform_batch(texts)
