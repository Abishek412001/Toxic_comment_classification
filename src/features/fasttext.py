"""
FastText Feature Extraction Module (Step 45).

Trains or loads Gensim FastText embeddings using subword character n-grams to resolve OOV toxic typos.
Inherits from BaseFeatureExtractor and auto-registers with FeatureFactory.
"""

import os
import joblib
import logging
from typing import List, Any, Optional
import numpy as np

from src.features.base_feature_extractor import BaseFeatureExtractor
from src.features.feature_factory import FeatureFactory
from src.features.config import FeatureConfig
from src.features.exceptions import FeatureExtractionError
from src.features.constants import FeatureType

try:
    from gensim.models import FastText
    HAS_FASTTEXT = True
except ImportError:
    HAS_FASTTEXT = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FastTextFeatureExtractor(BaseFeatureExtractor):
    """FastText Feature Extractor generating subword character n-gram dense embeddings."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(name="FastTextFeatureExtractor")
        self.config = config or FeatureConfig(feature_type=FeatureType.FASTTEXT.value)
        self.vector_size = self.config.embedding_dimension
        self.model = None

    def fit(self, texts: List[str]) -> "FastTextFeatureExtractor":
        """Trains FastText model on tokenized sentences."""
        try:
            sentences = [str(t).lower().split() for t in texts]
            if HAS_FASTTEXT:
                self.model = FastText(
                    sentences=sentences,
                    vector_size=self.vector_size,
                    window=5,
                    min_count=1,
                    workers=4,
                    epochs=10,
                )
            else:
                logger.warning("FastText not installed. Operating in fallback lookup mode.")
                self.model = {}

            self.is_fitted = True
            logger.info(f"Fitted FastTextFeatureExtractor (Vector Size = {self.vector_size}d)")
            return self
        except Exception as e:
            logger.error(f"Error fitting FastTextFeatureExtractor: {e}")
            raise FeatureExtractionError(f"FastText fitting failed: {e}") from e

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transforms text corpus into dense N x vector_size matrix via subword pooling."""
        if not self.is_fitted:
            raise FeatureExtractionError("FastTextFeatureExtractor must be fitted before calling transform().")

        try:
            embeddings = []
            for text in texts:
                tokens = str(text).lower().split()
                vectors = []
                for tok in tokens:
                    if HAS_FASTTEXT and self.model and hasattr(self.model, "wv"):
                        # FastText handles OOV using subwords automatically
                        vectors.append(self.model.wv[tok])
                    elif isinstance(self.model, dict) and tok in self.model:
                        vectors.append(self.model[tok])

                if vectors:
                    mean_vec = np.mean(vectors, axis=0)
                else:
                    mean_vec = np.zeros(self.vector_size)

                embeddings.append(mean_vec)

            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error transforming texts in FastTextFeatureExtractor: {e}")
            raise FeatureExtractionError(f"FastText transformation failed: {e}") from e

    def save(self, filepath: str) -> None:
        """Serializes FastText model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if HAS_FASTTEXT and isinstance(self.model, FastText):
            self.model.save(filepath)
        else:
            joblib.dump({"model": self.model, "dim": self.vector_size}, filepath)
        logger.info(f"Saved FastTextFeatureExtractor to {filepath}")

    def load(self, filepath: str) -> "FastTextFeatureExtractor":
        """Deserializes FastText model from disk."""
        if HAS_FASTTEXT:
            try:
                self.model = FastText.load(filepath)
            except Exception:
                data = joblib.load(filepath)
                self.model = data.get("model", {})
        else:
            data = joblib.load(filepath)
            self.model = data.get("model", {})

        self.is_fitted = True
        logger.info(f"Loaded FastTextFeatureExtractor from {filepath}")
        return self

    def get_feature_names(self) -> List[str]:
        """Returns feature dimension names."""
        return [f"ft_dim_{i}" for i in range(self.vector_size)]


# Auto-register with FeatureFactory
FeatureFactory.register(FeatureType.FASTTEXT.value, FastTextFeatureExtractor)
