"""
Word2Vec Feature Extraction Module (Step 44).

Trains or loads Gensim Word2Vec embeddings and performs average document vector pooling.
Inherits from BaseFeatureExtractor and auto-registers with FeatureFactory.
"""

import os
import joblib
import logging
from typing import List, Any, Optional, Dict
import numpy as np

from src.features.base_feature_extractor import BaseFeatureExtractor
from src.features.feature_factory import FeatureFactory
from src.features.config import FeatureConfig
from src.features.exceptions import FeatureExtractionError
from src.features.constants import FeatureType

try:
    from gensim.models import Word2Vec
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Word2VecFeatureExtractor(BaseFeatureExtractor):
    """Word2Vec Feature Extractor generating dense sentence embeddings via mean vector pooling."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(name="Word2VecFeatureExtractor")
        self.config = config or FeatureConfig(feature_type=FeatureType.WORD2VEC.value)
        self.vector_size = self.config.embedding_dimension
        self.model = None

    def fit(self, texts: List[str]) -> "Word2VecFeatureExtractor":
        """Trains Word2Vec model on tokenized sentences."""
        try:
            sentences = [str(t).lower().split() for t in texts]
            if HAS_GENSIM:
                self.model = Word2Vec(
                    sentences=sentences,
                    vector_size=self.vector_size,
                    window=5,
                    min_count=1,
                    workers=4,
                    epochs=10,
                    sg=1,  # Skip-Gram
                )
            else:
                logger.warning("Gensim not installed. Operating in fallback vector lookup mode.")
                self.model = {}

            self.is_fitted = True
            logger.info(f"Fitted Word2VecFeatureExtractor (Vector Size = {self.vector_size}d)")
            return self
        except Exception as e:
            logger.error(f"Error fitting Word2VecFeatureExtractor: {e}")
            raise FeatureExtractionError(f"Word2Vec fitting failed: {e}") from e

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transforms text corpus into dense N x vector_size matrix via mean pooling."""
        if not self.is_fitted:
            raise FeatureExtractionError("Word2VecFeatureExtractor must be fitted before calling transform().")

        try:
            embeddings = []
            for text in texts:
                tokens = str(text).lower().split()
                vectors = []
                for tok in tokens:
                    if HAS_GENSIM and self.model and hasattr(self.model, "wv") and tok in self.model.wv:
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
            logger.error(f"Error transforming texts in Word2VecFeatureExtractor: {e}")
            raise FeatureExtractionError(f"Word2Vec transformation failed: {e}") from e

    def save(self, filepath: str) -> None:
        """Serializes Word2Vec model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if HAS_GENSIM and isinstance(self.model, Word2Vec):
            self.model.save(filepath)
        else:
            joblib.dump({"model": self.model, "dim": self.vector_size}, filepath)
        logger.info(f"Saved Word2VecFeatureExtractor to {filepath}")

    def load(self, filepath: str) -> "Word2VecFeatureExtractor":
        """Deserializes Word2Vec model from disk."""
        if HAS_GENSIM:
            try:
                self.model = Word2Vec.load(filepath)
            except Exception:
                data = joblib.load(filepath)
                self.model = data.get("model", {})
        else:
            data = joblib.load(filepath)
            self.model = data.get("model", {})

        self.is_fitted = True
        logger.info(f"Loaded Word2VecFeatureExtractor from {filepath}")
        return self

    def get_feature_names(self) -> List[str]:
        """Returns feature dimension names."""
        return [f"w2v_dim_{i}" for i in range(self.vector_size)]


# Auto-register with FeatureFactory
FeatureFactory.register(FeatureType.WORD2VEC.value, Word2VecFeatureExtractor)
