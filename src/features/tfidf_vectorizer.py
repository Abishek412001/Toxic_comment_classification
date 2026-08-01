"""
TF-IDF Feature Extraction Module (Step 43).

Provides production-grade TFIDFFeatureExtractor wrapping Scikit-Learn TfidfVectorizer.
Supports sublinear_tf, smooth_idf, ngram_range=(1,2).
Inherits from BaseFeatureExtractor and auto-registers with FeatureFactory.
"""

import os
import joblib
import logging
from typing import List, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.base_feature_extractor import BaseFeatureExtractor
from src.features.feature_factory import FeatureFactory
from src.features.config import FeatureConfig
from src.features.exceptions import FeatureExtractionError
from src.features.constants import FeatureType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TFIDFFeatureExtractor(BaseFeatureExtractor):
    """TF-IDF Feature Extractor wrapping TfidfVectorizer."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(name="TFIDFFeatureExtractor")
        self.config = config or FeatureConfig(feature_type=FeatureType.TFIDF.value)
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            ngram_range=self.config.ngram_range,
            sublinear_tf=self.config.sublinear_tf,
            smooth_idf=True,
            norm="l2",
        )

    def fit(self, texts: List[str]) -> "TFIDFFeatureExtractor":
        """Fits TfidfVectorizer on text sequence."""
        try:
            if len(texts) < self.config.min_df:
                self.vectorizer.set_params(min_df=1)
            self.vectorizer.fit(texts)
            self.is_fitted = True
            logger.info(f"Fitted TFIDFFeatureExtractor. Vocabulary Size = {len(self.vectorizer.vocabulary_):,}")
            return self
        except Exception as e:
            logger.error(f"Error fitting TFIDFFeatureExtractor: {e}")
            raise FeatureExtractionError(f"TF-IDF fitting failed: {e}") from e

    def transform(self, texts: List[str]) -> Any:
        """Transforms texts into SciPy sparse TF-IDF matrix."""
        if not self.is_fitted:
            raise FeatureExtractionError("TFIDFFeatureExtractor must be fitted before calling transform().")
        try:
            return self.vectorizer.transform(texts)
        except Exception as e:
            logger.error(f"Error transforming texts in TFIDFFeatureExtractor: {e}")
            raise FeatureExtractionError(f"TF-IDF transformation failed: {e}") from e

    def save(self, filepath: str) -> None:
        """Serializes vectorizer to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        logger.info(f"Saved TFIDFFeatureExtractor to {filepath}")

    def load(self, filepath: str) -> "TFIDFFeatureExtractor":
        """Deserializes vectorizer from disk."""
        self.vectorizer = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded TFIDFFeatureExtractor from {filepath}")
        return self

    def get_feature_names(self) -> List[str]:
        """Returns list of vocabulary feature names."""
        if not self.is_fitted:
            return []
        return list(self.vectorizer.get_feature_names_out())


# Auto-register with FeatureFactory
FeatureFactory.register(FeatureType.TFIDF.value, TFIDFFeatureExtractor)
