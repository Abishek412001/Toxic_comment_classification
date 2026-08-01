"""
Bag of Words (BoW) Feature Extraction Module (Step 42).

Provides production-grade BoWVectorizer class wrapping Scikit-Learn CountVectorizer.
Inherits from BaseFeatureExtractor and auto-registers with FeatureFactory.
"""

import os
import joblib
import logging
from typing import List, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from src.features.base_feature_extractor import BaseFeatureExtractor
from src.features.feature_factory import FeatureFactory
from src.features.config import FeatureConfig
from src.features.exceptions import FeatureExtractionError
from src.features.constants import FeatureType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BoWVectorizer(BaseFeatureExtractor):
    """Bag of Words Feature Extractor wrapping CountVectorizer."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(name="BoWVectorizer")
        self.config = config or FeatureConfig(feature_type=FeatureType.BAG_OF_WORDS.value)
        self.vectorizer = CountVectorizer(
            max_features=self.config.max_features,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            ngram_range=self.config.ngram_range,
        )

    def fit(self, texts: List[str]) -> "BoWVectorizer":
        """Fits CountVectorizer vocabulary on text sequence."""
        try:
            if len(texts) < self.config.min_df:
                self.vectorizer.set_params(min_df=1)
            self.vectorizer.fit(texts)
            self.is_fitted = True
            logger.info(f"Fitted BoWVectorizer. Vocabulary Size = {len(self.vectorizer.vocabulary_):,}")
            return self
        except Exception as e:
            logger.error(f"Error fitting BoWVectorizer: {e}")
            raise FeatureExtractionError(f"BoW fitting failed: {e}") from e

    def transform(self, texts: List[str]) -> Any:
        """Transforms texts into SciPy sparse BoW matrix."""
        if not self.is_fitted:
            raise FeatureExtractionError("BoWVectorizer must be fitted before calling transform().")
        try:
            return self.vectorizer.transform(texts)
        except Exception as e:
            logger.error(f"Error transforming texts in BoWVectorizer: {e}")
            raise FeatureExtractionError(f"BoW transformation failed: {e}") from e

    def save(self, filepath: str) -> None:
        """Serializes vectorizer to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        logger.info(f"Saved BoWVectorizer to {filepath}")

    def load(self, filepath: str) -> "BoWVectorizer":
        """Deserializes vectorizer from disk."""
        self.vectorizer = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded BoWVectorizer from {filepath}")
        return self

    def get_feature_names(self) -> List[str]:
        """Returns list of vocabulary feature names."""
        if not self.is_fitted:
            return []
        return list(self.vectorizer.get_feature_names_out())


# Auto-register with FeatureFactory
FeatureFactory.register(FeatureType.BAG_OF_WORDS.value, BoWVectorizer)
