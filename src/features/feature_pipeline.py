"""
Feature Pipeline Module.

Coordinates validation, extractor instantiation via FeatureFactory,
feature matrix generation, shape verification, and artifact serialization.
"""

import os
import logging
from typing import List, Any, Optional
from src.features.config import FeatureConfig, ConfigurationManager
from src.features.feature_factory import FeatureFactory
from src.features.feature_validator import FeatureValidator
from src.features.base_feature_extractor import BaseFeatureExtractor
from src.features.exceptions import FeatureExtractionError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Master Feature Engineering Pipeline coordinating extractors and validation."""

    def __init__(self, config: Optional[FeatureConfig] = None, extractor: Optional[BaseFeatureExtractor] = None):
        """Initializes pipeline with supplied configuration or custom extractor instance.

        Args:
            config: FeatureConfig instance.
            extractor: Pre-instantiated BaseFeatureExtractor subclass instance.
        """
        self.config = config or ConfigurationManager.get_traditional_ml_config()
        if extractor is not None:
            self.extractor = extractor
        else:
            self.extractor = FeatureFactory.create(self.config)

    def fit(self, texts: List[str]) -> "FeaturePipeline":
        """Fits feature extractor on input text sequence.

        Args:
            texts: Candidate text corpus.

        Returns:
            Fitted pipeline instance.
        """
        validated_texts = FeatureValidator.validate_input_texts(texts)
        logger.info(f"Fitting FeaturePipeline ({self.extractor.name}) on {len(validated_texts):,} samples...")
        self.extractor.fit(validated_texts)
        return self

    def transform(self, texts: List[str]) -> Any:
        """Transforms text sequence into feature matrix.

        Args:
            texts: Candidate text corpus.

        Returns:
            Sparse or Dense Feature Matrix.
        """
        validated_texts = FeatureValidator.validate_input_texts(texts)
        logger.info(f"Generating feature vectors for {len(validated_texts):,} samples via {self.extractor.name}...")
        matrix = self.extractor.transform(validated_texts)
        FeatureValidator.validate_feature_matrix(matrix, expected_rows=len(validated_texts))
        return matrix

    def fit_transform(self, texts: List[str]) -> Any:
        """Fits extractor and transforms text sequence in one pass.

        Args:
            texts: Candidate text corpus.

        Returns:
            Sparse or Dense Feature Matrix.
        """
        return self.fit(texts).transform(texts)

    def save(self, filepath: Optional[str] = None) -> None:
        """Serializes fitted feature extractor artifact to disk.

        Args:
            filepath: Target file path.
        """
        target_path = filepath or os.path.join(self.config.artifact_dir, f"{self.extractor.name}.joblib")
        logger.info(f"Saving FeaturePipeline artifact to {target_path}...")
        self.extractor.save(target_path)

    def load(self, filepath: str) -> "FeaturePipeline":
        """Deserializes fitted feature extractor artifact from disk.

        Args:
            filepath: Input file path.

        Returns:
            Loaded pipeline instance.
        """
        logger.info(f"Loading FeaturePipeline artifact from {filepath}...")
        self.extractor.load(filepath)
        return self
