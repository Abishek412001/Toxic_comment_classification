"""
Feature Factory Module.

Implements Factory Method pattern for registering and instantiating feature extractors.
"""

import logging
from typing import Dict, Type, Any
from src.features.base_feature_extractor import BaseFeatureExtractor
from src.features.config import FeatureConfig
from src.features.exceptions import ConfigurationError, FeatureExtractionError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FeatureFactory:
    """Factory class registering and instantiating pluggable feature extractors."""

    _registry: Dict[str, Type[BaseFeatureExtractor]] = {}

    @classmethod
    def register(cls, feature_type: str, extractor_cls: Type[BaseFeatureExtractor]) -> None:
        """Registers a feature extractor class under a feature type key.

        Args:
            feature_type: Feature identifier string.
            extractor_cls: BaseFeatureExtractor subclass.
        """
        key = feature_type.lower()
        cls._registry[key] = extractor_cls
        logger.info(f"Registered Feature Extractor: '{key}' -> {extractor_cls.__name__}")

    @classmethod
    def create(cls, config: FeatureConfig) -> BaseFeatureExtractor:
        """Instantiates a feature extractor from supplied FeatureConfig.

        Args:
            config: FeatureConfig instance.

        Returns:
            Instantiated BaseFeatureExtractor subclass.

        Raises:
            ConfigurationError: If feature_type is not registered.
        """
        key = config.feature_type.lower()
        if key not in cls._registry:
            logger.error(f"Unknown feature_type: '{key}'. Available: {list(cls._registry.keys())}")
            raise ConfigurationError(
                f"Feature type '{key}' is not registered in FeatureFactory. "
                f"Registered types: {list(cls._registry.keys())}"
            )

        extractor_cls = cls._registry[key]
        logger.info(f"Instantiating Feature Extractor '{key}' via {extractor_cls.__name__}...")
        return extractor_cls(config)

    @classmethod
    def list_registered_types(cls) -> list:
        """Returns list of registered feature type names."""
        return list(cls._registry.keys())
