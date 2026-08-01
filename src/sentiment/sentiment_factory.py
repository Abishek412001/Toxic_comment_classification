"""
Sentiment Factory Module.

Implements Factory pattern for dynamic registration and instantiation of sentiment engines.
"""

import logging
from typing import Dict, Type, Any, Optional
from src.sentiment.base_sentiment import BaseSentimentAnalyzer
from src.sentiment.config import SentimentConfig
from src.sentiment.exceptions import ConfigurationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SentimentFactory:
    """Factory registry for sentiment analysis engines."""

    _registry: Dict[str, Type[BaseSentimentAnalyzer]] = {}

    @classmethod
    def register(cls, engine_name: str, analyzer_cls: Type[BaseSentimentAnalyzer]) -> None:
        """Registers a sentiment analyzer class under engine_name.

        Args:
            engine_name: Engine identifier string (e.g., 'vader', 'textblob').
            analyzer_cls: Class inheriting from BaseSentimentAnalyzer.
        """
        key = engine_name.lower().strip()
        cls._registry[key] = analyzer_cls
        logger.info(f"Registered Sentiment Engine: '{key}' -> {analyzer_cls.__name__}")

    @classmethod
    def create(cls, config: Optional[SentimentConfig] = None) -> BaseSentimentAnalyzer:
        """Instantiates a sentiment analyzer using config.

        Args:
            config: SentimentConfig instance.

        Returns:
            Instantiated BaseSentimentAnalyzer subclass.

        Raises:
            ConfigurationError: If engine_type is not registered.
        """
        config = config or SentimentConfig()
        key = config.engine_type.lower().strip()

        if key not in cls._registry:
            avail = list(cls._registry.keys())
            logger.error(f"Unknown engine_type: '{key}'. Available: {avail}")
            raise ConfigurationError(f"Engine '{key}' not registered. Available: {avail}")

        analyzer_cls = cls._registry[key]
        logger.info(f"Instantiating Sentiment Engine '{key}' via {analyzer_cls.__name__}...")
        return analyzer_cls(config)
