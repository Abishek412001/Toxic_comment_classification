"""
Emotion Factory Module.

Implements Factory pattern for dynamic registration and instantiation of emotion engines.
"""

import logging
from typing import Dict, Type, Any, Optional
from src.emotion.base_emotion import BaseEmotionAnalyzer
from src.emotion.config import EmotionConfig
from src.emotion.exceptions import ConfigurationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EmotionFactory:
    """Factory registry for emotion mining engines."""

    _registry: Dict[str, Type[BaseEmotionAnalyzer]] = {}

    @classmethod
    def register(cls, engine_name: str, analyzer_cls: Type[BaseEmotionAnalyzer]) -> None:
        """Registers an emotion analyzer class under engine_name.

        Args:
            engine_name: Engine identifier string (e.g., 'nrc', 'transformer').
            analyzer_cls: Class inheriting from BaseEmotionAnalyzer.
        """
        key = engine_name.lower().strip()
        cls._registry[key] = analyzer_cls
        logger.info(f"Registered Emotion Engine: '{key}' -> {analyzer_cls.__name__}")

    @classmethod
    def create(cls, config: Optional[EmotionConfig] = None) -> BaseEmotionAnalyzer:
        """Instantiates an emotion analyzer using config.

        Args:
            config: EmotionConfig instance.

        Returns:
            Instantiated BaseEmotionAnalyzer subclass.

        Raises:
            ConfigurationError: If engine_type is not registered.
        """
        config = config or EmotionConfig()
        key = config.engine_type.lower().strip()

        if key not in cls._registry:
            avail = list(cls._registry.keys())
            logger.error(f"Unknown engine_type: '{key}'. Available: {avail}")
            raise ConfigurationError(f"Engine '{key}' not registered. Available: {avail}")

        analyzer_cls = cls._registry[key]
        logger.info(f"Instantiating Emotion Engine '{key}' via {analyzer_cls.__name__}...")
        return analyzer_cls(config)
