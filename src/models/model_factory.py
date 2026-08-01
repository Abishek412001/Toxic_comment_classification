"""
Model Factory Module.

Implements Factory Method pattern for registering and instantiating multi-label classifiers.
"""

import logging
from typing import Dict, Type, Any
from src.models.base_model import BaseModel
from src.models.config import ModelConfig
from src.models.exceptions import ConfigurationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class registering and instantiating multi-label classifiers."""

    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, model_name: str, model_cls: Type[BaseModel]) -> None:
        """Registers a model class under a model name key.

        Args:
            model_name: Model identifier string.
            model_cls: BaseModel subclass.
        """
        key = model_name.lower()
        cls._registry[key] = model_cls
        logger.info(f"Registered Model: '{key}' -> {model_cls.__name__}")

    @classmethod
    def create(cls, config: ModelConfig) -> BaseModel:
        """Instantiates a model from supplied ModelConfig.

        Args:
            config: ModelConfig instance.

        Returns:
            Instantiated BaseModel subclass.

        Raises:
            ConfigurationError: If model_name is not registered.
        """
        key = config.model_name.lower()
        if key not in cls._registry:
            logger.error(f"Unknown model_name: '{key}'. Registered models: {list(cls._registry.keys())}")
            raise ConfigurationError(
                f"Model name '{key}' is not registered in ModelFactory. "
                f"Registered types: {list(cls._registry.keys())}"
            )

        model_cls = cls._registry[key]
        logger.info(f"Instantiating Model '{key}' via {model_cls.__name__}...")
        return model_cls(config)

    @classmethod
    def list_registered_models(cls) -> list:
        """Returns list of registered model names."""
        return list(cls._registry.keys())
