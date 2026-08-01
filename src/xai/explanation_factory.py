"""
Explanation Factory Module.

Implements Factory pattern for dynamic registration and instantiation of XAI explainers.
"""

import logging
from typing import Dict, Type, Any, Optional
from src.xai.base_explainer import BaseExplainer
from src.xai.config import XAIConfig
from src.xai.exceptions import ConfigurationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ExplanationFactory:
    """Factory registry for Explainable AI engines."""

    _registry: Dict[str, Type[BaseExplainer]] = {}

    @classmethod
    def register(cls, method_name: str, explainer_cls: Type[BaseExplainer]) -> None:
        """Registers an explainer class under method_name.

        Args:
            method_name: Method identifier string (e.g., 'shap', 'lime').
            explainer_cls: Class inheriting from BaseExplainer.
        """
        key = method_name.lower().strip()
        cls._registry[key] = explainer_cls
        logger.info(f"Registered XAI Explainer: '{key}' -> {explainer_cls.__name__}")

    @classmethod
    def create(cls, config: Optional[XAIConfig] = None) -> BaseExplainer:
        """Instantiates an XAI explainer using config.

        Args:
            config: XAIConfig instance.

        Returns:
            Instantiated BaseExplainer subclass.

        Raises:
            ConfigurationError: If method is not registered.
        """
        config = config or XAIConfig()
        key = config.method.lower().strip()

        if key not in cls._registry:
            avail = list(cls._registry.keys())
            logger.error(f"Unknown XAI method: '{key}'. Available: {avail}")
            raise ConfigurationError(f"Method '{key}' not registered. Available: {avail}")

        explainer_cls = cls._registry[key]
        logger.info(f"Instantiating XAI Explainer '{key}' via {explainer_cls.__name__}...")
        return explainer_cls(config)
