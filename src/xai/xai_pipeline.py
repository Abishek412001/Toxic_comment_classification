"""
XAI Pipeline Module.

Master pipeline coordinating validation, explainer execution, positive/negative feature contribution split, and batch explanation.
"""

import logging
from typing import Dict, Any, List, Optional
from src.xai.base_explainer import BaseExplainer
from src.xai.explanation_factory import ExplanationFactory
from src.xai.config import XAIConfig
from src.xai.validator import Validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class XAIPipeline:
    """Master Explainable AI Pipeline."""

    def __init__(self, explainer: Optional[BaseExplainer] = None, config: Optional[XAIConfig] = None):
        """Initializes pipeline.

        Args:
            explainer: Instantiated BaseExplainer subclass or None to instantiate via factory.
            config: XAIConfig instance.
        """
        self.config = config or XAIConfig()
        self.explainer = explainer or ExplanationFactory.create(self.config)

    def explain_text(self, text: str, model: Any, target_label: Optional[str] = None) -> Dict[str, Any]:
        """Validates and explains a single text prediction.

        Args:
            text: Input string.
            model: Trained classifier model.
            target_label: Optional target label string.

        Returns:
            Dict containing prediction, confidence, feature_importance, positive_contributors, and negative_contributors.
        """
        val_text = Validator.validate_text(text)
        val_model = Validator.validate_model(model)
        return self.explainer.explain_instance(val_text, val_model, target_label=target_label)

    def explain_batch(self, texts: List[str], model: Any) -> List[Dict[str, Any]]:
        """Validates and explains a batch of text predictions.

        Args:
            texts: List of text strings.
            model: Trained classifier model.

        Returns:
            List of result dictionaries.
        """
        val_texts = [Validator.validate_text(t) for t in texts]
        val_model = Validator.validate_model(model)
        return self.explainer.explain_batch(val_texts, val_model)
