"""
Model Trainer Module.

Executes model fitting loops, validation tracking, and logging.
"""

import time
import logging
from typing import Any, Optional, Dict
from src.models.base_model import BaseModel
from src.models.exceptions import TrainingError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trainer orchestrating model training and logging."""

    def __init__(self, model: BaseModel):
        """Initializes trainer with BaseModel subclass instance.

        Args:
            model: Instantiated BaseModel subclass.
        """
        self.model = model

    def train(self, X_train: Any, y_train: Any, X_val: Optional[Any] = None, y_val: Optional[Any] = None) -> Dict[str, Any]:
        """Trains model and logs execution metrics.

        Args:
            X_train: Training feature matrix.
            y_train: Training target labels.
            X_val: Optional validation feature matrix.
            y_val: Optional validation target labels.

        Returns:
            Dict containing training wall-clock time and status.
        """
        logger.info(f"Starting training for Model '{self.model.name}' on {X_train.shape[0]:,} samples...")
        t0 = time.perf_counter()
        try:
            self.model.fit(X_train, y_train)
            train_time_sec = round(time.perf_counter() - t0, 4)
            logger.info(f"Training completed for '{self.model.name}' in {train_time_sec}s.")
            return {
                "model_name": self.model.name,
                "status": "success",
                "train_time_sec": train_time_sec,
            }
        except Exception as e:
            logger.error(f"Training failed for '{self.model.name}': {e}")
            raise TrainingError(f"Model training failed: {e}") from e
