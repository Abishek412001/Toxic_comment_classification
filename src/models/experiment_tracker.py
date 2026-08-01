"""
Experiment Tracker Module.

Logs hyperparameters, metrics, and execution metadata to JSON files (MLflow style).
"""

import os
import json
import logging
from typing import Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Tracker class recording training experiment metadata."""

    def __init__(self, experiment_dir: str = "outputs/experiments"):
        """Initializes experiment tracker directory.

        Args:
            experiment_dir: Output log directory.
        """
        self.experiment_dir = experiment_dir
        os.makedirs(self.experiment_dir, exist_ok=True)

    def log_run(self, model_name: str, params: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        """Logs a single model training run.

        Args:
            model_name: Name of model.
            params: Hyperparameter dictionary.
            metrics: Evaluation metric dictionary.

        Returns:
            Log filepath.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{model_name}_{timestamp}"
        record = {
            "run_id": run_id,
            "model_name": model_name,
            "timestamp": timestamp,
            "parameters": params,
            "metrics": metrics,
        }

        filepath = os.path.join(self.experiment_dir, f"{run_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        logger.info(f"Logged experiment run to {filepath}")
        return filepath
