"""
Model Utilities Module.

Provides seed initialization, tensor conversions, and helper functions.
"""

import os
import random
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Sets random seed across python, numpy, and torch/tensorflow for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set global random seed to {seed}")
