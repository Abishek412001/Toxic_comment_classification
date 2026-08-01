"""
Settings Loader Module (Step 132).

Loads YAML config and environment variables with fallbacks.
"""

import os
import yaml
from typing import Dict, Any
from src.mlops.environment import Environment

class Settings:
    """Settings loader class."""

    def __init__(self, config_path: str = "config.yaml"):
        self.env = Environment.get_env()
        self.config_data = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
                self.config_data = raw_config.get(self.env, {})

    def get(self, key: str, default: Any = None) -> Any:
        env_val = os.getenv(key.upper())
        if env_val is not None:
            return env_val
        return self.config_data.get(key, default)
