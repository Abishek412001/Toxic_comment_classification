"""
Environment Detector Module.

Detects current runtime environment (development, testing, staging, production).
"""

import os

class Environment:
    """Environment detector class."""

    @staticmethod
    def get_env() -> str:
        """Returns active environment string."""
        return os.getenv("APP_ENV", "development").lower().strip()

    @staticmethod
    def is_production() -> bool:
        """Returns True if running in production mode."""
        return Environment.get_env() == "production"
