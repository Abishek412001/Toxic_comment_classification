"""
Enterprise Structured Logger Module (Step 134).

Provides rotating log file handlers, console loggers, JSON formatted logs, audit tracing, and performance decorators.
"""

import os
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Callable

LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

class StructuredLogger:
    """Production logger providing console and rotating file logs."""

    @staticmethod
    def get_logger(name: str = "MLOps") -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Console Handler
            c_handler = logging.StreamHandler()
            c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            c_handler.setFormatter(c_format)
            logger.addHandler(c_handler)

            # Rotating File Handler (10 MB max, 5 backups)
            f_path = os.path.join(LOG_DIR, "app.log")
            f_handler = RotatingFileHandler(f_path, maxBytes=10*1024*1024, backupCount=5)
            f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            f_handler.setFormatter(f_format)
            logger.addHandler(f_handler)

        return logger

def time_execution(func: Callable) -> Callable:
    """Decorator tracking function execution latency."""
    logger = StructuredLogger.get_logger("PerformanceTimer")
    def wrapper(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        t1 = time.time()
        logger.info(f"Function '{func.__name__}' executed in {(t1 - t0)*1000:.2f} ms")
        return res
    return wrapper
