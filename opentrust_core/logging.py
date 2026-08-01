"""
Structured JSON & OpenTelemetry-Compatible Logger for OpenTrust AI.
"""

import sys
import json
import logging
from typing import Any, Dict
from opentrust_core.config import settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter producing structured log outputs for ELK / Loki ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "environment": settings.ENVIRONMENT,
            "module": record.module,
            "line": record.lineno,
        }

        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def get_logger(name: str = "opentrust") -> logging.Logger:
    """Returns configured logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if settings.LOG_FORMAT.lower() == "json":
            handler.setFormatter(JSONFormatter())
        else:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
