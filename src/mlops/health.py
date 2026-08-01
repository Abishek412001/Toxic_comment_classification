"""
Health Check & Telemetry Module (Step 138).

Executes liveness probes, readiness probes, startup diagnostics, and resource telemetry.
"""

import os
import psutil
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class HealthChecker:
    """Health checking suite for application liveness and readiness probes."""

    @staticmethod
    def check_liveness() -> Dict[str, Any]:
        """Returns liveness status."""
        return {"status": "UP", "probe": "liveness"}

    @staticmethod
    def check_readiness() -> Dict[str, Any]:
        """Returns readiness probe including memory and disk space."""
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        is_ready = mem.percent < 95.0 and disk.percent < 95.0

        return {
            "status": "UP" if is_ready else "DOWN",
            "probe": "readiness",
            "memory_usage_percent": mem.percent,
            "disk_usage_percent": disk.percent,
            "cpu_percent": psutil.cpu_percent(),
        }
