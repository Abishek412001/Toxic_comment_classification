"""
Enterprise Health Probes for System Liveness, Readiness, and Diagnostics.
"""

import os
import psutil
from typing import Dict, Any
from opentrust_core.config import settings
from opentrust_core.schemas.health import HealthStatus, HealthComponent


class HealthChecker:
    """Evaluates liveness and readiness probes across system resources."""

    @staticmethod
    def get_liveness() -> HealthStatus:
        """Lightweight liveness probe checking process responsiveness."""
        return HealthStatus(
            status="UP",
            service=settings.PROJECT_NAME,
            version=settings.VERSION,
            environment=settings.ENVIRONMENT,
            components={
                "process": HealthComponent(status="UP", details={"pid": os.getpid()})
            },
        )

    @staticmethod
    def get_readiness() -> HealthStatus:
        """Deep readiness probe checking memory, CPU, and system storage status."""
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        cpu = psutil.cpu_percent()

        memory_status = "UP" if mem.percent < 95.0 else "DEGRADED"
        disk_status = "UP" if disk.percent < 95.0 else "DEGRADED"

        overall_status = "UP" if (memory_status == "UP" and disk_status == "UP") else "DEGRADED"

        return HealthStatus(
            status=overall_status,
            service=settings.PROJECT_NAME,
            version=settings.VERSION,
            environment=settings.ENVIRONMENT,
            components={
                "memory": HealthComponent(
                    status=memory_status,
                    details={"usage_percent": mem.percent, "available_mb": mem.available // (1024 * 1024)},
                ),
                "disk": HealthComponent(
                    status=disk_status,
                    details={"usage_percent": disk.percent, "free_gb": disk.free // (1024 * 1024 * 1024)},
                ),
                "cpu": HealthComponent(
                    status="UP",
                    details={"usage_percent": cpu, "core_count": psutil.cpu_count()},
                ),
            },
        )
