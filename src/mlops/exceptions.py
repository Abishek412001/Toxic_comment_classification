"""
Custom Exception Hierarchy for MLOps & Deployment Framework (Phase 12).
"""


class MLOpsError(Exception):
    """Base exception class for all MLOps framework errors."""
    pass


class ConfigurationError(MLOpsError):
    """Raised when environment variables or YAML configuration files fail validation."""
    pass


class ModelLoadError(MLOpsError):
    """Raised when model weights, checkpoints, or serialization files fail to load."""
    pass


class HealthCheckError(MLOpsError):
    """Raised when system health probes or telemetry checks fail."""
    pass
