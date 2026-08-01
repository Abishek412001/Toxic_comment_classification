"""
Enterprise Exception Hierarchy for OpenTrust AI Platform.
"""

from typing import Any, Dict, Optional


class OpenTrustException(Exception):
    """Base exception class for all OpenTrust AI platform errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class ValidationError(OpenTrustException):
    """Raised when request payload or data validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=422,
            details=details,
        )


class AuthenticationError(OpenTrustException):
    """Raised when authentication fails or token is invalid."""

    def __init__(self, message: str = "Invalid or missing credentials"):
        super().__init__(
            message=message,
            code="UNAUTHENTICATED",
            status_code=401,
        )


class AuthorizationError(OpenTrustException):
    """Raised when user or service lacks required permissions."""

    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            code="FORBIDDEN",
            status_code=403,
        )


class NotFoundError(OpenTrustException):
    """Raised when requested resource does not exist."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(
            message=message,
            code="NOT_FOUND",
            status_code=404,
        )


class RateLimitExceededError(OpenTrustException):
    """Raised when client exceeds API rate limit quotas."""

    def __init__(self, message: str = "API rate limit exceeded"):
        super().__init__(
            message=message,
            code="RATE_LIMIT_EXCEEDED",
            status_code=429,
        )


class InferenceError(OpenTrustException):
    """Raised when machine learning model inference fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="INFERENCE_FAILED",
            status_code=502,
            details=details,
        )


class ServiceUnavailableError(OpenTrustException):
    """Raised when downstream dependency or microservice is unreachable."""

    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__(
            message=message,
            code="SERVICE_UNAVAILABLE",
            status_code=503,
        )
