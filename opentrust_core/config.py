"""
Multi-Environment Pydantic v2 Settings Manager for OpenTrust AI.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core platform configuration loaded from environment variables and .env files."""

    # Platform Info
    PROJECT_NAME: str = "OpenTrust AI"
    ENVIRONMENT: str = "development"  # development, testing, staging, production
    DEBUG: bool = True
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # Gateway Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # Security & CORS
    SECRET_KEY: str = "super-secret-key-change-in-production-opentrust-ai-2026"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    ALLOWED_HOSTS: List[str] = ["*"]
    CORS_ORIGINS: List[str] = ["*"]

    # Database Settings (PostgreSQL Async)
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "opentrust"
    POSTGRES_PASSWORD: str = "opentrust_pass"
    POSTGRES_DB: str = "opentrust_db"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: Optional[str] = None

    # Cache & Queue (Redis / Valkey)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_URL: Optional[str] = None

    # Model Inference Settings
    DEFAULT_MODEL: str = "distilbert"
    MODEL_DIR: str = "models"
    ENABLE_GPU: bool = False
    MAX_BATCH_SIZE: int = 64

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or console

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_database_url(self) -> str:
        """Returns PostgreSQL async connection string."""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    def get_redis_url(self) -> str:
        """Returns Redis connection string."""
        if self.REDIS_URL:
            return self.REDIS_URL
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/0"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"


settings = Settings()
