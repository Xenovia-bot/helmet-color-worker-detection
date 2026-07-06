"""Application configuration settings."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Construction Safety Detection"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False)

    # API
    API_V1_PREFIX: str = "/api/v1"

    # CORS
    CORS_ALLOWED_ORIGINS: list[str] = ["*"]

    # Model
    MODEL_PATH: str = "best.pt"
    MODEL_CONFIDENCE_THRESHOLD: float = 0.5

    # Detection
    PROXIMITY_THRESHOLD: float = 0.8  # Proximity multiplier (0.0 - 1.0)
    MIN_EQUIPMENT_DETECTION_CONFIDENCE: float = 0.25  # Lower threshold for equipment detection
    MIN_PERSON_ANALYSIS_CONFIDENCE: float = 0.65  # Min person confidence to include in safety analysis

    # Required safety equipment (comma-separated list)
    REQUIRED_EQUIPMENT: str = "helmet,vest"  # Options: helmet, vest, blue helmet, red helmet, white helmet, yellow helmet

    # Processing
    MAX_FILE_SIZE_MB: int = 100
    MAX_VIDEO_SIZE_MB: int = 500
    SUPPORTED_FORMATS: list[str] = ["png", "jpg", "jpeg", "webp"]
    VIDEO_FRAME_SAMPLE_INTERVAL: float = 1.0  # seconds between sampled frames

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent

    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def model_absolute_path(self) -> Path:
        """Get absolute path to model file."""
        if Path(self.MODEL_PATH).is_absolute():
            return Path(self.MODEL_PATH)
        return self.BASE_DIR / self.MODEL_PATH

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings singleton.
    """
    return Settings()