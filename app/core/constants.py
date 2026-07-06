"""Constants used throughout the application."""

from enum import Enum


class ProcessingStatus(str, Enum):
    """Status values for API responses."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class EquipmentType(str, Enum):
    """Types of safety equipment to detect."""
    HELMET = "helmet"
    VEST = "vest"
    BLUE_HELMET = "blue helmet"
    RED_HELMET = "red helmet"
    WHITE_HELMET = "white helmet"
    YELLOW_HELMET = "yellow helmet"


# API constants
MAX_FILE_SIZE_MB: int = 100
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

# Model constants
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
MIN_CONFIDENCE_THRESHOLD: float = 0.1
MAX_CONFIDENCE_THRESHOLD: float = 0.99

# Supported image formats
SUPPORTED_FORMATS: tuple = ("png", "jpg", "jpeg", "webp")
SUPPORTED_MIME_TYPES: tuple = ("image/png", "image/jpeg", "image/webp")
VIDEO_MIME_TYPES: tuple = ("video/mp4",)
