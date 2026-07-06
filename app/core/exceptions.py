"""Custom exceptions for the application."""


class ApplicationError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, code: str | None = None):
        self.message = message
        self.code = code or "APPLICATION_ERROR"
        super().__init__(self.message)


class ImageProcessingError(ApplicationError):
    """Raised when image processing fails."""

    def __init__(self, message: str):
        super().__init__(message, code="IMAGE_PROCESSING_ERROR")


class UnsupportedFormatError(ApplicationError):
    """Raised when file format is not supported."""

    def __init__(self, format_name: str):
        message = f"Unsupported file format: {format_name}"
        super().__init__(message, code="UNSUPPORTED_FORMAT_ERROR")


class ModelLoadError(ApplicationError):
    """Raised when model loading fails."""

    def __init__(self, model_path: str, reason: str):
        message = f"Failed to load model from {model_path}: {reason}"
        super().__init__(message, code="MODEL_LOAD_ERROR")


class FileSizeExceededError(ApplicationError):
    """Raised when file size exceeds the limit."""

    def __init__(self, file_size: int, max_size: int):
        message = f"File size {file_size} bytes exceeds maximum allowed size of {max_size} bytes"
        super().__init__(message, code="FILE_SIZE_EXCEEDED_ERROR")


class ValidationError(ApplicationError):
    """Raised when validation fails."""

    def __init__(self, field: str, reason: str):
        message = f"Validation failed for '{field}': {reason}"
        super().__init__(message, code="VALIDATION_ERROR")
