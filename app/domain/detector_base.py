"""Abstract base for detector implementations."""

from abc import ABC, abstractmethod
from typing import Protocol


class DetectorProtocol(Protocol):
    """Protocol defining the detector interface."""

    async def detect(self, image) -> list[dict]:
        """Detect objects in the image."""
        ...

    def is_available(self) -> bool:
        """Check if detector is available."""
        ...


class BaseDetector(ABC):
    """Abstract base class for object detectors."""

    @abstractmethod
    async def detect(self, image) -> list[dict]:
        """Detect objects in the image.

        Args:
            image: Image array to detect objects in.

        Returns:
            List of detection results.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if detector is available and ready.

        Returns:
            True if detector is available, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def supported_classes(self) -> list[str]:
        """List of object classes this detector can identify.

        Returns:
            List of class names.
        """
        pass