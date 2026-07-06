"""Image format conversion utilities."""

import asyncio
import base64
import io
import logging
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image

from app.config.settings import Settings
from app.core.constants import SUPPORTED_FORMATS
from app.core.exceptions import UnsupportedFormatError, ImageProcessingError


logger = logging.getLogger(__name__)


class ImageConverter:
    """Handles image format conversions and validation."""

    def __init__(self, settings: Settings | None = None):
        """Initialize the converter.

        Args:
            settings: Application settings. If None, loads default settings.
        """
        self._settings = settings or Settings()
        self._supported_formats = self._settings.SUPPORTED_FORMATS

    async def convert_to_cv2(
        self,
        image_data: bytes,
        filename: str
    ) -> np.ndarray:
        """Convert uploaded image data to OpenCV BGR format.

        Args:
            image_data: Raw image bytes.
            filename: Original filename for format detection.

        Returns:
            OpenCV image in BGR format.

        Raises:
            UnsupportedFormatError: If file format is not supported.
            ImageProcessingError: If image cannot be processed.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._convert_sync,
            image_data,
            filename
        )

    def _convert_sync(
        self,
        image_data: bytes,
        filename: str
    ) -> np.ndarray:
        """Convert image synchronously with validation.

        Args:
            image_data: Raw image bytes.
            filename: Original filename.

        Returns:
            OpenCV BGR image.

        Raises:
            UnsupportedFormatError: If format not supported.
            ImageProcessingError: If conversion fails.
        """
        file_ext = self._extract_extension(filename)
        self._validate_format(file_ext)

        try:
            pil_image = self._load_pil_image(image_data)
            pil_image = self._ensure_rgb_mode(pil_image)
            return self._pil_to_cv2(pil_image)
        except UnsupportedFormatError:
            raise
        except Exception as e:
            logger.error("Image conversion failed: %s", str(e))
            raise ImageProcessingError(f"Failed to convert image: {str(e)}")

    def _extract_extension(self, filename: str) -> str:
        """Extract file extension from filename.

        Args:
            filename: Name of the file.

        Returns:
            Lowercase extension without dot.
        """
        if '.' not in filename:
            return ''
        return filename.rsplit('.', 1)[-1].lower()

    def _validate_format(self, file_ext: str) -> None:
        """Validate that file format is supported.

        Args:
            file_ext: Lowercase file extension.

        Raises:
            UnsupportedFormatError: If format not supported.
        """
        if file_ext not in self._supported_formats:
            raise UnsupportedFormatError(file_ext)

    def _load_pil_image(self, image_data: bytes) -> Image.Image:
        """Load image data into PIL Image.

        Args:
            image_data: Raw image bytes.

        Returns:
            PIL Image object.

        Raises:
            ImageProcessingError: If image cannot be loaded.
        """
        try:
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise ImageProcessingError(f"Cannot decode image data: {str(e)}")

    def _ensure_rgb_mode(self, image: Image.Image) -> Image.Image:
        """Ensure image is in RGB mode.

        Args:
            image: PIL Image.

        Returns:
            RGB PIL Image.
        """
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image

    def _pil_to_cv2(self, image: Image.Image) -> np.ndarray:
        """Convert PIL RGB image to OpenCV BGR format.

        Args:
            image: PIL RGB Image.

        Returns:
            OpenCV BGR image array.
        """
        cv2_image = np.array(image)
        return cv2_image[:, :, ::-1].copy()

    def from_base64(self, base64_string: str) -> np.ndarray:
        """Convert base64 encoded image to OpenCV format.

        Args:
            base64_string: Base64 encoded image string (with or without prefix).

        Returns:
            OpenCV BGR image.

        Raises:
            ImageProcessingError: If conversion fails.
        """
        try:
            clean_base64 = self._strip_base64_prefix(base64_string)
            image_data = base64.b64decode(clean_base64)
            np_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if image is None:
                raise ImageProcessingError("Failed to decode image from base64")

            return image
        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ImageProcessingError(f"Base64 conversion failed: {str(e)}")

    def to_base64(self, image: np.ndarray, format: str = '.jpg') -> str:
        """Convert OpenCV image to base64 string.

        Args:
            image: OpenCV BGR image.
            format: Output format (.jpg, .png).

        Returns:
            Base64 encoded image string.
        """
        _, buffer = cv2.imencode(format, image)
        return base64.b64encode(buffer).decode('utf-8')

    def _strip_base64_prefix(self, base64_string: str) -> str:
        """Remove data URL prefix from base64 string.

        Args:
            base64_string: Base64 string possibly with prefix.

        Returns:
            Clean base64 string without prefix.
        """
        if ',' in base64_string:
            return base64_string.split(',', 1)[1]
        return base64_string

    @property
    def supported_formats(self) -> list[str]:
        """Get list of supported formats."""
        return self._supported_formats.copy()
