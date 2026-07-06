"""Image processing orchestrator service."""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.core.constants import EquipmentType, ProcessingStatus
from app.core.exceptions import ImageProcessingError
from app.models.response import Detection, ProcessingResponse, ProcessingResult, Violation
from app.services.detector import YOLODetector
from app.services.safety_analyzer import SafetyAnalyzer, SafetyViolation
from app.utils.image_converter import ImageConverter


logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for image processing."""
    request_id: str
    processing_time: float
    image_width: int
    image_height: int
    device: str


class ImageProcessor:
    """Orchestrates the complete image processing pipeline."""

    def __init__(self):
        """Initialize the processor with all required components."""
        self._converter = ImageConverter()
        self._detector = YOLODetector()
        self._settings = Settings()

        # Parse required equipment from settings
        required_equipment = self._parse_required_equipment()

        self._analyzer = SafetyAnalyzer(
            required_equipment=required_equipment,
            proximity_threshold=self._settings.PROXIMITY_THRESHOLD
        )

    def _parse_required_equipment(self) -> list[EquipmentType]:
        """Parse required equipment from settings string.

        Returns:
            List of EquipmentType enum values.
        """
        equipment_map = {
            "helmet": EquipmentType.HELMET,
            "vest": EquipmentType.VEST,
            "blue helmet": EquipmentType.BLUE_HELMET,
            "red helmet": EquipmentType.RED_HELMET,
            "white helmet": EquipmentType.WHITE_HELMET,
            "yellow helmet": EquipmentType.YELLOW_HELMET,
        }

        result = []
        for item in self._settings.REQUIRED_EQUIPMENT.lower().split(","):
            item = item.strip()
            if item in equipment_map:
                result.append(equipment_map[item])
        return result

    async def process(
        self,
        image_data: bytes,
        filename: str
    ) -> ProcessingResponse:
        """Process an image through the complete detection pipeline.

        Args:
            image_data: Raw image bytes.
            filename: Original filename.

        Returns:
            ProcessingResponse with detection results.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            image = await self._convert_image(image_data, filename)
            detections = await self._run_detection(image)
            analysis_result = self._analyzer.analyze(detections, image.shape)
            processing_time = time.time() - start_time

            return self._build_response(
                request_id=request_id,
                analysis_result=analysis_result,
                processing_time=processing_time,
                raw_detections=detections,
            )

        except ImageProcessingError:
            raise
        except Exception as e:
            logger.error("Unexpected error during processing: %s", str(e))
            return self._build_error_response(
                request_id=request_id,
                error_message=str(e),
                processing_time=time.time() - start_time
            )

    async def _convert_image(
        self,
        image_data: bytes,
        filename: str
    ) -> np.ndarray:
        """Convert image data to processable format.

        Args:
            image_data: Raw image bytes.
            filename: Original filename.

        Returns:
            OpenCV BGR image array.

        Raises:
            ImageProcessingError: If conversion fails.
        """
        try:
            return await self._converter.convert_to_cv2(image_data, filename)
        except Exception as e:
            logger.error("Image conversion failed: %s", str(e))
            raise ImageProcessingError(f"Image conversion failed: {str(e)}")

    async def _run_detection(
        self,
        image: np.ndarray
    ) -> list[dict[str, Any]]:
        """Run object detection on the image.

        Args:
            image: OpenCV BGR image.

        Returns:
            List of detection dictionaries.
        """
        return await self._detector.detect(
            image,
            equipment_confidence_threshold=self._settings.MIN_EQUIPMENT_DETECTION_CONFIDENCE
        )

    def _build_response(
        self,
        request_id: str,
        analysis_result: SafetyAnalyzer,
        processing_time: float,
        raw_detections: list[dict[str, Any]] | None = None,
    ) -> ProcessingResponse:
        """Build the final response from analysis results.

        Args:
            request_id: Unique request identifier.
            analysis_result: Safety analysis results.
            processing_time: Total processing time in seconds.
            raw_detections: All raw model detections.

        Returns:
            ProcessingResponse with formatted results.
        """
        violations = [
            Violation(
                person_id=v.person_id,
                missing_equipment=v.missing_equipment,
                bbox=v.bbox,
                confidence=v.confidence
            )
            for v in analysis_result.violations
        ]

        all_detections = [
            Detection(class_name=d["class_name"], bbox=d["bbox"], confidence=d["confidence"])
            for d in (raw_detections or [])
        ]

        result = ProcessingResult(
            people_detected=analysis_result.people_count,
            violations=violations,
            detections=all_detections,
            processing_time=round(processing_time, 3)
        )

        status = ProcessingStatus.WARNING if analysis_result.has_violations else ProcessingStatus.SUCCESS

        return ProcessingResponse(
            request_id=request_id,
            status=status.value,
            results=result,
            message=None
        )

    def _build_error_response(
        self,
        request_id: str,
        error_message: str,
        processing_time: float
    ) -> ProcessingResponse:
        """Build an error response.

        Args:
            request_id: Unique request identifier.
            error_message: Error description.
            processing_time: Time spent before error.

        Returns:
            ProcessingResponse with error status.
        """
        return ProcessingResponse(
            request_id=request_id,
            status=ProcessingStatus.ERROR.value,
            results=None,
            message=error_message
        )

    @property
    def detector(self) -> YOLODetector:
        """Get the detector instance."""
        return self._detector

    @property
    def analyzer(self) -> SafetyAnalyzer:
        """Get the analyzer instance."""
        return self._analyzer