"""Video processing service — samples frames and runs safety analysis."""

import asyncio
import logging
import tempfile
import time
import uuid

import cv2
import numpy as np

from app.config.settings import Settings
from app.core.constants import EquipmentType, ProcessingStatus
from app.core.exceptions import ImageProcessingError
from app.models.response import Detection, FrameViolation, VideoProcessingResponse, VideoProcessingResult, Violation
from app.services.detector import YOLODetector
from app.services.safety_analyzer import SafetyAnalyzer


logger = logging.getLogger(__name__)


class VideoProcessor:
    """Samples frames from an MP4 and runs safety analysis on each."""

    def __init__(self):
        self._settings = Settings()
        self._detector = YOLODetector()
        self._analyzer = SafetyAnalyzer(
            required_equipment=self._parse_required_equipment(),
            proximity_threshold=self._settings.PROXIMITY_THRESHOLD,
        )

    def _parse_required_equipment(self) -> list[EquipmentType]:
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

    async def process(self, video_data: bytes, filename: str) -> VideoProcessingResponse:
        """Process an MP4 video by sampling one frame per interval.

        Frames are analyzed sequentially one at a time to minimise memory and CPU load.
        Only frames that contain safety violations are included in the response.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            loop = asyncio.get_event_loop()
            violation_frames, frames_analyzed = await loop.run_in_executor(
                None, self._process_video_bytes, video_data
            )
            processing_time = time.time() - start_time

            total_violations = sum(len(f.violations) for f in violation_frames)
            status = ProcessingStatus.WARNING if total_violations else ProcessingStatus.SUCCESS

            return VideoProcessingResponse(
                request_id=request_id,
                status=status.value,
                results=VideoProcessingResult(
                    frames_analyzed=frames_analyzed,
                    total_violations=total_violations,
                    violation_frames=violation_frames,
                    processing_time=round(processing_time, 3),
                ),
                message=None,
            )

        except ImageProcessingError:
            raise
        except Exception as e:
            logger.error("Video processing failed: %s", str(e))
            return VideoProcessingResponse(
                request_id=request_id,
                status=ProcessingStatus.ERROR.value,
                results=None,
                message=str(e),
            )

    def _process_video_bytes(self, video_data: bytes) -> tuple[list[FrameViolation], int]:
        """Write video to temp file, iterate sampled frames, analyze each in place."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            tmp.write(video_data)
            tmp.flush()
            return self._iterate_and_analyze(tmp.name)

    def _iterate_and_analyze(self, path: str) -> tuple[list[FrameViolation], int]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ImageProcessingError("Cannot open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(1, int(fps * self._settings.VIDEO_FRAME_SAMPLE_INTERVAL))
        equip_conf = self._settings.MIN_EQUIPMENT_DETECTION_CONFIDENCE

        analyzed_frames: list[FrameViolation] = []
        frame_index = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % frame_interval == 0:
                    result = self._analyze_frame(frame, frame_index, fps, equip_conf)
                    analyzed_frames.append(result)

                frame_index += 1
        finally:
            cap.release()

        return analyzed_frames, len(analyzed_frames)

    def _analyze_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        fps: float,
        equip_conf: float,
    ) -> FrameViolation:
        detections = self._detector.detect_sync(frame, equip_conf)
        analysis = self._analyzer.analyze(detections, frame.shape)

        return FrameViolation(
            frame_index=frame_index,
            timestamp_seconds=round(frame_index / fps, 2),
            people_detected=analysis.people_count,
            violations=[
                Violation(
                    person_id=v.person_id,
                    missing_equipment=v.missing_equipment,
                    bbox=v.bbox,
                    confidence=v.confidence,
                )
                for v in analysis.violations
            ],
            detections=[
                Detection(class_name=d["class_name"], bbox=d["bbox"], confidence=d["confidence"])
                for d in detections
            ],
        )
