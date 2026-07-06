"""Object detection service using YOLO model."""

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from app.config.settings import Settings
from app.core.constants import DEFAULT_CONFIDENCE_THRESHOLD
from app.core.exceptions import ModelLoadError
from app.domain.detector_base import BaseDetector


logger = logging.getLogger(__name__)


class YOLODetector(BaseDetector):
    """YOLO-based object detector for safety equipment detection."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float = DEFAULT_CONFIDENCE_THRESHOLD
    ):
        """Initialize the detector.

        Args:
            model_path: Path to the YOLO model file.
            confidence: Confidence threshold for detections.
        """
        self._settings = Settings()
        self._model_path = model_path or self._settings.MODEL_PATH
        self._confidence = confidence
        self._model: YOLO | None = None
        self._device = self._get_device()
        self._class_names: list[str] = []
        self._load_model()

    def _get_device(self) -> str:
        """Determine the best available device for inference.

        Returns:
            Device string ('cuda', 'mps', or 'cpu').
        """
        if torch.cuda.is_available():
            logger.info("Using CUDA GPU for inference")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Using Apple MPS for inference")
            return "mps"
        else:
            logger.info("Using CPU for inference")
            return "cpu"

    def _load_model(self) -> None:
        """Load the YOLO model from disk."""
        model_file = Path(self._model_path)

        if not model_file.exists():
            logger.warning(
                "Model file not found at %s, using default YOLO11n",
                self._model_path
            )
            self._load_default_model()
            return

        try:
            self._model = YOLO(str(model_file))
            self._model.to(self._device)
            self._class_names = list(self._model.names.values())
            logger.info(
                "Model loaded successfully from %s (classes: %s)",
                self._model_path,
                self._class_names
            )
        except Exception as e:
            logger.error("Failed to load model from %s: %s", self._model_path, str(e))
            self._load_default_model()

    def _load_default_model(self) -> None:
        """Load the default YOLO11n model as fallback."""
        try:
            self._model = YOLO("yolo11n.pt")
            self._model.to(self._device)
            self._class_names = list(self._model.names.values())
            logger.info(
                "Default YOLO11n model loaded successfully (classes: %s)",
                self._class_names
            )
        except Exception as e:
            logger.error("Failed to load default model: %s", str(e))
            raise ModelLoadError(
                self._model_path,
                f"Could not load custom model or fallback: {str(e)}"
            )

    async def detect(
        self,
        image: np.ndarray,
        equipment_confidence_threshold: float | None = None
    ) -> list[dict[str, Any]]:
        """Detect objects in the given image.

        Args:
            image: BGR image array from OpenCV.
            equipment_confidence_threshold: Lower confidence threshold for equipment.

        Returns:
            List of detection dictionaries with class, confidence, and bbox.
        """
        if self._model is None:
            raise ModelLoadError(self._model_path, "Model not initialized")

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            self._run_inference,
            image,
            equipment_confidence_threshold
        )

        return self._parse_results(results, equipment_confidence_threshold)

    def _run_inference(
        self,
        image: np.ndarray,
        equipment_confidence_threshold: float | None = None
    ) -> Any:
        """Run model inference synchronously.

        Args:
            image: BGR image array.
            equipment_confidence_threshold: Lower threshold for equipment detection.

        Returns:
            Raw inference results.
        """
        conf_threshold = equipment_confidence_threshold or self._confidence
        results = self._model.predict(
            image,
            conf=conf_threshold,
            verbose=False
        )
        return results[0] if results else None

    def _parse_results(
        self,
        results: Any,
        equipment_confidence_threshold: float | None = None
    ) -> list[dict[str, Any]]:
        """Parse YOLO results into a standardized format.

        Args:
            results: Raw YOLO inference results.
            equipment_confidence_threshold: Lower threshold for equipment.

        Returns:
            List of detection dictionaries.
        """
        if results is None:
            return []

        detections = []
        boxes = results.boxes
        equipment_classes = {"vest", "blue helmet", "red helmet", "white helmet", "yellow helmet"}
        min_equip_conf = equipment_confidence_threshold or self._confidence

        for box in boxes:
            conf = float(box.conf.item())
            class_id = int(box.cls.item())
            class_name = self._class_names[class_id] if self._class_names else "unknown"

            # Apply lower threshold for equipment classes
            if class_name.lower() in equipment_classes and conf < min_equip_conf:
                if equipment_confidence_threshold is not None:
                    continue
            elif conf < self._confidence:
                continue

            detections.append({
                "class": class_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox": box.xywhn[0].tolist()
            })

        return detections

    def detect_sync(
        self,
        image: np.ndarray,
        equipment_confidence_threshold: float | None = None
    ) -> list[dict]:
        """Synchronous detection — use inside executor threads."""
        results = self._run_inference(image, equipment_confidence_threshold)
        return self._parse_results(results, equipment_confidence_threshold)

    def is_available(self) -> bool:
        """Check if the detector is available.

        Returns:
            True if model is loaded, False otherwise.
        """
        return self._model is not None

    @property
    def supported_classes(self) -> list[str]:
        """Get list of supported object classes.

        Returns:
            List of class names from the model.
        """
        return self._class_names.copy()

    @property
    def device(self) -> str:
        """Get the device being used for inference."""
        return self._device

    @property
    def model_path(self) -> str:
        """Get the model path."""
        return self._model_path