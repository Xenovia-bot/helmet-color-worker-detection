"""Response models for API endpoints."""

from pydantic import BaseModel, Field
from typing import Optional

from app.core.constants import ProcessingStatus


class Detection(BaseModel):
    """A single object detection from the model."""
    class_name: str = Field(..., description="Detected object class")
    bbox: list[float] = Field(..., description="Bounding box [cx, cy, w, h] normalized 0-1")
    confidence: float = Field(..., ge=0.0, le=1.0)


class Violation(BaseModel):
    """Represents a detected safety violation."""
    person_id: int = Field(..., description="Unique identifier for the detected person")
    missing_equipment: list[str] = Field(..., description="List of missing safety equipment")
    bbox: list[float] = Field(..., description="Bounding box [cx, cy, w, h] normalized 0-1")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")


class ProcessingResult(BaseModel):
    """Results of image processing."""
    people_detected: int = Field(0, description="Number of people detected")
    violations: list[Violation] = Field(default_factory=list, description="List of safety violations")
    detections: list[Detection] = Field(default_factory=list, description="All detected objects")
    processing_time: float = Field(0.0, ge=0.0, description="Processing time in seconds")


class ProcessingResponse(BaseModel):
    """Complete API response model."""
    request_id: str = Field(..., description="Unique request identifier")
    status: str = Field(..., description="Response status (success, warning, error)")
    results: Optional[ProcessingResult] = Field(None, description="Processing results if successful")
    message: Optional[str] = Field(None, description="Error message if status is error")

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "warning",
                "results": {
                    "people_detected": 3,
                    "violations": [
                        {
                            "person_id": 1,
                            "missing_equipment": ["helmet"],
                            "bbox": [100.0, 200.0, 150.0, 300.0],
                            "confidence": 0.95
                        }
                    ],
                    "processing_time": 0.452
                },
                "message": None
            }
        }


class FrameViolation(BaseModel):
    """Safety violations detected in a single video frame."""
    frame_index: int = Field(..., description="Frame number in the video")
    timestamp_seconds: float = Field(..., description="Timestamp of the frame in seconds")
    people_detected: int = Field(...)
    violations: list[Violation] = Field(default_factory=list)
    detections: list[Detection] = Field(default_factory=list, description="All detected objects in this frame")


class VideoProcessingResult(BaseModel):
    """Results of video processing."""
    frames_analyzed: int = Field(0, description="Number of frames sampled and analyzed")
    total_violations: int = Field(0, description="Total violation instances across all frames")
    violation_frames: list[FrameViolation] = Field(default_factory=list, description="Only frames with violations")
    processing_time: float = Field(0.0, ge=0.0)


class VideoProcessingResponse(BaseModel):
    """Complete API response model for video processing."""
    request_id: str = Field(..., description="Unique request identifier")
    status: str = Field(..., description="Response status (success, warning, error)")
    results: Optional[VideoProcessingResult] = Field(None)
    message: Optional[str] = Field(None)
