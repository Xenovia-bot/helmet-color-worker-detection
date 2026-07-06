"""Image processing API endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.constants import SUPPORTED_MIME_TYPES, VIDEO_MIME_TYPES
from app.core.exceptions import (
    ApplicationError,
    FileSizeExceededError,
    ImageProcessingError,
    UnsupportedFormatError,
)
from app.models.response import ProcessingResponse, VideoProcessingResponse
from app.services.image_processor import ImageProcessor
from app.services.video_processor import VideoProcessor


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/process-image",
    response_model=ProcessingResponse,
    responses={
        400: {"description": "Invalid request"},
        413: {"description": "File too large"},
        500: {"description": "Processing error"},
    },
    summary="Process Image for Safety Detection",
    description="Upload an image to detect construction workers and their safety equipment."
)
async def process_image(file: Annotated[UploadFile, File(description="Image file to process")]) -> ProcessingResponse:
    """Process an uploaded image to detect people and safety equipment.

    Args:
        file: Multipart form file upload containing the image.

    Returns:
        ProcessingResponse with detection results or error information.

    Raises:
        HTTPException: On validation errors or processing failures.
    """
    try:
        await _validate_file(file)

        contents = await file.read()
        logger.info("Processing image: %s (%d bytes)", file.filename, len(contents))

        processor = ImageProcessor()
        result = await processor.process(contents, file.filename)

        logger.info(
            "Image processed successfully: request_id=%s, people=%d, violations=%d",
            result.request_id,
            result.results.people_detected if result.results else 0,
            len(result.results.violations) if result.results else 0
        )

        return result

    except UnsupportedFormatError as e:
        logger.warning("Unsupported format: %s", file.filename)
        raise HTTPException(status_code=400, detail=str(e))

    except FileSizeExceededError as e:
        logger.warning("File too large: %d bytes", len(contents) if 'contents' in dir() else 0)
        raise HTTPException(status_code=413, detail=str(e))

    except ImageProcessingError as e:
        logger.error("Image processing failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

    except ApplicationError as e:
        logger.error("Application error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.exception("Unexpected error processing image")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post(
    "/process-video",
    response_model=VideoProcessingResponse,
    responses={
        400: {"description": "Invalid request"},
        413: {"description": "File too large"},
        500: {"description": "Processing error"},
    },
    summary="Process Video for Safety Detection",
    description="Upload an MP4 video to detect safety equipment violations. "
                "Frames are sampled at 1-second intervals to minimise device load. "
                "Only frames containing violations are returned.",
)
async def process_video(
    file: Annotated[UploadFile, File(description="MP4 video file to process")]
) -> VideoProcessingResponse:
    """Process an uploaded MP4 video to detect safety equipment violations."""
    try:
        await _validate_video_file(file)

        contents = await file.read()
        logger.info("Processing video: %s (%d bytes)", file.filename, len(contents))

        processor = VideoProcessor()
        result = await processor.process(contents, file.filename)

        logger.info(
            "Video processed: request_id=%s, frames=%d, violations=%d",
            result.request_id,
            result.results.frames_analyzed if result.results else 0,
            result.results.total_violations if result.results else 0,
        )
        return result

    except UnsupportedFormatError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileSizeExceededError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except ImageProcessingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ApplicationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error processing video")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


async def _validate_file(file: UploadFile) -> None:
    """Validate the uploaded file.

    Args:
        file: Uploaded file to validate.

    Raises:
        UnsupportedFormatError: If file type is not supported.
    """
    if not file.content_type:
        raise UnsupportedFormatError("unknown")

    if file.content_type not in SUPPORTED_MIME_TYPES:
        raise UnsupportedFormatError(file.content_type)


async def _validate_video_file(file: UploadFile) -> None:
    if not file.content_type:
        raise UnsupportedFormatError("unknown")
    if file.content_type not in VIDEO_MIME_TYPES:
        raise UnsupportedFormatError(file.content_type)
