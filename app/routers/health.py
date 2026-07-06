from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "construction-safety-detection"
    }

@router.get("/ready")
async def readiness_check():
    """Readiness probe endpoint."""
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "construction-safety-detection"
    }