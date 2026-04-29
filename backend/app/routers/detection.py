from fastapi import APIRouter, File, UploadFile, HTTPException
import cv2
import numpy as np
import base64

from app.services.detection_service import detection_service
from app.schemas.detection_schema import (
    DetectionResponse, LiveFrameRequest, LiveFrameResponse,
    CorrectionRequest, CorrectionResponse,
    Detection, BoundingBox, ImageShape
)
from app.config import settings

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def bytes_to_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return img


def base64_to_image(b64: str) -> np.ndarray:
    try:
        if "," in b64:
            b64 = b64.split(",")[1]
        data = base64.b64decode(b64)
        return bytes_to_image(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")


def image_to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def build_response(result: dict, include_image: bool = True) -> dict:
    detections = [
        Detection(
            id=d["id"], label=d["label"], confidence=d["confidence"],
            bbox=BoundingBox(**d["bbox"])
        )
        for d in result["detections"]
    ]
    count = result["count"]
    return {
        "success":     True,
        "count":       count,
        "detections":  detections,
        "image_shape": ImageShape(**result["image_shape"]),
        "annotated_image_base64": (
            image_to_base64(result["annotated_image"]) if include_image else None
        ),
        "model_trained": result.get("model_loaded", True),
        "message": f"Detected {count} wood log{'s' if count != 1 else ''}."
    }


# ── Endpoint 1: Upload Image File ─────────────────────────────────────────────

@router.post(
    "/detect/upload",
    response_model=DetectionResponse,
    summary="Upload image file to count wood logs"
)
async def detect_upload(
    file: UploadFile = File(..., description="JPG/PNG/WEBP image of wood logs")
):
    # Validate file extension
    ext = file.filename.split(".")[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '.{ext}' not allowed. Supported: {settings.ALLOWED_EXTENSIONS}"
        )

    # Validate file size
    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > settings.MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max: {settings.MAX_IMAGE_SIZE_MB}MB"
        )

    image  = bytes_to_image(raw)
    result = detection_service.detect(image)
    return DetectionResponse(**build_response(result, include_image=True))


# ── Endpoint 2: Base64 Image ──────────────────────────────────────────────────

@router.post(
    "/detect/base64",
    response_model=DetectionResponse,
    summary="Send base64 encoded image to count wood logs"
)
async def detect_base64(request: LiveFrameRequest):
    image  = base64_to_image(request.image_base64)
    result = detection_service.detect(image)
    return DetectionResponse(**build_response(result, include_image=True))


# ── Endpoint 3: Live Camera Frame ─────────────────────────────────────────────

@router.post(
    "/detect/live",
    response_model=LiveFrameResponse,
    summary="Process live camera frame — fast, no annotated image returned"
)
async def detect_live(request: LiveFrameRequest):
    image  = base64_to_image(request.image_base64)
    result = detection_service.detect(image)

    detections = [
        Detection(
            id=d["id"], label=d["label"], confidence=d["confidence"],
            bbox=BoundingBox(**d["bbox"])
        )
        for d in result["detections"]
    ]
    count = result["count"]

    return LiveFrameResponse(
        success=True,
        count=count,
        detections=detections,
        image_shape=ImageShape(**result["image_shape"]),
        message=f"{count} log{'s' if count != 1 else ''} in frame"
    )


# ── Endpoint 4: Manual Correction ────────────────────────────────────────────

@router.post(
    "/detect/correct",
    response_model=CorrectionResponse,
    summary="Submit manual correction to AI count"
)
async def correct_count(request: CorrectionRequest):
    diff      = request.corrected_count - request.original_count
    abs_diff  = abs(diff)
    direction = "added" if diff > 0 else "removed"

    return CorrectionResponse(
        success=True,
        original_count=request.original_count,
        corrected_count=request.corrected_count,
        difference=diff,
        message=(
            f"User {direction} {abs_diff} log{'s' if abs_diff != 1 else ''}. "
            f"Final count: {request.corrected_count}."
        ) if diff != 0 else "AI count was accurate. No correction needed."
    )


# ── Endpoint 5: Model Info ────────────────────────────────────────────────────

@router.get(
    "/model/info",
    summary="Get Roboflow model info and configuration"
)
def model_info():
    return detection_service.get_model_info()