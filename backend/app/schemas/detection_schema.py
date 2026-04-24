from pydantic import BaseModel, Field
from typing import List, Optional


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int


class Detection(BaseModel):
    id: int
    label: str
    confidence: float
    bbox: BoundingBox


class ImageShape(BaseModel):
    width: int
    height: int


# ── Upload / Base64 Response ──────────────────────────────────────────────────

class DetectionResponse(BaseModel):
    success: bool = True
    count: int
    detections: List[Detection]
    image_shape: ImageShape
    annotated_image_base64: Optional[str] = None
    model_trained: bool
    message: str


# ── Live Frame Request & Response ─────────────────────────────────────────────

class LiveFrameRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded camera frame")


class LiveFrameResponse(BaseModel):
    success: bool = True
    count: int
    detections: List[Detection]
    image_shape: ImageShape
    message: str


# ── Manual Correction ─────────────────────────────────────────────────────────

class CorrectionRequest(BaseModel):
    original_count: int
    corrected_count: int
    image_id: Optional[str] = None


class CorrectionResponse(BaseModel):
    success: bool = True
    original_count: int
    corrected_count: int
    difference: int
    message: str