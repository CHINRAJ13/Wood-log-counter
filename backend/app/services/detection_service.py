import cv2
import numpy as np
import base64
import requests
import logging
from app.config import settings

logger = logging.getLogger(__name__)


class LogDetectionService:
    """
    Wood log detection using Roboflow hosted cloud model.
    No local model file needed — calls Roboflow API directly.
    """

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model. Falls back to base model if custom not found."""
        model_path = Path(settings.MODEL_PATH)

        if model_path.exists() and model_path.stat().st_size > 1000:
            # Load your trained model
            self.model = YOLO(str(model_path))
            self.model_loaded = True
            logger.info(f"✅ Loaded trained model from: {model_path}")
        else:
            # Fall back to base YOLOv8 nano (before training is done)
            logger.warning(
                "⚠️  Custom model not found or empty. "
                "Using base YOLOv8 model. Run training/train_model.py first."
            )
            self.model = YOLO("yolov8n.pt")
            self.model_loaded = False

    def _ensure_model_loaded(self):
        """Load model only when first needed."""
        if self.model is None:
            self._load_model()

    def detect(self, image: np.ndarray) -> dict:
        """
        Run detection on a numpy image array.

        Args:
            image: BGR numpy array (from OpenCV)

        Returns:
            dict with count, detections, annotated_image, image_shape
        """
        results = self.model(
            image,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            imgsz=settings.IMAGE_SIZE,
            verbose=False
        )

        detections = []
        for pred in result.get("predictions", []):
            cx = pred["x"]
            cy = pred["y"]
            w  = pred["width"]
            h  = pred["height"]

            detections.append({
                "id":         len(detections) + 1,
                "label":      pred["class"],
                "confidence": round(pred["confidence"], 3),
                "bbox": {
                    "x1": round(cx - w / 2),
                    "y1": round(cy - h / 2),
                    "x2": round(cx + w / 2),
                    "y2": round(cy + h / 2),
                    "cx": round(cx),
                    "cy": round(cy),
                }
            })

        annotated = self._draw_boxes(image.copy(), detections)

        return {
            "count":           len(detections),
            "detections":      detections,
            "annotated_image": annotated,
            "image_shape": {
                "width":  image.shape[1],
                "height": image.shape[0]
            },
            "model_loaded": True
        }

    def _draw_boxes(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Draw green bounding boxes and count banner on image."""
        for det in detections:
            b = det["bbox"]
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)

            # Label background + text
            label = f"#{det['id']} {det['confidence']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 0), -1)
            cv2.putText(image, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Total count banner at top
        count_text = f"Total Logs: {len(detections)}"
        cv2.rectangle(image, (8, 8), (260, 52), (0, 0, 0), -1)
        cv2.putText(image, text, (14, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        return image

    def get_model_info(self) -> dict:
        return {
            "type":       "roboflow_cloud",
            "model_id":   self.model_id,
            "version":    self.version,
            "confidence": settings.CONFIDENCE_THRESHOLD,
            "api_url":    f"https://detect.roboflow.com/{self.model_id}/{self.version}",
        }


# Singleton — loaded once when server starts
detection_service = LogDetectionService()
