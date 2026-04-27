import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class LogDetectionService:
    """
    Core AI detection service.
    Loads YOLOv8 model and runs wood log detection on images.
    """

    def __init__(self):
        self.model = None
        self.model_loaded = False

    def _load_model(self):
        """Load YOLOv8 model. Falls back to base model if custom not found."""
        model_path = Path(settings.MODEL_PATH)

        if model_path.exists() and model_path.stat().st_size > 1000:
            self.model = YOLO(str(model_path))
            self.model_loaded = True
            logger.info(f"✅ Loaded trained model from: {model_path}")
        else:
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
        self._ensure_model_loaded()
        results = self.model(
            image,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            imgsz=settings.IMAGE_SIZE,
            verbose=False
        )

        detections = []
        result = results[0]

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = self.model.names[class_id]

            detections.append({
                "id": len(detections) + 1,
                "label": label,
                "confidence": round(confidence, 3),
                "bbox": {
                    "x1": round(x1),
                    "y1": round(y1),
                    "x2": round(x2),
                    "y2": round(y2),
                    "cx": round((x1 + x2) / 2),
                    "cy": round((y1 + y2) / 2),
                }
            })

        annotated = self._draw_boxes(image.copy(), detections)

        return {
            "count": len(detections),
            "detections": detections,
            "annotated_image": annotated,
            "image_shape": {
                "width": image.shape[1],
                "height": image.shape[0]
            },
            "model_loaded": self.model_loaded
        }

    def _draw_boxes(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Draw green bounding boxes and log count on image."""
        for det in detections:
            b = det["bbox"]
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)

            label = f"#{det['id']} {det['confidence']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 0), -1)
            cv2.putText(image, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        count_text = f"Total Logs: {len(detections)}"
        cv2.rectangle(image, (8, 8), (260, 52), (0, 0, 0), -1)
        cv2.putText(image, count_text, (14, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        return image

    def get_model_info(self) -> dict:
        return {
            "model_path": settings.MODEL_PATH,
            "model_trained": self.model_loaded,
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
            "iou_threshold": settings.IOU_THRESHOLD,
            "image_size": settings.IMAGE_SIZE,
        }


# Singleton
detection_service = LogDetectionService()
