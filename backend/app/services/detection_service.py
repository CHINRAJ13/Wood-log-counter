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
        self.api_key  = settings.ROBOFLOW_API_KEY
        self.model_id = settings.ROBOFLOW_MODEL_ID
        self.version  = settings.ROBOFLOW_VERSION
        self.conf     = int(settings.CONFIDENCE_THRESHOLD * 100)

        # Roboflow inference API URL
        self.api_url = (
            f"https://detect.roboflow.com/"
            f"{self.model_id}/{self.version}"
            f"?api_key={self.api_key}"
            f"&confidence={self.conf}"
        )

        logger.info(f"✅ Roboflow service ready!")
        logger.info(f"   Model  : {self.model_id} v{self.version}")
        logger.info(f"   Conf   : {self.conf}%")

    def detect(self, image: np.ndarray) -> dict:
        """
        Send image to Roboflow API and return log detections.

        Args:
            image: BGR numpy array (OpenCV format)

        Returns:
            dict with count, detections, annotated_image, image_shape
        """
        # Encode image to JPEG base64
        _, buffer  = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        # POST to Roboflow inference API
        try:
            response = requests.post(
                self.api_url,
                data=img_base64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise Exception("Roboflow API request timed out. Check your internet connection.")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Roboflow API error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Roboflow API. Check your internet connection.")

        result = response.json()

        # Parse predictions into our format
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

        # Draw boxes on image
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

            # Green bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)

            # Label with background
            label = f"#{det['id']} {det['confidence']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 0), -1)
            cv2.putText(image, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Total count banner at top-left
        text = f"Total Logs: {len(detections)}"
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


# Single instance shared across all requests
detection_service = LogDetectionService()