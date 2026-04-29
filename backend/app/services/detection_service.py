import cv2
import numpy as np
import base64
import requests
import logging
from app.config import settings

logger = logging.getLogger(__name__)


class LogDetectionService:

    def __init__(self):
        self.api_key  = settings.ROBOFLOW_API_KEY
        self.model_id = settings.ROBOFLOW_MODEL_ID
        self.version  = settings.ROBOFLOW_VERSION
        self.conf     = int(settings.CONFIDENCE_THRESHOLD * 100)

        self.api_url = (
            f"https://detect.roboflow.com/"
            f"{self.model_id}/{self.version}"
            f"?api_key={self.api_key}"
            f"&confidence={self.conf}"
        )

        # Just log — don't crash on startup
        if not self.api_key:
            logger.warning("⚠️  ROBOFLOW_API_KEY is not set!")
        else:
            logger.info(f"✅ Roboflow service ready: {self.model_id} v{self.version}")

    def detect(self, image: np.ndarray) -> dict:
        # Check API key at request time
        if not self.api_key:
            raise Exception(
                "ROBOFLOW_API_KEY is not set. "
                "Add it in Render → Environment variables."
            )

        _, buffer  = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        try:
            response = requests.post(
                self.api_url,
                data=img_base64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise Exception("Roboflow API timed out.")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Roboflow API error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Roboflow API.")

        result = response.json()

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
        for det in detections:
            b = det["bbox"]
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"#{det['id']} {det['confidence']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 0), -1)
            cv2.putText(image, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
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
            "api_key_set": bool(self.api_key)
        }


detection_service = LogDetectionService()