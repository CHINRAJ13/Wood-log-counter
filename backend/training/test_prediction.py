# training/test_prediction.py

from ultralytics import YOLO
from pathlib import Path
import cv2
import sys


def test_image(image_path: str):
    MODEL_PATH = "models/wood_logs.pt"

    # Check model exists
    if not Path(MODEL_PATH).exists():
        print("❌ No trained model found. Run train_model.py first.")
        return

    # Check image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return

    print(f"🔍 Testing model on: {image_path}")

    # Load model and run detection
    model  = YOLO(MODEL_PATH)
    result = model(image_path, conf=0.5)[0]
    count  = len(result.boxes)

    print(f"🪵 Logs detected: {count}")
    print()

    # Show each detection
    for i, box in enumerate(result.boxes):
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"  Log #{i+1}: confidence={conf:.1%}  "
              f"position=({int(x1)},{int(y1)}) → ({int(x2)},{int(y2)})")

    # Save result image with boxes drawn
    output_path = "test_result.jpg"
    result.save(filename=output_path)
    print()
    print(f"✅ Result image saved: {output_path}")
    print("   Open it to see bounding boxes on detected logs.")


if __name__ == "__main__":
    # Usage: python training/test_prediction.py path/to/image.jpg
    if len(sys.argv) < 2:
        # Default: use first test image from dataset
        test_images = list(Path("dataset/test/images").glob("*.jpg"))
        if test_images:
            test_image(str(test_images[0]))
        else:
            print("Usage: python training/test_prediction.py <image_path>")
    else:
        test_image(sys.argv[1])