# training/evaluate_model.py

from ultralytics import YOLO
from pathlib import Path


def evaluate():
    MODEL_PATH = "models/wood_logs.pt"
    DATA_YAML  = "dataset/data.yaml"

    if not Path(MODEL_PATH).exists():
        print("❌ No trained model found. Run train_model.py first.")
        return

    print("📊 Evaluating model on test dataset...")
    model = YOLO(MODEL_PATH)

    metrics = model.val(data=DATA_YAML, split="test")

    print()
    print("=" * 50)
    print("📊 Model Evaluation Results")
    print("=" * 50)
    print(f"mAP50       : {metrics.box.map50:.3f}   ← aim for 0.70+")
    print(f"mAP50-95    : {metrics.box.map:.3f}")
    print(f"Precision   : {metrics.box.mp:.3f}")
    print(f"Recall      : {metrics.box.mr:.3f}")
    print()

    if metrics.box.map50 >= 0.80:
        print("✅ Excellent! Model is ready for production.")
    elif metrics.box.map50 >= 0.65:
        print("⚠️ Good. Consider adding more images for better accuracy.")
    else:
        print("❌ Low accuracy. Add more images and retrain.")


if __name__ == "__main__":
    evaluate()