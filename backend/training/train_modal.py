# training/train_model.py

from ultralytics import YOLO
from pathlib import Path
import os

def train():
    print("🪵 Starting Wood Log Counter Training...")
    print("=" * 50)

    # ── Settings ──────────────────────────────────────
    DATA_YAML   = "dataset/data.yaml"   # path to your dataset config
    BASE_MODEL  = "yolov8n.pt"          # nano = smallest & fastest
    EPOCHS      = 100                 # training rounds
    IMG_SIZE    = 640                   # image input size
    BATCH_SIZE  = 16                    # images per batch (reduce to 8 if RAM issues)
    RUN_NAME    = "wood_log_v1"         # name for this training run

    # ── Check dataset exists ──────────────────────────
    if not Path(DATA_YAML).exists():
        print(f"❌ Dataset not found at: {DATA_YAML}")
        print("Make sure your dataset folder is inside the project root.")
        return

    # ── Load base YOLOv8 model ────────────────────────
    print(f"📦 Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    # ── Start training ────────────────────────────────
    print(f"🚀 Training started!")
    print(f"   Dataset : {DATA_YAML}")
    print(f"   Epochs  : {EPOCHS}")
    print(f"   Img Size: {IMG_SIZE}")
    print()

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=RUN_NAME,
        patience=20,        # stop early if no improvement for 20 epochs
        save=True,          # save best model
        save_period=10,     # save checkpoint every 10 epochs
        device="cpu",       # change to "cuda" if you have NVIDIA GPU
        workers=2,          # parallel data loading
        verbose=True
    )

    # ── Save best model to models/ folder ────────────
    best_model_path = f"runs/detect/{RUN_NAME}/weights/best.pt"

    if Path(best_model_path).exists():
        os.makedirs("models", exist_ok=True)
        import shutil
        shutil.copy(best_model_path, "models/wood_logs.pt")
        print()
        print("=" * 50)
        print("✅ Training Complete!")
        print(f"📁 Best model saved to: models/wood_logs.pt")
        print(f"📊 Check results at   : runs/detect/{RUN_NAME}/")
    else:
        print("⚠️ Training finished but best model not found.")

if __name__ == "__main__":
    train()