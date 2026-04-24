# Run this in terminal to verify setup
# python check_dataset.py

from pathlib import Path
import yaml

# Load and print yaml
with open("dataset/data.yaml", "r") as f:
    config = yaml.safe_load(f)
    print("📄 data.yaml contents:")
    print(config)
    print()

# Check all folders exist
folders = [
    "dataset/train/images",
    "dataset/train/labels",
    "dataset/valid/images",
    "dataset/valid/labels",
    "dataset/test/images",
    "dataset/test/labels",
]

print("📁 Checking folders:")
all_good = True
for folder in folders:
    exists = Path(folder).exists()
    count  = len(list(Path(folder).glob("*"))) if exists else 0
    status = "✅" if exists else "❌"
    print(f"  {status} {folder}  ({count} files)")
    if not exists:
        all_good = False

print()
if all_good:
    print("✅ All folders found! Ready to train.")
else:
    print("❌ Some folders missing. Check your dataset path.")