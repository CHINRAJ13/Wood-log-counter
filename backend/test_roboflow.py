"""
Quick test to verify Roboflow API key and model are working.
Run this BEFORE starting the FastAPI server.

Usage:
    python test_roboflow.py
    python test_roboflow.py path/to/image.jpg
"""
import requests
import base64
import sys
from pathlib import Path


# ── Paste your details here for quick test ────────────────────────────────────
API_KEY  = "NcZZtYzMxNxEOlCXkJMb"       # ← replace with your key
MODEL_ID = "my-first-project-lca2k"  # ← your Roboflow project ID
VERSION  = 1                          # ← your model version number
# ─────────────────────────────────────────────────────────────────────────────


def test_roboflow(image_path: str):
    print("🔍 Testing Roboflow API connection...")
    print(f"   Model  : {MODEL_ID} v{VERSION}")
    print(f"   Image  : {image_path}")
    print()

    # Read and encode image
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Call API
    url = (
        f"https://detect.roboflow.com/{MODEL_ID}/{VERSION}"
        f"?api_key={API_KEY}&confidence=50"
    )

    try:
        response = requests.post(
            url,
            data=img_base64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30
        )

        if response.status_code == 401:
            print("❌ Invalid API key!")
            print("   Get your key from: https://app.roboflow.com → Settings → API")
            return

        if response.status_code == 404:
            print("❌ Model not found!")
            print(f"   Check your MODEL_ID: '{MODEL_ID}' and VERSION: {VERSION}")
            return

        response.raise_for_status()
        result = response.json()

        count = len(result.get("predictions", []))
        print(f"✅ Roboflow API is working!")
        print(f"🪵 Logs detected: {count}")
        print()

        if count > 0:
            print("Detections:")
            for i, pred in enumerate(result["predictions"]):
                print(f"  Log #{i+1}: {pred['class']} "
                      f"confidence={pred['confidence']:.1%} "
                      f"at ({pred['x']:.0f}, {pred['y']:.0f})")
        else:
            print("⚠️  No logs detected. Try a clearer image or lower confidence threshold.")

        print()
        print("✅ Your API key and model are working correctly!")
        print("   Now update your .env file and start the server:")
        print("   uvicorn main:app --reload --port 8000")

    except requests.exceptions.ConnectionError:
        print("❌ No internet connection.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    if API_KEY == "your_api_key_here":
        print("⚠️  Please update API_KEY in this file before running.")
        print("   Get it from: https://app.roboflow.com → Settings → Roboflow API")
        sys.exit(1)

    # Use provided image or first test image
    if len(sys.argv) >= 2:
        img = sys.argv[1]
    else:
        test_imgs = list(Path("dataset/test/images").glob("*.jpg"))
        if test_imgs:
            img = str(test_imgs[0])
            print(f"No image provided — using: {img}")
        else:
            print("Usage: python test_roboflow.py <image_path>")
            print("   eg: python test_roboflow.py dataset/test/images/log1.jpg")
            sys.exit(1)

    test_roboflow(img)