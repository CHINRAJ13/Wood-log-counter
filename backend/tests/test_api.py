"""
Test all FastAPI endpoints.
Make sure server is running first:
  uvicorn main:app --reload --port 8000
"""
import requests
import base64
import sys
from pathlib import Path


BASE_URL = "http://localhost:8000"


def test_health():
    print("1️⃣  Testing health endpoint...")
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    print(f"   ✅ {r.json()}")


def test_model_info():
    print("2️⃣  Testing model info...")
    r = requests.get(f"{BASE_URL}/api/v1/model/info")
    assert r.status_code == 200
    info = r.json()
    print(f"   ✅ Model trained: {info['model_trained']}")
    print(f"   ✅ Confidence: {info['confidence_threshold']}")


def test_upload(image_path: str):
    print(f"3️⃣  Testing image upload: {image_path}")
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/api/v1/detect/upload",
            files={"file": (Path(image_path).name, f, "image/jpeg")}
        )
    assert r.status_code == 200
    data = r.json()
    print(f"   ✅ Logs detected : {data['count']}")
    print(f"   ✅ Model trained : {data['model_trained']}")
    print(f"   ✅ Message       : {data['message']}")


def test_base64(image_path: str):
    print(f"4️⃣  Testing base64 endpoint: {image_path}")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    r = requests.post(
        f"{BASE_URL}/api/v1/detect/base64",
        json={"image_base64": b64}
    )
    assert r.status_code == 200
    data = r.json()
    print(f"   ✅ Logs detected: {data['count']}")


def test_live(image_path: str):
    print(f"5️⃣  Testing live frame endpoint: {image_path}")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    r = requests.post(
        f"{BASE_URL}/api/v1/detect/live",
        json={"image_base64": b64}
    )
    assert r.status_code == 200
    data = r.json()
    print(f"   ✅ Live count: {data['count']}")


def test_correction():
    print("6️⃣  Testing manual correction...")
    r = requests.post(
        f"{BASE_URL}/api/v1/detect/correct",
        json={"original_count": 10, "corrected_count": 12}
    )
    assert r.status_code == 200
    data = r.json()
    print(f"   ✅ {data['message']}")


if __name__ == "__main__":
    # Find a test image
    test_imgs = list(Path("dataset/test/images").glob("*.jpg"))
    if not test_imgs:
        print("❌ No test images found in dataset/test/images/")
        sys.exit(1)

    img = str(test_imgs[0])
    print(f"🪵 Using test image: {img}")
    print()

    try:
        test_health()
        test_model_info()
        test_upload(img)
        test_base64(img)
        test_live(img)
        test_correction()
        print()
        print("=" * 40)
        print("✅ All tests passed!")
    except requests.exceptions.ConnectionError:
        print()
        print("❌ Cannot connect to server.")
        print("   Start it first: uvicorn main:app --reload --port 8000")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")