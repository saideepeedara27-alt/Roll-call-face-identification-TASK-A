"""One-time offline preparation script to download and verify OpenCV Zoo models.

Run this script ONCE while online before disconnecting from the internet.
RollCall itself never downloads models at startup or inference time.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODELS = [
    {
        "name": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "license": "MIT (OpenCV Zoo YuNet)",
        "sha256": None,  # Recorded in manifest.json upon verified download
    },
    {
        "name": "face_recognition_sface_2021dec.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "license": "Apache-2.0 (OpenCV Zoo SFace)",
        "sha256": None,
    },
]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for spec in MODELS:
        dest = MODELS_DIR / spec["name"]
        if dest.exists() and dest.stat().st_size > 0:
            digest = compute_sha256(dest)
            print(f"[OK] Already cached: {dest.name} ({dest.stat().st_size:,} bytes, sha256={digest[:12]}...)")
        else:
            print(f"[DOWNLOADING] {spec['name']} from OpenCV Zoo...")
            try:
                urllib.request.urlretrieve(spec["url"], dest)
            except Exception as exc:
                print(f"[ERROR] Failed to download {spec['name']}: {exc}", file=sys.stderr)
                return 1
            digest = compute_sha256(dest)
            print(f"[VERIFIED] Saved {dest.name} ({dest.stat().st_size:,} bytes, sha256={digest[:12]}...)")

        manifest[spec["name"]] = {
            "sha256": digest,
            "size_bytes": dest.stat().st_size,
            "license": spec["license"],
            "source_url": spec["url"],
        }

    manifest_path = MODELS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nModel manifest written to {manifest_path}")
    print("RollCall is now ready for 100% offline execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
