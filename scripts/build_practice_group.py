"""Builds a synthetic practice group photograph in data/practice/group/ for Demo Mode & UI testing.

Note: Composed strictly from the already-generated synthetic reference portraits
for UI, bounding-box, quality-gate, absent-list, and export testing only.
Never use this composite to claim real-world recognition accuracy.
"""

from pathlib import Path
import cv2
import numpy as np

from rollcall.config import PRACTICE_DATA_DIR
from rollcall.engine import OpenCVYuNetSFaceEngine
from rollcall.image_utils import load_image_from_bytes


def main() -> None:
    ref_dir = PRACTICE_DATA_DIR / "reference"
    group_dir = PRACTICE_DATA_DIR / "group"
    group_dir.mkdir(parents=True, exist_ok=True)

    engine = OpenCVYuNetSFaceEngine()

    # Soft office wall gradient background (1400 x 820)
    h, w = 820, 1400
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        val = int(218 - (y / h) * 38)
        canvas[y, :, :] = (val + 6, val + 2, val - 4)  # subtle cool-neutral BGR

    # Draw subtle conference-room frame lines in background
    cv2.rectangle(canvas, (70, 60), (1330, 520), (195, 198, 204), 2)
    cv2.line(canvas, (0, 620), (w, 620), (165, 170, 178), 3)

    placements = [
        ("Priya Sharma.jpg", (90, 210), 340),
        ("Rahul Verma.jpg", (480, 210), 340),
        ("Mei Lin.jpg", (870, 220), 330),
        # Intentionally small back-row placement to exercise small-face / quality-gate handling
        ("David Brooks.jpg", (1260, 175), 52),
    ]

    for fname, (px, py), target_w in placements:
        p = ref_dir / fname
        if not p.is_file():
            continue
        loaded = load_image_from_bytes(p.read_bytes(), p.name)
        faces = engine.detect_faces(loaded.bgr)
        if not faces:
            continue
        # Pick largest face
        main_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        bx, by, bw, bh = main_face.bbox
        img_h, img_w = loaded.bgr.shape[:2]

        # Generous upper-body/portrait crop around face
        x1 = max(0, bx - int(bw * 0.75))
        y1 = max(0, by - int(bh * 0.55))
        x2 = min(img_w, bx + bw + int(bw * 0.75))
        y2 = min(img_h, by + bh + int(bh * 1.15))
        crop = loaded.bgr[y1:y2, x1:x2]

        scale = target_w / float(crop.shape[1])
        target_h = max(1, int(round(crop.shape[0] * scale)))
        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

        rh, rw = resized.shape[:2]
        paste_x = max(0, min(w - rw, px))
        paste_y = max(0, min(h - rh, py))
        canvas[paste_y : paste_y + rh, paste_x : paste_x + rw] = resized

    out_path = group_dir / "demo_office_group.jpg"
    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"Created {out_path} ({w}x{h})")


if __name__ == "__main__":
    main()
