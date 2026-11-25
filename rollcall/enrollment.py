"""Enrollment Health Studio helpers: system-file filtering, thumbnails, and additive counts."""

import base64
from pathlib import Path
import cv2
import numpy as np

from rollcall.calibration import SFACE_FLOOR
from rollcall.image_utils import encode_jpeg_bytes

_SYSTEM_FILENAMES = {".ds_store", "thumbs.db", "desktop.ini"}


def is_system_file(raw_path: str) -> bool:
    """Return True for OS metadata and hidden files that should be ignored, not failed."""
    norm = raw_path.replace("\\", "/")
    parts = [p for p in norm.split("/") if p]
    if any(p == "__MACOSX" or p.startswith(".") for p in parts):
        return True
    name = Path(norm).name.lower()
    return name in _SYSTEM_FILENAMES


def render_thumbnail_b64(
    bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]] | None = None,
    active_idx: int | None = None,
    max_side: int = 128,
) -> str:
    """Render a compact base64 JPEG thumbnail, optionally highlighting detected face boxes."""
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / float(max(h, w, 1)))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    thumb = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if boxes:
        for idx, (bx, by, bw, bh) in enumerate(boxes):
            sx = int(round(bx * scale))
            sy = int(round(by * scale))
            sw = max(2, int(round(bw * scale)))
            sh = max(2, int(round(bh * scale)))
            is_active = active_idx is not None and idx == active_idx
            color = (235, 99, 37) if is_active else (148, 163, 184)  # BGR: blue vs slate
            thickness = 2 if is_active else 1
            cv2.rectangle(thumb, (sx, sy), (sx + sw, sy + sh), color, thickness)
            cv2.putText(
                thumb,
                f"#{idx + 1}",
                (sx + 2, max(11, sy - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    jpg = encode_jpeg_bytes(thumb, quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")


def check_extra_photo_consistency(
    existing_embeddings: list[np.ndarray],
    new_embedding: np.ndarray,
    threshold: float = SFACE_FLOOR,
) -> tuple[bool, float | None]:
    """Compare a new reference photo against existing embeddings of the same identity_id."""
    if not existing_embeddings:
        return True, None
    sims = [float(np.dot(prev, new_embedding)) for prev in existing_embeddings]
    best = round(max(sims), 4)
    return best >= threshold, best


def build_health_counts(file_records: list[dict]) -> dict:
    """Compute mutually exclusive enrollment counts that always sum to selected."""
    counts = {
        "selected": len(file_records),
        "enrolled": 0,
        "enrolled_with_warnings": 0,
        "needs_attention": 0,
        "failed": 0,
        "ignored": 0,
    }
    for rec in file_records:
        st = rec["status"]
        if st in counts:
            counts[st] += 1

    parts = [
        f"{counts['enrolled']} enrolled",
        f"{counts['enrolled_with_warnings']} enrolled with warnings",
    ]
    if counts["needs_attention"] > 0:
        parts.append(f"{counts['needs_attention']} needs attention")
    parts.append(f"{counts['failed']} failed")
    parts.append(f"{counts['ignored']} ignored")

    counts["headline"] = f"{counts['selected']} selected = " + ", ".join(parts)
    return counts
