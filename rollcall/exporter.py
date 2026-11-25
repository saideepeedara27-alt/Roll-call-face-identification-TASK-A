"""Export utilities for Annotated PNG group photos and Attendance CSV reports."""

import csv
from io import StringIO
import cv2
import numpy as np


# BGR colors matching UI status palette
_STATUS_COLORS = {
    "Confirmed": (110, 185, 16),   # Emerald green
    "Likely": (11, 158, 245),      # Amber orange
    "unknown": (94, 63, 244),      # Coral rose
}


def render_annotated_png(bgr_image: np.ndarray, faces: list[dict]) -> bytes:
    """Draw color-coded bounding boxes and readable labels onto the group photo."""
    canvas = bgr_image.copy()
    h, w = canvas.shape[:2]
    thickness = max(2, int(round(min(w, h) / 450)))
    font_scale = max(0.48, min(0.85, min(w, h) / 1100.0))

    for face in faces:
        bx, by, bw, bh = face["bbox"]
        status = face["status"]
        color = _STATUS_COLORS.get(status, (148, 163, 184))

        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), color, thickness)

        idx_num = face["face_index"] + 1
        if status == "Confirmed":
            label = f"#{idx_num} {face['display_name']}"
        elif status == "Likely":
            label = f"#{idx_num} {face['display_name']}?"
        else:
            label = f"#{idx_num} unknown"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1)
        )
        label_y = max(th + 8, by - 4)
        bg_x1 = max(0, bx)
        bg_y1 = max(0, label_y - th - 6)
        bg_x2 = min(w, bx + tw + 10)
        bg_y2 = min(h, label_y + baseline)

        cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(
            canvas,
            label,
            (bg_x1 + 5, bg_y2 - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Failed to encode annotated PNG.")
    return buf.tobytes()


def build_attendance_csv(faces: list[dict], absent_identities: list[dict]) -> str:
    """Generate a clean CSV report of detected faces and absent enrolled people."""
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(
        [
            "record_type",
            "face_index",
            "identity_name",
            "status",
            "similarity_score",
            "runner_up_name",
            "runner_up_margin",
            "face_size_px",
            "blur_laplacian",
            "decision_reason",
        ]
    )

    for face in faces:
        q = face.get("quality", {})
        writer.writerow(
            [
                "detected_face",
                face["face_index"] + 1,
                face["display_name"],
                face["status"],
                face["similarity"] if face["similarity"] is not None else "",
                face.get("runner_up_name") or "",
                face["runner_up_margin"] if face.get("runner_up_margin") is not None else "",
                f"{q.get('width_px', 0)}x{q.get('height_px', 0)}",
                q.get("blur_laplacian", ""),
                face.get("decision_reason", ""),
            ]
        )

    for person in absent_identities:
        writer.writerow(
            [
                "absent_enrolled",
                "",
                person["display_name"],
                "Absent",
                "",
                "",
                "",
                "",
                "",
                "Not detected in group photograph",
            ]
        )

    return out.getvalue()
