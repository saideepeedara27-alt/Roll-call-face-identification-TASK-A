"""Export utilities for PhotoProof Annotated PNG group photos, CSV results, and summary sheet."""

import csv
from datetime import datetime, timezone
import html
from io import StringIO
import json
import cv2
import numpy as np


# BGR colors matching calm PhotoProof light-editorial palette
_STATUS_COLORS = {
    "High-confidence match": (86, 134, 48),   # Forest sage green
    "Needs review": (36, 134, 214),           # Warm amber ochre
    "Unknown": (96, 98, 122),                 # Neutral plum-slate
}


def _csv_safe(val: object) -> object:
    """Neutralize spreadsheet formula prefixes while keeping numbers intact."""
    if isinstance(val, str) and val[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return f"'{val}"
    return val


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _choose_non_overlapping_rect(
    bx: int,
    by: int,
    bw: int,
    bh: int,
    box_w: int,
    box_h: int,
    img_w: int,
    img_h: int,
    placed: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """Find a clean rectangle near the face box that avoids previously placed labels."""
    x_candidates = [bx, bx + bw - box_w, max(0, (bx + bw // 2) - box_w // 2), 0]
    y_candidates = [
        by - box_h - 4,
        by + bh + 4,
        by - 2 * (box_h + 6),
        by + bh + box_h + 10,
        by + 4,
    ]
    step = box_h + 6
    for k in range(2, 6):
        y_candidates.append(by - (k + 1) * step)
        y_candidates.append(by + bh + k * step)

    best_rect: tuple[int, int, int, int] | None = None
    for cy in y_candidates:
        for cx in x_candidates:
            rx1 = max(0, min(img_w - box_w, cx))
            ry1 = max(0, min(img_h - box_h, cy))
            cand = (rx1, ry1, min(img_w, rx1 + box_w), min(img_h, ry1 + box_h))
            if best_rect is None:
                best_rect = cand
            if not any(_rects_overlap(cand, prev) for prev in placed):
                return cand
    return best_rect if best_rect is not None else (0, 0, min(img_w, box_w), min(img_h, box_h))


def render_annotated_png(bgr_image: np.ndarray, faces: list[dict]) -> bytes:
    """Draw readable, non-overlapping PhotoProof labels onto the group photo."""
    canvas = bgr_image.copy()
    h, w = canvas.shape[:2]
    thickness = max(2, int(round(min(w, h) / 450)))
    font_scale = max(0.46, min(0.80, min(w, h) / 1150.0))
    font_thickness = max(1, thickness - 1)

    placed_labels: list[tuple[int, int, int, int]] = []
    ordered_faces = sorted(faces, key=lambda f: (f["bbox"][1], f["bbox"][0]))

    for face in ordered_faces:
        bx, by, bw, bh = face["bbox"]
        status = face["status"]
        color = _STATUS_COLORS.get(status, (120, 125, 138))
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), color, thickness)

    for face in ordered_faces:
        bx, by, bw, bh = face["bbox"]
        status = face["status"]
        is_manual = face.get("decision_source") == "manual" or bool(face.get("reviewed_at"))
        color = _STATUS_COLORS.get(status, (120, 125, 138))

        idx_num = face["face_index"] + 1
        manual_tag = " [Manual]" if is_manual else ""
        if status == "High-confidence match":
            label = f"Face {idx_num}: {face['display_name']}{manual_tag}"
        elif status == "Needs review":
            label = f"Face {idx_num}: {face['display_name']}?{manual_tag}"
        else:
            label = f"Face {idx_num}: Unknown{manual_tag}"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        box_w = min(w, tw + 12)
        box_h = min(h, th + baseline + 10)
        rect = _choose_non_overlapping_rect(bx, by, bw, bh, box_w, box_h, w, h, placed_labels)
        placed_labels.append(rect)
        rx1, ry1, rx2, ry2 = rect

        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), color, -1)
        if is_manual:
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1)
        cv2.putText(
            canvas,
            label,
            (rx1 + 6, ry2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Failed to encode annotated PNG.")
    return buf.tobytes()


def build_results_csv(faces: list[dict], _unused_absent: list[dict] | None = None) -> str:
    """Generate a formula-safe CSV report of detected faces and PhotoProof decisions."""
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(
        [
            "face_label",
            "face_index",
            "final_status",
            "final_name",
            "decision_source",
            "original_status",
            "original_name",
            "similarity_score",
            "runner_up_name",
            "runner_up_margin",
            "face_box_xywh",
            "face_size_px",
            "blur_laplacian",
            "reason",
        ]
    )

    for face in faces:
        q = face.get("quality", {})
        is_manual = face.get("decision_source") == "manual" or bool(face.get("reviewed_at"))
        source_label = "Manual decision" if is_manual else "Model suggestion"
        idx = face["face_index"] + 1
        writer.writerow(
            [
                f"Face {idx}",
                idx,
                _csv_safe(face["status"]),
                _csv_safe(face["display_name"]),
                source_label,
                _csv_safe(face.get("original_status", face["status"])),
                _csv_safe(face.get("original_display_name", face["display_name"])),
                face["similarity"] if face["similarity"] is not None else "",
                _csv_safe(face.get("runner_up_name") or ""),
                face["runner_up_margin"] if face["runner_up_margin"] is not None else "",
                json.dumps(face["bbox"]) if "bbox" in face else "",
                f"{q.get('width_px', 0)}x{q.get('height_px', 0)}",
                q.get("blur_laplacian", ""),
                _csv_safe(face.get("decision_reason", "")),
            ]
        )

    return out.getvalue()


build_attendance_csv = build_results_csv


def _esc(val: object) -> str:
    return html.escape("" if val is None else str(val), quote=True)


def build_print_attendance_html(
    session_payload: dict,
    is_demo: bool = False,
    include_thumbs: bool = False,
) -> str:
    """Render a strict-CSP, XSS-escaped, print-ready PhotoProof annotation summary."""
    faces = [f for f in session_payload.get("faces", []) if not f.get("removed")]
    summary = session_payload.get("summary", {})
    trail = session_payload.get("decision_trail", [])
    cal = session_payload.get("calibration") or {}

    confirmed_faces = [f for f in faces if f["status"] == "High-confidence match"]
    review_faces = [f for f in faces if f["status"] == "Needs review"]
    unknown_faces = [f for f in faces if f["status"] == "Unknown"]

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    t_low = f"{cal['final_t_low']:.4f}" if cal.get("final_t_low") is not None else "N/A"
    t_high = f"{cal['final_t_high']:.4f}" if cal.get("final_t_high") is not None else "N/A"

    demo_banner = (
        '<div class="demo-banner">SAMPLE DEMO DATA &mdash; Bundled Offline Practice Images</div>'
        if is_demo
        else ""
    )

    def _face_rows(items: list[dict]) -> str:
        if not items:
            cols = 6 if include_thumbs else 5
            return f'<tr><td colspan="{cols}">None</td></tr>'
        rows = []
        for f in items:
            sim = f"{f['similarity']:.4f}" if f.get("similarity") is not None else "N/A"
            margin = (
                f"+{f['runner_up_margin']:.4f}"
                if f.get("runner_up_margin") is not None
                else "N/A"
            )
            q = f.get("quality", {})
            size_txt = f"{q.get('width_px', 0)}&times;{q.get('height_px', 0)} px"
            is_manual = f.get("decision_source") == "manual" or bool(f.get("reviewed_at"))
            source_txt = "Manual decision" if is_manual else "Model suggestion"
            orig_txt = ""
            if is_manual:
                orig_txt = (
                    f" (model initially suggested: {_esc(f.get('original_status'))} &mdash; "
                    f"{_esc(f.get('original_display_name'))})"
                )
            thumb_td = (
                f'<td><img class="thumb-img" src="{_esc(f.get("crop_b64", ""))}" alt="Face {f["face_index"] + 1}" /></td>'
                if include_thumbs
                else ""
            )
            rows.append(
                "<tr>"
                f"{thumb_td}"
                f"<td>Face {f['face_index'] + 1}</td>"
                f"<td><strong>{_esc(f['display_name'])}</strong> ({_esc(source_txt)}){orig_txt}</td>"
                f"<td>{_esc(f['status'])}</td>"
                f"<td>{sim} / &Delta; {margin}</td>"
                f"<td>{size_txt} &mdash; {_esc(f.get('decision_reason', ''))}</td>"
                "</tr>"
            )
        return "".join(rows)

    thumb_th = "<th>Face</th>" if include_thumbs else ""

    trail_rows = (
        "".join(
            "<tr>"
            f"<td>{t['trail_id']}</td>"
            f"<td>{_esc(t['created_at'])}</td>"
            f"<td>{_esc(t['actor'])}</td>"
            f"<td>{_esc(t['action'])}</td>"
            f"<td>{'Face ' + str(t['face_index'] + 1) if t['face_index'] >= 0 else 'Session'}</td>"
            f"<td>{_esc(t.get('previous_status') or '—')} ({_esc(t.get('previous_display_name') or '—')})</td>"
            f"<td>{_esc(t['new_status'])} ({_esc(t['new_display_name'])})</td>"
            f"<td>{_esc(t.get('reason_note') or '')}</td>"
            "</tr>"
            for t in trail
        )
        or '<tr><td colspan="8">No edits recorded.</td></tr>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>PhotoProof Summary &mdash; {_esc(session_payload.get('filename', 'Session'))}</title>
  <link rel="stylesheet" href="/static/print.css" />
</head>
<body>
  <div class="print-toolbar">
    <span>PhotoProof Offline Summary Sheet</span>
    <button type="button" id="btn-print-now" class="btn-print">Print / Save as PDF</button>
  </div>

  {demo_banner}

  <header class="sheet-header">
    <h1>PhotoProof: Who Is in the Photo?</h1>
    <p class="meta-line">
      <strong>Group Photo:</strong> {_esc(session_payload.get('filename', ''))} &bull;
      <strong>Generated:</strong> {_esc(generated_at)} &bull;
      <strong>Model:</strong> {_esc(session_payload.get('model_version', ''))}
    </p>
    <p class="meta-line">
      <strong>References loaded:</strong> {summary.get('references_loaded', 0)} &bull;
      <strong>References rejected:</strong> {summary.get('references_rejected', 0)} &bull;
      <strong>Thresholds:</strong> T_low={t_low}, T_high={t_high} &bull;
      <strong>Note:</strong> Cosine similarity scores are not probabilities or accuracy percentages.
    </p>
  </header>

  <section class="summary-cards" aria-label="Demo readiness summary counts">
    <div class="summary-box"><span>References loaded</span><strong>{summary.get('references_loaded', 0)}</strong></div>
    <div class="summary-box"><span>References rejected</span><strong>{summary.get('references_rejected', 0)}</strong></div>
    <div class="summary-box"><span>Faces detected</span><strong>{summary.get('faces_detected', 0)}</strong></div>
    <div class="summary-box"><span>High-confidence matches</span><strong>{summary.get('high_confidence_matches', 0)}</strong></div>
    <div class="summary-box"><span>Needs-review faces</span><strong>{summary.get('needs_review_faces', 0)}</strong></div>
    <div class="summary-box"><span>Unknown faces</span><strong>{summary.get('unknown_faces', 0)}</strong></div>
  </section>

  <h2>High-Confidence Matches</h2>
  <table>
    <thead><tr>{thumb_th}<th>Face</th><th>Name &amp; Source</th><th>Status</th><th>Similarity / Margin</th><th>Details</th></tr></thead>
    <tbody>{_face_rows(confirmed_faces)}</tbody>
  </table>

  <h2>Needs Review</h2>
  <table>
    <thead><tr>{thumb_th}<th>Face</th><th>Candidate &amp; Source</th><th>Status</th><th>Similarity / Margin</th><th>Reason</th></tr></thead>
    <tbody>{_face_rows(review_faces)}</tbody>
  </table>

  <h2>Unknown Faces</h2>
  <table>
    <thead><tr>{thumb_th}<th>Face</th><th>Label &amp; Source</th><th>Status</th><th>Similarity / Margin</th><th>Reason</th></tr></thead>
    <tbody>{_face_rows(unknown_faces)}</tbody>
  </table>

  <section class="appendix" aria-label="Session Edits Appendix">
    <h2>Session Edits ({len(trail)} entries)</h2>
    <table>
      <thead>
        <tr><th>ID</th><th>Timestamp</th><th>Actor</th><th>Action</th><th>Face</th><th>Previous</th><th>New</th><th>Note</th></tr>
      </thead>
      <tbody>{trail_rows}</tbody>
    </table>
  </section>

  <script src="/static/print.js"></script>
</body>
</html>"""
