"""Export utilities for Annotated PNG group photos and Attendance CSV reports."""

import csv
from datetime import datetime, timezone
import html
from io import StringIO
import json
import cv2
import numpy as np


# BGR colors matching UI status palette
_STATUS_COLORS = {
    "Confirmed": (110, 185, 16),              # Emerald green
    "Confirmed by reviewer": (136, 148, 13),  # Teal-emerald
    "Likely": (11, 158, 245),                 # Amber orange
    "unknown": (94, 63, 244),                 # Coral rose
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
        elif status == "Confirmed by reviewer":
            label = f"#{idx_num} {face['display_name']} [R]"
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
            "original_status",
            "original_identity",
            "similarity_score",
            "runner_up_name",
            "runner_up_margin",
            "face_box_xywh",
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
                face.get("original_status", face["status"]),
                face.get("original_display_name", face["display_name"]),
                face["similarity"] if face["similarity"] is not None else "",
                face.get("runner_up_name") or "",
                face["runner_up_margin"] if face.get("runner_up_margin") is not None else "",
                json.dumps(face["bbox"]) if "bbox" in face else "",
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
                "",
                "",
                "",
                "Not confirmed in group photograph",
            ]
        )

    return out.getvalue()


def _esc(val: object) -> str:
    return html.escape("" if val is None else str(val), quote=True)


def build_print_attendance_html(
    session_payload: dict,
    is_demo: bool = False,
    include_thumbs: bool = False,
) -> str:
    """Render a strict-CSP, XSS-escaped, print-ready HTML attendance report."""
    faces = [f for f in session_payload.get("faces", []) if not f.get("removed")]
    summary = session_payload.get("summary", {})
    absent = session_payload.get("absent_identities", [])
    not_enrolled = session_payload.get("not_enrolled_identities", [])
    trail = session_payload.get("decision_trail", [])
    cal = session_payload.get("calibration") or {}

    auto_confirmed = [f for f in faces if f["status"] == "Confirmed"]
    reviewer_confirmed = [f for f in faces if f["status"] == "Confirmed by reviewer"]
    likely_faces = [f for f in faces if f["status"] == "Likely"]
    unknown_faces = [f for f in faces if f["status"] == "unknown"]

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    t_low = f"{cal['final_t_low']:.4f}" if cal.get("final_t_low") is not None else "N/A"
    t_high = f"{cal['final_t_high']:.4f}" if cal.get("final_t_high") is not None else "N/A"

    demo_banner = (
        '<div class="demo-banner">DEMO DATA &mdash; Bundled Practice Dataset Only</div>'
        if is_demo
        else ""
    )

    def _face_rows(items: list[dict], show_orig: bool = False) -> str:
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
            orig_txt = ""
            if show_orig:
                orig_txt = f" (orig: {_esc(f.get('original_status'))} &mdash; {_esc(f.get('original_display_name'))})"
            thumb_td = (
                f'<td><img class="thumb-img" src="{_esc(f.get("crop_b64", ""))}" alt="Face #{f["face_index"] + 1}" /></td>'
                if include_thumbs
                else ""
            )
            rows.append(
                "<tr>"
                f"{thumb_td}"
                f"<td>#{f['face_index'] + 1}</td>"
                f"<td><strong>{_esc(f['display_name'])}</strong>{orig_txt}</td>"
                f"<td>{_esc(f['status'])}</td>"
                f"<td>{sim} / &Delta; {margin}</td>"
                f"<td>{size_txt} &mdash; {_esc(f.get('decision_reason', ''))}</td>"
                "</tr>"
            )
        return "".join(rows)

    thumb_th = "<th>Face</th>" if include_thumbs else ""

    absent_rows = (
        "".join(
            f"<tr><td>{_esc(p['display_name'])}</td>"
            f"<td>{'Needs Review (Likely match)' if p.get('pending_review') else 'Absent'}</td></tr>"
            for p in absent
        )
        or '<tr><td colspan="2">None</td></tr>'
    )

    not_enrolled_section = ""
    if not_enrolled:
        ne_rows = "".join(
            f"<tr><td>{_esc(p['display_name'])}</td><td>{_esc(p.get('original_filename', ''))}</td><td>Not enrolled (pending face pick)</td></tr>"
            for p in not_enrolled
        )
        not_enrolled_section = (
            "<h2>Not Enrolled (Multi-Face Pending)</h2>"
            "<table><thead><tr><th>Name</th><th>File</th><th>Status</th></tr></thead>"
            f"<tbody>{ne_rows}</tbody></table>"
        )

    trail_rows = (
        "".join(
            "<tr>"
            f"<td>{t['trail_id']}</td>"
            f"<td>{_esc(t['created_at'])}</td>"
            f"<td>{_esc(t['actor'])}</td>"
            f"<td>{_esc(t['action'])}</td>"
            f"<td>{'#' + str(t['face_index'] + 1) if t['face_index'] >= 0 else 'Session'}</td>"
            f"<td>{_esc(t.get('previous_status') or '—')} ({_esc(t.get('previous_display_name') or '—')})</td>"
            f"<td>{_esc(t['new_status'])} ({_esc(t['new_display_name'])})</td>"
            f"<td>{_esc(t.get('reason_note') or '')}</td>"
            "</tr>"
            for t in trail
        )
        or '<tr><td colspan="8">No trail events recorded.</td></tr>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>RollCall Attendance Sheet &mdash; {_esc(session_payload.get('filename', 'Session'))}</title>
  <link rel="stylesheet" href="/static/print.css" />
</head>
<body>
  <div class="print-toolbar">
    <span>RollCall Print-Ready Attendance Sheet (Offline Local Report)</span>
    <button type="button" id="btn-print-now" class="btn-print">Print / Save as PDF</button>
  </div>

  {demo_banner}

  <header class="sheet-header">
    <h1>RollCall Attendance Report</h1>
    <p class="meta-line">
      <strong>Group Photo:</strong> {_esc(session_payload.get('filename', ''))} &bull;
      <strong>Generated:</strong> {_esc(generated_at)} &bull;
      <strong>Model:</strong> {_esc(session_payload.get('model_version', ''))}
    </p>
    <p class="meta-line">
      <strong>Enrolled:</strong> {summary.get('enrolled_total', 0)} &bull;
      <strong>Thresholds:</strong> T_low={t_low}, T_high={t_high} &bull;
      <strong>Disclosure:</strong> Similarity scores are cosine similarities, not calibrated probabilities.
    </p>
  </header>

  <section class="summary-cards" aria-label="Attendance summary counts">
    <div class="summary-box"><span>Detected</span><strong>{summary.get('total_detected', 0)}</strong></div>
    <div class="summary-box"><span>Auto Confirmed</span><strong>{summary.get('confirmed', 0)}</strong></div>
    <div class="summary-box"><span>Reviewer Confirmed</span><strong>{summary.get('confirmed_by_reviewer', 0)}</strong></div>
    <div class="summary-box"><span>Needs Review</span><strong>{summary.get('likely', 0)}</strong></div>
    <div class="summary-box"><span>unknown</span><strong>{summary.get('unknown', 0)}</strong></div>
    <div class="summary-box"><span>Absent</span><strong>{summary.get('absent_count', 0)}</strong></div>
  </section>

  <h2>Present &mdash; Automated (Confirmed)</h2>
  <table>
    <thead><tr>{thumb_th}<th>#</th><th>Identity</th><th>Status</th><th>Similarity / Margin</th><th>Details</th></tr></thead>
    <tbody>{_face_rows(auto_confirmed)}</tbody>
  </table>

  <h2>Present &mdash; Confirmed by Reviewer</h2>
  <p class="note-muted">Separated from automated Confirmed; original automated predictions are preserved.</p>
  <table>
    <thead><tr>{thumb_th}<th>#</th><th>Identity</th><th>Status</th><th>Similarity / Margin</th><th>Details</th></tr></thead>
    <tbody>{_face_rows(reviewer_confirmed, show_orig=True)}</tbody>
  </table>

  <h2>Needs Review (Likely &mdash; Not Counted as Present)</h2>
  <table>
    <thead><tr>{thumb_th}<th>#</th><th>Candidate</th><th>Status</th><th>Similarity / Margin</th><th>Reason</th></tr></thead>
    <tbody>{_face_rows(likely_faces)}</tbody>
  </table>

  <h2>unknown Faces Detected</h2>
  <table>
    <thead><tr>{thumb_th}<th>#</th><th>Label</th><th>Status</th><th>Similarity / Margin</th><th>Reason</th></tr></thead>
    <tbody>{_face_rows(unknown_faces)}</tbody>
  </table>

  <h2>Absent (Enrolled Identities Not Confirmed Present)</h2>
  <table>
    <thead><tr><th>Enrolled Name</th><th>Attendance Status</th></tr></thead>
    <tbody>{absent_rows}</tbody>
  </table>

  {not_enrolled_section}

  <section class="appendix" aria-label="Append-Only Decision Trail Appendix">
    <h2>Appendix &mdash; Append-Only Decision Trail ({len(trail)} entries)</h2>
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
