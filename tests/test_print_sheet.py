"""Tests for Step 9: Print-Ready Attendance Sheet (XSS escaping, DEMO banner, state & wipe)."""

from io import BytesIO
from pathlib import Path
import time
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from rollcall.app import create_app
from rollcall.db import RollCallDB
from rollcall.engine.base import DetectedFace, FaceEngine


class PrintMockEngine(FaceEngine):
    @property
    def model_version(self) -> str:
        return "print_mock_v1"

    def detect_faces(
        self, bgr_image: np.ndarray, score_threshold: float | None = None
    ) -> list[DetectedFace]:
        if bgr_image.shape[1] >= 600:
            return [
                DetectedFace((40, 40, 110, 110), 0.95, np.zeros(15, dtype=np.float32)),
                DetectedFace((260, 40, 110, 110), 0.93, np.zeros(15, dtype=np.float32)),
            ]
        return [DetectedFace((40, 40, 110, 110), 0.96, np.zeros(15, dtype=np.float32))]

    def extract_embedding(self, bgr_image: np.ndarray, face: DetectedFace) -> np.ndarray:
        vec = np.zeros(128, dtype=np.float32)
        if bgr_image.shape[1] < 600:
            vec[0] = 1.0
        elif face.bbox[0] < 150:
            vec[0] = 1.0  # Confirmed
        else:
            vec[0] = 0.40  # Likely
            vec[1] = float(np.sqrt(1 - 0.40**2))
        return vec


def _make_jpeg(w: int, h: int) -> bytes:
    rng = np.random.default_rng(5)
    arr = rng.integers(60, 200, size=(h, w, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_print_sheet_escaping_no_thumbs_state_changes_and_wipe(tmp_path):
    db = RollCallDB(db_path=tmp_path / "print_test.db")
    app = create_app(db=db, engine=PrintMockEngine())
    client = TestClient(app)

    # Enroll with XSS payload in filename (both <script> and <img onerror=...>)
    resp = client.post(
        "/api/enroll",
        files=[
            ("files", ("<script>alert(1)</script>.jpg", _make_jpeg(300, 300), "image/jpeg")),
            ("files", ("<img src=x onerror=alert(1)>.jpg", _make_jpeg(300, 300), "image/jpeg")),
        ],
    )
    job_id = resp.json()["job_id"]
    for _ in range(40):
        if client.get(f"/api/enroll/{job_id}").json()["status"] == "completed":
            break
        time.sleep(0.05)

    grp = client.post(
        "/api/identify",
        files={"file": ("<script>alert(1)</script>.jpg", _make_jpeg(680, 320), "image/jpeg")},
    )
    assert grp.status_code == 200
    sid = grp.json()["session_id"]

    # 1. Print sheet escapes all XSS strings, hides thumbs by default, omits DEMO banner
    pr1 = client.get(f"/api/sessions/{sid}/export/print")
    assert pr1.status_code == 200
    html1 = pr1.text
    assert "<script>alert(1)</script>" not in html1
    assert "<img src=x" not in html1.lower()
    assert "&lt;img" in html1 and "onerror=alert(1)&gt;" in html1.lower()
    assert "<img " not in html1  # No face thumbnails by default
    assert "DEMO DATA" not in html1
    assert 'class="appendix"' in html1

    # 2. Reviewer confirms face #1 with XSS note -> reflected on print sheet; Undo reverts it!
    client.post(
        f"/api/sessions/{sid}/correct",
        json={"face_index": 1, "action": "confirm", "reason_note": "<script>alert(1)</script>"},
    )
    pr2 = client.get(f"/api/sessions/{sid}/export/print")
    assert "Confirmed by reviewer" in pr2.text
    assert "<script>alert(1)</script>" not in pr2.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in pr2.text

    client.post(f"/api/sessions/{sid}/undo")
    pr3 = client.get(f"/api/sessions/{sid}/export/print")
    assert "Undid previous reviewer action" in pr3.text

    # 3. CSS print rules prevent row splitting and break appendix onto new page
    css = Path("static/print.css").read_text(encoding="utf-8")
    assert "break-inside: avoid" in css
    assert "break-before: page" in css

    # 4. Wipe All Data returns 404 (nothing) for print sheet
    assert client.post("/api/wipe").status_code == 200
    assert client.get(f"/api/sessions/{sid}/export/print").status_code == 404


def test_demo_print_sheet_shows_demo_data_banner(tmp_path):
    db = RollCallDB(db_path=tmp_path / "demo_print.db")
    app = create_app(db=db)
    client = TestClient(app)

    demo_resp = client.post("/api/demo")
    assert demo_resp.status_code == 200
    sid = demo_resp.json()["session_id"]

    pr = client.get(f"/api/sessions/{sid}/export/print")
    assert pr.status_code == 200
    assert "DEMO DATA" in pr.text
    assert "<img " not in pr.text
