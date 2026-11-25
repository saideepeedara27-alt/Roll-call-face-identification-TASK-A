"""API tests for RollCall FastAPI server."""

from io import BytesIO
import time
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from rollcall.app import create_app
from rollcall.db import RollCallDB
from rollcall.engine.base import DetectedFace, FaceEngine


class MockFaceEngine(FaceEngine):
    """Deterministic lightweight mock engine for API integration tests."""

    @property
    def model_version(self) -> str:
        return "mock_v1"

    def detect_faces(self, bgr_image: np.ndarray) -> list[DetectedFace]:
        h, w = bgr_image.shape[:2]
        # Blank dark images simulate 0 faces
        if bgr_image.mean() < 20:
            return []
        # Wide images simulate a 2-face group photo
        if w >= 600:
            return [
                DetectedFace((40, 40, 120, 120), 0.95, np.zeros(15, dtype=np.float32)),
                DetectedFace((340, 40, 120, 120), 0.92, np.zeros(15, dtype=np.float32)),
            ]
        return [DetectedFace((40, 40, 120, 120), 0.96, np.zeros(15, dtype=np.float32))]

    def extract_embedding(self, bgr_image: np.ndarray, face: DetectedFace) -> np.ndarray:
        # Left face gets vector [1, 0, ...], right face gets [0, 1, ...]
        vec = np.zeros(128, dtype=np.float32)
        if face.bbox[0] < 200:
            vec[0] = 1.0
        else:
            vec[1] = 1.0
        return vec


def _make_textured_jpeg(width: int, height: int, dark: bool = False) -> bytes:
    rng = np.random.default_rng(42)
    if dark:
        arr = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        arr = rng.integers(50, 220, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_enroll_identify_correct_export_and_wipe(tmp_path):
    db = RollCallDB(db_path=tmp_path / "test_rollcall.db")
    app = create_app(db=db, engine=MockFaceEngine())
    client = TestClient(app)

    # 1. Health endpoint reports offline local mode
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["offline_mode"] is True
    assert health.json()["bind_host"] == "127.0.0.1"
    assert health.json()["model_ready"] is True

    # 2. Enroll folder with 1 valid face, 1 zero-face image, and 1 bad extension
    valid_bytes = _make_textured_jpeg(300, 300)
    zero_face_bytes = _make_textured_jpeg(300, 300, dark=True)
    resp = client.post(
        "/api/enroll",
        files=[
            ("files", ("Priya_Sharma.jpg", valid_bytes, "image/jpeg")),
            ("files", ("Blank_Photo.jpg", zero_face_bytes, "image/jpeg")),
            ("files", ("notes.txt", b"hello", "text/plain")),
        ],
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    # Poll until completed
    summary = None
    for _ in range(30):
        st = client.get(f"/api/enroll/{job_id}").json()
        if st["status"] == "completed":
            summary = st["summary"]
            break
        time.sleep(0.05)

    assert summary is not None
    assert summary["enrolled_photos"] == 1
    assert summary["unique_identities"] == 1
    assert len(summary["errors"]) == 2  # Blank + notes.txt did not halt enrollment

    # 3. Identify group photo with 1 known (Priya) and 1 unknown
    group_bytes = _make_textured_jpeg(700, 350)
    id_resp = client.post(
        "/api/identify",
        files={"file": ("team_group.jpg", group_bytes, "image/jpeg")},
    )
    assert id_resp.status_code == 200
    data = id_resp.json()
    assert data["summary"]["total_detected"] == 2
    assert data["summary"]["confirmed"] == 1
    assert data["summary"]["unknown"] == 1
    assert data["faces"][0]["display_name"] == "Priya Sharma"
    assert data["faces"][1]["display_name"] == "unknown"

    session_id = data["session_id"]

    # 4. Human-in-the-loop: enroll the unknown face locally
    corr = client.post(
        f"/api/sessions/{session_id}/correct",
        json={"face_index": 1, "action": "enroll_new", "new_display_name": "Arjun Mehta"},
    )
    assert corr.status_code == 200
    assert corr.json()["summary"]["confirmed"] == 2
    assert corr.json()["summary"]["enrolled_total"] == 2

    # 5. Export PNG and CSV
    png_resp = client.get(f"/api/sessions/{session_id}/export/png")
    assert png_resp.status_code == 200
    assert png_resp.headers["content-type"] == "image/png"
    assert png_resp.content.startswith(b"\x89PNG")

    csv_resp = client.get(f"/api/sessions/{session_id}/export/csv")
    assert csv_resp.status_code == 200
    assert "Priya Sharma" in csv_resp.text
    assert "Arjun Mehta" in csv_resp.text

    # 6. Wipe All Data clears everything
    wipe = client.post("/api/wipe")
    assert wipe.status_code == 200
    assert wipe.json()["deleted"]["identities"] == 2
    assert client.get("/api/identities").json()["identities"] == []


def test_frontend_serves_locally_with_zero_external_urls(tmp_path):
    db = RollCallDB(db_path=tmp_path / "test_ui.db")
    app = create_app(db=db, engine=MockFaceEngine())
    client = TestClient(app)

    index_resp = client.get("/")
    assert index_resp.status_code == 200
    assert "RollCall" in index_resp.text
    assert "http://" not in index_resp.text
    assert "https://" not in index_resp.text

    css_resp = client.get("/static/styles.css")
    assert css_resp.status_code == 200
    assert "http://" not in css_resp.text
    assert "https://" not in css_resp.text

    js_resp = client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "http://" not in js_resp.text
    assert "https://" not in js_resp.text
