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
    assert data["summary"]["faces_detected"] == 2
    assert data["summary"]["high_confidence_matches"] == 1
    assert data["summary"]["unknown_faces"] == 1
    assert data["faces"][0]["display_name"] == "Priya Sharma"
    assert data["faces"][1]["display_name"] == "Unknown"

    session_id = data["session_id"]

    # 4. Human-in-the-loop: assign unknown face to enrolled Priya Sharma (does not mutate embeddings)
    corr = client.post(
        f"/api/sessions/{session_id}/correct",
        json={"face_index": 1, "action": "assign", "identity_id": "priya_sharma", "reason_note": "Side angle"},
    )
    assert corr.status_code == 200
    assert corr.json()["summary"]["high_confidence_matches"] == 2
    assert corr.json()["summary"]["manual_decisions"] == 1
    assert corr.json()["faces"][1]["decision_source"] == "manual"
    assert corr.json()["summary"]["enrolled_total"] == 1

    # 5. Export PNG and CSV
    png_resp = client.get(f"/api/sessions/{session_id}/export/png")
    assert png_resp.status_code == 200
    assert png_resp.headers["content-type"] == "image/png"
    assert png_resp.content.startswith(b"\x89PNG")

    csv_resp = client.get(f"/api/sessions/{session_id}/export/csv")
    assert csv_resp.status_code == 200
    assert "Priya Sharma" in csv_resp.text
    assert "Manual decision" in csv_resp.text
    assert "Absent" not in csv_resp.text

    # 6. Clear session wipes everything including decision trail
    wipe = client.post("/api/wipe")
    assert wipe.status_code == 200
    assert wipe.json()["deleted"]["identities"] == 1
    assert wipe.json()["deleted"]["decision_trail"] >= 3
    assert client.get("/api/identities").json()["identities"] == []


def test_frontend_serves_locally_with_zero_external_urls(tmp_path):
    db = RollCallDB(db_path=tmp_path / "test_ui.db")
    app = create_app(db=db, engine=MockFaceEngine())
    client = TestClient(app)

    index_resp = client.get("/")
    assert index_resp.status_code == 200
    assert "PhotoProof" in index_resp.text
    assert "Attendance" not in index_resp.text
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


def test_ephemeral_cache_default_and_enroll_from_path_safety(tmp_path):
    db = RollCallDB(db_path=tmp_path / "test_ephemeral.db")
    app = create_app(db=db, engine=MockFaceEngine())
    client = TestClient(app)

    folder = tmp_path / "team_photos"
    folder.mkdir()
    (folder / "Kavya_Nair.jpg").write_bytes(_make_textured_jpeg(300, 300))

    # 1. Path fallback enrolls cleanly with ephemeral default (no cache rows written)
    resp = client.post("/api/enroll/from-path", json={"folder_path": str(folder)})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

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

    # Verify embedding_cache stayed empty under ephemeral default
    with db._connect() as conn:
        cache_rows = conn.execute("SELECT COUNT(*) FROM embedding_cache;").fetchone()[0]
    assert cache_rows == 0

    # 2. Rejects missing directory and forbidden system paths
    assert client.post("/api/enroll/from-path", json={"folder_path": str(tmp_path / "nope")}).status_code == 400
    assert client.post("/api/enroll/from-path", json={"folder_path": "/etc"}).status_code == 400
    assert client.post("/api/enroll/from-path", json={"folder_path": "/"}).status_code == 400


def test_strict_csp_headers_and_startup_self_check(tmp_path):
    from rollcall.self_check import run_startup_self_check

    db = RollCallDB(db_path=tmp_path / "test_csp.db")
    app = create_app(db=db, engine=MockFaceEngine())
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    csp = resp.headers.get("content-security-policy", "")
    assert "default-src 'none'" in csp
    assert "script-src 'self'" in csp
    assert "style-src 'self'" in csp
    assert "'unsafe-inline'" not in csp
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"

    health = client.get("/api/health").json()
    assert health["self_check"]["ok"] is True
    assert health["self_check"]["checks"]["static_assets"]["ok"] is True
    assert health["self_check"]["checks"]["database"]["ok"] is True

    # Real model checksum self-check against local models/ directory
    from rollcall.engine import OpenCVYuNetSFaceEngine

    real_engine = OpenCVYuNetSFaceEngine()
    real_check = run_startup_self_check(db=db, engine=real_engine, model_error=None)
    assert real_check["ok"] is True
    assert real_check["checks"]["models"]["ok"] is True

    # Failing check when model is missing
    bad_check = run_startup_self_check(db=db, engine=None, model_error="Missing weights")
    assert bad_check["ok"] is False
    assert bad_check["checks"]["models"]["ok"] is False
