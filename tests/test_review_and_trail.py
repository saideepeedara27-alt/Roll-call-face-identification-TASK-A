"""Tests for Step 6: Review Queue rules, Confirmed by reviewer, Undo, Append-Only Trail & Attendance."""

from io import BytesIO
from pathlib import Path
import time
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from rollcall.app import create_app
from rollcall.db import RollCallDB
from rollcall.engine.base import DetectedFace, FaceEngine


class ReviewMockEngine(FaceEngine):
    """Produces 1 Confirmed, 1 Likely, and 1 unknown face in a group photo."""

    @property
    def model_version(self) -> str:
        return "review_mock_v1"

    def detect_faces(self, bgr_image: np.ndarray) -> list[DetectedFace]:
        if bgr_image.shape[1] >= 700:
            # 3 faces in group photo
            return [
                DetectedFace((40, 40, 120, 120), 0.96, np.zeros(15, dtype=np.float32)),
                DetectedFace((260, 40, 120, 120), 0.94, np.zeros(15, dtype=np.float32)),
                DetectedFace((480, 40, 120, 120), 0.93, np.zeros(15, dtype=np.float32)),
            ]
        return [DetectedFace((40, 40, 120, 120), 0.96, np.zeros(15, dtype=np.float32))]

    def extract_embedding(self, bgr_image: np.ndarray, face: DetectedFace) -> np.ndarray:
        vec = np.zeros(128, dtype=np.float32)
        # Reference portraits differentiated by blue channel mean
        if bgr_image.shape[1] < 700:
            mean_b = float(bgr_image[:, :, 0].mean())
            if mean_b > 180:
                vec[1] = 1.0  # Rahul Verma
            else:
                vec[0] = 1.0  # Priya Sharma
            return vec

        # Group faces:
        if face.bbox[0] < 150:
            vec[0] = 1.0  # Confirmed Priya Sharma (sim 1.0)
        elif face.bbox[0] < 350:
            # Borderline Rahul Verma (sim 0.40 between 0.363 and 0.44 -> Likely)
            vec[1] = 0.40
            vec[2] = float(np.sqrt(1 - 0.40**2))
        else:
            vec[9] = 1.0  # Unknown stranger
        return vec


def _make_jpeg(width: int, height: int, blue: int) -> bytes:
    rng = np.random.default_rng(7)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    noise = rng.integers(10, 40, size=(height, width), dtype=np.uint8)
    arr[:, :, 0] = np.clip(100 + noise, 0, 255)
    arr[:, :, 1] = np.clip(120 + noise, 0, 255)
    arr[:, :, 2] = np.clip(blue + noise // 3, 0, 255)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _enroll_two(client: TestClient) -> None:
    resp = client.post(
        "/api/enroll",
        files=[
            ("files", ("Priya Sharma.jpg", _make_jpeg(300, 300, 80), "image/jpeg")),
            ("files", ("Rahul Verma.jpg", _make_jpeg(300, 300, 210), "image/jpeg")),
        ],
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    for _ in range(40):
        st = client.get(f"/api/enroll/{job_id}").json()
        if st["status"] == "completed":
            return
        time.sleep(0.05)
    raise AssertionError("Enroll timed out")


def test_review_actions_undo_append_only_trail_attendance_and_wipe(tmp_path):
    db = RollCallDB(db_path=tmp_path / "review_test.db")
    app = create_app(db=db, engine=ReviewMockEngine())
    client = TestClient(app)

    _enroll_two(client)
    grp = client.post(
        "/api/identify",
        files={"file": ("meeting.jpg", _make_jpeg(750, 350, 100), "image/jpeg")},
    )
    assert grp.status_code == 200
    data = grp.json()
    sid = data["session_id"]

    # Initial states: 0=High-confidence match(Priya), 1=Needs review(Rahul), 2=Unknown
    assert data["faces"][0]["status"] == "High-confidence match"
    assert data["faces"][1]["status"] == "Needs review"
    assert data["faces"][2]["status"] == "Unknown"
    assert data["faces"][0]["decision_source"] == "model"
    assert data["summary"]["high_confidence_matches"] == 1
    assert data["summary"]["needs_review_faces"] == 1
    assert data["summary"]["unknown_faces"] == 1
    assert data["summary"]["manual_decisions"] == 0

    # 1. enroll_new is disabled (400)
    bad_enroll = client.post(
        f"/api/sessions/{sid}/correct",
        json={"face_index": 2, "action": "enroll_new", "new_display_name": "Stranger"},
    )
    assert bad_enroll.status_code == 400

    # 2. Confirm on Unknown face does nothing / rejected (400)
    bad_confirm = client.post(
        f"/api/sessions/{sid}/correct",
        json={"face_index": 2, "action": "confirm"},
    )
    assert bad_confirm.status_code == 400

    # 3. Confirm on Needs review Rahul -> High-confidence match + Manual decision!
    conf = client.post(
        f"/api/sessions/{sid}/correct",
        json={"face_index": 1, "action": "confirm", "reason_note": "Verified glasses"},
    )
    assert conf.status_code == 200
    c_data = conf.json()
    assert c_data["faces"][1]["status"] == "High-confidence match"
    assert c_data["faces"][1]["decision_source"] == "manual"
    assert c_data["faces"][1]["original_status"] == "Needs review"
    assert c_data["faces"][1]["original_display_name"] == "Rahul Verma"
    assert c_data["summary"]["high_confidence_matches"] == 2
    assert c_data["summary"]["manual_decisions"] == 1
    assert c_data["can_undo"] is True

    # 4. Mark Unknown on Rahul -> Unknown + Manual decision
    rej = client.post(
        f"/api/sessions/{sid}/correct",
        json={"face_index": 1, "action": "mark_unknown", "reason_note": "Actually not Rahul"},
    )
    assert rej.status_code == 200
    r_data = rej.json()
    assert r_data["faces"][1]["status"] == "Unknown"
    assert r_data["faces"][1]["decision_source"] == "manual"
    assert r_data["summary"]["high_confidence_matches"] == 1
    assert r_data["summary"]["unknown_faces"] == 2

    # 5. Undo reverts face 1 to High-confidence match (Manual decision) and appends undo entry
    undone = client.post(f"/api/sessions/{sid}/undo")
    assert undone.status_code == 200
    u_data = undone.json()
    assert u_data["faces"][1]["status"] == "High-confidence match"
    assert u_data["faces"][1]["decision_source"] == "manual"
    assert u_data["summary"]["high_confidence_matches"] == 2

    # Trail is append-only: 3 automated initial + 3 reviewer (confirm, mark_unknown, undo) = 6 rows
    trail = u_data["decision_trail"]
    assert len(trail) == 6
    face1_trail = [t for t in trail if t["face_index"] == 1]
    assert [t["action"] for t in face1_trail] == ["initial_detection", "confirm", "mark_unknown", "undo"]
    assert face1_trail[0]["actor"] == "automated"
    assert face1_trail[1]["actor"] == "reviewer"
    assert face1_trail[1]["session_id"] == sid
    assert face1_trail[1]["group_image_sha256"]
    assert face1_trail[1]["face_bbox"] == [260, 40, 120, 120]
    assert face1_trail[1]["reason_note"] == "Verified glasses"

    # 6. Clear session deletes decision_trail cleanly
    wipe = client.post("/api/wipe")
    assert wipe.status_code == 200
    assert wipe.json()["deleted"]["decision_trail"] == 6
    assert db.get_session_trail(sid) == []


def test_frontend_evidence_card_and_results_list_contract():
    html_text = Path("static/index.html").read_text(encoding="utf-8")
    js_text = Path("static/app.js").read_text(encoding="utf-8")
    assert 'id="evidence-card"' in html_text
    assert 'id="results-list"' in html_text
    assert 'id="ev-person-select"' in html_text
    assert "Manual decision" in js_text
    assert "Model suggestion" in js_text
    assert "High-confidence match" in js_text
    assert "Needs review" in js_text
    assert "Unknown" in js_text
