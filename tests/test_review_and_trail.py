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

    # Initial states: 0=Confirmed(Priya), 1=Likely(Rahul), 2=unknown
    assert data["faces"][0]["status"] == "Confirmed"
    assert data["faces"][1]["status"] == "Likely"
    assert data["faces"][2]["status"] == "unknown"
    assert data["summary"]["confirmed"] == 1
    assert data["summary"]["confirmed_by_reviewer"] == 0
    assert data["summary"]["absent_count"] == 1  # Rahul is Absent while Likely!
    assert [a["display_name"] for a in data["absent_identities"]] == ["Rahul Verma"]

    # 1. enroll_new is disabled (400)
    bad_enroll = client.post(
        f"/api/sessions/{sid}/correct",
        json={"face_index": 2, "action": "enroll_new", "new_display_name": "Stranger"},
    )
    assert bad_enroll.status_code == 400

    # 2. Confirm (C) on unknown face does nothing / rejected (400)
    bad_confirm = client.post(
        f"/api/sessions/{sid}/correct",
        json={"face_index": 2, "action": "confirm"},
    )
    assert bad_confirm.status_code == 400

    # 3. Confirm (C) on Likely Rahul -> "Confirmed by reviewer", attendance recomputes!
    conf = client.post(
        f"/api/sessions/{sid}/correct",
        json={"face_index": 1, "action": "confirm", "reason_note": "Verified glasses"},
    )
    assert conf.status_code == 200
    c_data = conf.json()
    assert c_data["faces"][1]["status"] == "Confirmed by reviewer"
    assert c_data["faces"][1]["original_status"] == "Likely"
    assert c_data["faces"][1]["original_display_name"] == "Rahul Verma"
    assert c_data["summary"]["confirmed"] == 1
    assert c_data["summary"]["confirmed_by_reviewer"] == 1
    assert c_data["summary"]["absent_count"] == 0  # Rahul moved from Absent to Present!
    assert c_data["can_undo"] is True

    # 4. Reject Rahul to unknown -> Rahul moves back to Absent!
    rej = client.post(
        f"/api/sessions/{sid}/correct",
        json={"face_index": 1, "action": "mark_unknown", "reason_note": "Actually not Rahul"},
    )
    assert rej.status_code == 200
    r_data = rej.json()
    assert r_data["faces"][1]["status"] == "unknown"
    assert r_data["summary"]["confirmed_by_reviewer"] == 0
    assert r_data["summary"]["absent_count"] == 1
    assert [a["display_name"] for a in r_data["absent_identities"]] == ["Rahul Verma"]

    # 5. Undo (Z) restores "Confirmed by reviewer" and appends undo entry (never deletes trail!)
    undone = client.post(f"/api/sessions/{sid}/undo")
    assert undone.status_code == 200
    u_data = undone.json()
    assert u_data["faces"][1]["status"] == "Confirmed by reviewer"
    assert u_data["summary"]["absent_count"] == 0

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

    # 6. Wipe All Data deletes decision_trail cleanly
    wipe = client.post("/api/wipe")
    assert wipe.status_code == 200
    assert wipe.json()["deleted"]["decision_trail"] == 6
    assert db.get_session_trail(sid) == []


def test_frontend_keyboard_queue_and_input_guard_contract():
    js_text = Path("static/app.js").read_text(encoding="utf-8")
    # Queue filters strictly to Likely and unknown
    assert 'f.status === "Likely" || f.status === "unknown"' in js_text
    # Shortcuts ignored while typing in inputs/selects/textareas
    assert '["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)' in js_text
    # J/K used for navigation instead of arrow keys
    assert 'k === "j"' in js_text and 'k === "k"' in js_text
    assert "ArrowDown" not in js_text and "ArrowUp" not in js_text
    # Searchable type-to-filter box for Assign (A)
    assert "assign-search-input" in js_text
    assert 'k === "z"' in js_text
