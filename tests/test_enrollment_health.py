"""Tests for Enrollment Health Studio (Step 5)."""

from io import BytesIO
import time
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from rollcall.app import create_app
from rollcall.db import RollCallDB
from rollcall.engine.base import DetectedFace, FaceEngine


class HealthMockEngine(FaceEngine):
    """Simulates single-face, multi-face, zero-face, same-person, and impostor extra photos."""

    @property
    def model_version(self) -> str:
        return "health_mock_v1"

    def detect_faces(self, bgr_image: np.ndarray) -> list[DetectedFace]:
        mean_r = float(bgr_image[:, :, 2].mean())
        if mean_r < 20:  # dark -> 0 faces
            return []
        if mean_r > 230:  # bright -> 2 faces (multi-face reference photo)
            return [
                DetectedFace((20, 20, 80, 80), 0.96, np.zeros(15, dtype=np.float32)),
                DetectedFace((150, 20, 80, 80), 0.91, np.zeros(15, dtype=np.float32)),
            ]
        return [DetectedFace((40, 40, 120, 120), 0.95, np.zeros(15, dtype=np.float32))]

    def extract_embedding(self, bgr_image: np.ndarray, face: DetectedFace) -> np.ndarray:
        mean_b = float(bgr_image[:, :, 0].mean())
        vec = np.zeros(128, dtype=np.float32)
        if mean_b > 200:
            # Distinct orthogonal person (sim = 0.0 against dim 0)
            vec[5] = 1.0
        elif face.bbox[0] > 100:
            vec[2] = 1.0
        else:
            vec[0] = 1.0
        return vec


def _make_textured_rgb_jpeg(r_base: int, b_base: int) -> bytes:
    rng = np.random.default_rng(11)
    arr = np.zeros((280, 280, 3), dtype=np.uint8)
    noise = rng.integers(-15, 16, size=(280, 280), dtype=np.int16)
    arr[:, :, 0] = np.clip(r_base + noise, 0, 255).astype(np.uint8)
    arr[:, :, 1] = np.clip(120 + noise, 0, 255).astype(np.uint8)
    arr[:, :, 2] = np.clip(b_base + noise, 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _poll_enroll(client: TestClient, job_id: str) -> dict:
    for _ in range(40):
        st = client.get(f"/api/enroll/{job_id}").json()
        if st["status"] == "completed":
            return st["summary"]
        time.sleep(0.05)
    raise AssertionError("Enrollment job timed out")


def test_enrollment_health_studio_counts_merge_mismatch_multiface_and_system_ignore(tmp_path):
    db = RollCallDB(db_path=tmp_path / "health_test.db")
    app = create_app(db=db, engine=HealthMockEngine())
    client = TestClient(app)

    priya_img = _make_textured_rgb_jpeg(120, 100)       # 1 face, vec[0]=1
    priya_extra_same = _make_textured_rgb_jpeg(122, 100)  # 1 face, vec[0]=1 (same person)
    priya_impostor = _make_textured_rgb_jpeg(120, 230)  # 1 face, vec[5]=1 (different person!)
    multiface_img = _make_textured_rgb_jpeg(245, 100)   # 2 faces -> needs_attention
    zeroface_img = _make_textured_rgb_jpeg(5, 5)        # 0 faces -> failed

    resp = client.post(
        "/api/enroll",
        files=[
            ("files", ("Priya Sharma.jpg", priya_img, "image/jpeg")),
            ("files", ("priya_sharma.png", priya_extra_same, "image/jpeg")),
            ("files", ("Priya Sharma (2).jpg", priya_impostor, "image/jpeg")),
            ("files", ("Arjun Mehta.jpg", multiface_img, "image/jpeg")),
            ("files", ("Blank.jpg", zeroface_img, "image/jpeg")),
            ("files", (".DS_Store", b"junk", "application/octet-stream")),
            ("files", ("Thumbs.db", b"junk", "application/octet-stream")),
            ("files", ("__MACOSX/meta.jpg", b"junk", "application/octet-stream")),
        ],
    )
    assert resp.status_code == 200
    summary = _poll_enroll(client, resp.json()["job_id"])

    counts = summary["counts"]
    # Exact additive check: 8 selected = 2 enrolled + 1 warning + 1 needs_attention + 1 failed + 3 ignored
    assert counts["selected"] == 8
    assert counts["enrolled"] == 2
    assert counts["enrolled_with_warnings"] == 1
    assert counts["needs_attention"] == 1
    assert counts["failed"] == 1
    assert counts["ignored"] == 3
    assert (
        counts["enrolled"]
        + counts["enrolled_with_warnings"]
        + counts["needs_attention"]
        + counts["failed"]
        + counts["ignored"]
        == counts["selected"]
    )

    records_by_name = {r["filename"]: r for r in summary["file_records"]}
    assert records_by_name["Priya Sharma.jpg"]["status"] == "enrolled"
    assert records_by_name["Priya Sharma.jpg"]["thumb_b64"].startswith("data:image/jpeg;base64,")

    # Same-person extra photo merges as info, NOT warning
    assert records_by_name["priya_sharma.png"]["status"] == "enrolled"
    assert "Extra photo merged into Priya Sharma" in records_by_name["priya_sharma.png"]["reason"]

    # Different-person extra photo flagged as warning
    assert records_by_name["Priya Sharma (2).jpg"]["status"] == "enrolled_with_warnings"
    assert "May be a different person" in records_by_name["Priya Sharma (2).jpg"]["reason"]

    # Multi-face stays in needs_attention until user chooses; provides boxed thumbnails
    assert records_by_name["Arjun Mehta.jpg"]["status"] == "needs_attention"
    assert len(summary["multiface_pending"]) == 1
    pending = summary["multiface_pending"][0]
    assert len(pending["candidates"]) == 2
    assert pending["candidates"][0]["annotated_thumb_b64"].startswith("data:image/jpeg;base64,")

    # System files ignored, not failed
    assert records_by_name[".DS_Store"]["status"] == "ignored"
    assert records_by_name["Thumbs.db"]["status"] == "ignored"
    assert records_by_name["meta.jpg"]["status"] == "ignored"
    assert records_by_name["Blank.jpg"]["status"] == "failed"

    # Before resolving Arjun Mehta, group photo upload works non-blockingly and lists Arjun as Not enrolled
    grp_resp = client.post(
        "/api/identify",
        files={"file": ("group.jpg", priya_img, "image/jpeg")},
    )
    assert grp_resp.status_code == 200
    grp_data = grp_resp.json()
    assert len(grp_data["not_enrolled_identities"]) == 1
    assert grp_data["not_enrolled_identities"][0]["display_name"] == "Arjun Mehta"
    assert grp_data["not_enrolled_identities"][0]["status"] == "Not enrolled"

    # Now user resolves multi-face Arjun Mehta -> moves from needs_attention to enrolled
    res = client.post(
        "/api/enroll/resolve-multiface",
        json={"token": pending["token"], "candidate_index": 0},
    )
    assert res.status_code == 200
    updated_counts = res.json()["summary"]["counts"]
    assert updated_counts["selected"] == 8
    assert updated_counts["enrolled"] == 3
    assert updated_counts["needs_attention"] == 0
    assert (
        updated_counts["enrolled"]
        + updated_counts["enrolled_with_warnings"]
        + updated_counts["needs_attention"]
        + updated_counts["failed"]
        + updated_counts["ignored"]
        == updated_counts["selected"]
    )


def test_demo_mode_reports_auto_selection_note(tmp_path):
    db = RollCallDB(db_path=tmp_path / "demo_note.db")
    app = create_app(db=db)  # real YuNet + SFace engine
    client = TestClient(app)

    resp = client.post("/api/demo")
    assert resp.status_code == 200
    data = resp.json()
    assert data["demo_note"] == "Demo: dominant face auto-selected for 2 photos"
    assert data["summary"]["enrolled_total"] == 6
