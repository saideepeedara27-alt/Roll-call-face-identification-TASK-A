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
    # Exact additive check: 8 selected = 2 enrolled + 0 warnings + 2 needs_attention + 1 failed + 3 ignored
    assert counts["selected"] == 8
    assert counts["enrolled"] == 2
    assert counts["enrolled_with_warnings"] == 0
    assert counts["needs_attention"] == 2
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

    # Suspicious extra photo (< T_high) is held out of matching in needs_attention with side-by-side thumbs
    assert records_by_name["Priya Sharma (2).jpg"]["status"] == "needs_attention"
    assert "may be a different person" in records_by_name["Priya Sharma (2).jpg"]["reason"].lower()
    assert len(summary["extra_photo_pending"]) == 1
    extra_pend = summary["extra_photo_pending"][0]
    assert extra_pend["existing_thumb_b64"].startswith("data:image/jpeg;base64,")
    assert extra_pend["new_thumb_b64"].startswith("data:image/jpeg;base64,")

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
    assert updated_counts["needs_attention"] == 1

    # Rejecting flagged extra photo clears needs_attention without enrolling it
    rej = client.post(
        "/api/enroll/resolve-extra",
        json={"token": extra_pend["token"], "decision": "reject"},
    )
    assert rej.status_code == 200
    final_counts = rej.json()["summary"]["counts"]
    assert final_counts["needs_attention"] == 0
    assert final_counts["ignored"] == 4
    assert (
        final_counts["enrolled"]
        + final_counts["enrolled_with_warnings"]
        + final_counts["needs_attention"]
        + final_counts["failed"]
        + final_counts["ignored"]
        == final_counts["selected"]
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


class CalibratedExtraMockEngine(FaceEngine):
    """Deterministic engine keyed by red-channel tag for testing 0.49 extra-photo holdout."""

    @property
    def model_version(self) -> str:
        return "cal_extra_mock_v1"

    def detect_faces(self, bgr_image: np.ndarray) -> list[DetectedFace]:
        return [DetectedFace((40, 40, 120, 120), 0.96, np.zeros(15, dtype=np.float32))]

    def extract_embedding(self, bgr_image: np.ndarray, face: DetectedFace) -> np.ndarray:
        tag = int(round(float(bgr_image[:, :, 2].mean())))
        vec = np.zeros(128, dtype=np.float32)
        if tag < 60:
            # First 16 primary identities occupy Orthogonal dims + high-pair cluster in dims 0..15
            # Put pair (0, 1) with sim = 0.56 so 99.5th percentile across >=15 identities hits cap T_low=0.50 -> T_high=0.56
            if tag == 0:
                vec[0] = 1.0
            elif tag == 1:
                vec[0] = 0.56
                vec[1] = float(np.sqrt(1.0 - 0.56**2))
            elif tag == 4:
                # Second high-sim pair (3, 4) so 99.5th percentile across 120 pairs hits cap T_low=0.50 -> T_high=0.56
                vec[3] = 0.56
                vec[4] = float(np.sqrt(1.0 - 0.56**2))
            else:
                vec[tag] = 1.0
        elif tag == 100:
            # Genuine same-person extra for Identity 0 (sim = 0.90 >= T_high=0.56)
            vec[0] = 0.90
            vec[20] = float(np.sqrt(1.0 - 0.90**2))
        elif tag == 120:
            # Suspicious extra photo for Identity 2 (dim 2): sim = 0.49 (between 0.363 and T_high=0.56),
            # with remaining norm in dim 25 so a matching probe along dim 25 + dim 2 tests holdout vs keep!
            vec[2] = 0.49
            vec[25] = float(np.sqrt(1.0 - 0.49**2))
        elif tag == 140:
            # Probe face identical to tag 120's person (dim 25 dominant: sim to tag 120 == 1.0, sim to Identity 2's original vec[2] == 0.49 < T_low=0.50 -> unknown unless kept!)
            vec[2] = 0.49
            vec[25] = float(np.sqrt(1.0 - 0.49**2))
        return vec


def test_extra_photo_049_holdout_reject_and_keep_with_calibrated_t_high(tmp_path):
    db = RollCallDB(db_path=tmp_path / "extra_049.db")
    engine = CalibratedExtraMockEngine()
    app = create_app(db=db, engine=engine)
    client = TestClient(app)

    # Use distinct non-numeric suffixes (A..P) so filename parser produces 16 distinct identities
    letters = "ABCDEFGHIJKLMNOP"
    files = [
        ("files", (f"Member {ch}.png", _make_solid_tag_png(i), "image/png"))
        for i, ch in enumerate(letters)
    ]
    # Add genuine high-similarity extra photo for Member A (tag 0, sim 0.90 >= 0.56 -> merges silently)
    files.append(("files", ("Member A_2.png", _make_solid_tag_png(100), "image/png")))
    # Add 0.49 extra photo for Member C (tag 2, 0.363 <= 0.49 < T_high=0.56 -> flagged & held out of matching)
    files.append(("files", ("Member C_2.png", _make_solid_tag_png(120), "image/png")))

    resp = client.post("/api/enroll", files=files)
    assert resp.status_code == 200
    summary = _poll_enroll(client, resp.json()["job_id"])

    assert summary["calibration"]["final_t_low"] == 0.50
    assert summary["calibration"]["final_t_high"] == 0.56

    recs = {r["filename"]: r for r in summary["file_records"]}
    assert recs["Member A_2.png"]["status"] == "enrolled"
    assert "Extra photo merged into Member A" in recs["Member A_2.png"]["reason"]

    assert recs["Member C_2.png"]["status"] == "needs_attention"
    assert "may be a different person" in recs["Member C_2.png"]["reason"].lower()
    assert len(summary["extra_photo_pending"]) == 1
    pend = summary["extra_photo_pending"][0]
    assert pend["similarity"] == 0.49
    assert pend["t_high"] == 0.56
    assert pend["existing_thumb_b64"].startswith("data:image/jpeg;base64,")
    assert pend["new_thumb_b64"].startswith("data:image/jpeg;base64,")

    # Held photo is NOT in DB exemplars and NOT used for matching (probe returns unknown)
    exemplars_before = db.load_all_exemplars(engine.model_version)
    assert [e["original_filename"] for e in exemplars_before if e["identity_id"] == "member_c"] == ["Member C.png"]
    probe_img = _make_solid_tag_png(140)
    ident_held = client.post("/api/identify", files={"file": ("probe.png", probe_img, "image/png")}).json()
    assert ident_held["faces"][0]["status"] == "unknown"

    # Rejecting it removes it from pending review without adding to exemplars
    rej = client.post("/api/enroll/resolve-extra", json={"token": pend["token"], "decision": "reject"})
    assert rej.status_code == 200
    assert rej.json()["status"] == "rejected"
    assert rej.json()["summary"]["counts"]["needs_attention"] == 0
    assert [e["original_filename"] for e in db.load_all_exemplars(engine.model_version) if e["identity_id"] == "member_c"] == ["Member C.png"]
    ident_rejected = client.post("/api/identify", files={"file": ("probe.png", probe_img, "image/png")}).json()
    assert ident_rejected["faces"][0]["status"] == "unknown"

    # Re-upload the 0.49 photo onto the existing roster (clear_existing=false) and choose Keep -> adds as exemplar
    resp2 = client.post(
        "/api/enroll",
        data={"clear_existing": "false"},
        files=[("files", ("Member C_2.png", _make_solid_tag_png(120), "image/png"))],
    )
    summary2 = _poll_enroll(client, resp2.json()["job_id"])
    assert len(summary2["extra_photo_pending"]) == 1
    token2 = summary2["extra_photo_pending"][0]["token"]

    keep = client.post("/api/enroll/resolve-extra", json={"token": token2, "decision": "keep"})
    assert keep.status_code == 200
    assert keep.json()["status"] == "enrolled"
    assert keep.json()["summary"]["counts"]["needs_attention"] == 0
    exemplars_after = db.load_all_exemplars(engine.model_version)
    assert sorted(e["original_filename"] for e in exemplars_after if e["identity_id"] == "member_c") == [
        "Member C.png",
        "Member C_2.png",
    ]
    ident_kept = client.post("/api/identify", files={"file": ("probe.png", probe_img, "image/png")}).json()
    assert ident_kept["faces"][0]["status"] == "Confirmed"
    assert ident_kept["faces"][0]["display_name"] == "Member C"


def _make_solid_tag_png(r_val: int) -> bytes:
    arr = np.zeros((280, 280, 3), dtype=np.uint8)
    ys, xs = np.indices((280, 280))
    checker = np.where((xs + ys) % 2 == 0, 8, -8)
    arr[:, :, 0] = r_val
    arr[:, :, 1] = np.clip(120 + checker, 0, 255).astype(np.uint8)
    arr[:, :, 2] = np.clip(120 + checker, 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
