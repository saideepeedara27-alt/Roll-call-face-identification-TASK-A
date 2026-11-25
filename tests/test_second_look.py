"""Tests for Step 7: Second Look tiled detection, manual box, dedup, and low-score cap."""

from io import BytesIO
import time
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from rollcall.app import create_app
from rollcall.db import RollCallDB
from rollcall.engine.base import DetectedFace, FaceEngine
from rollcall.second_look import compute_tile_windows, is_duplicate_box


class SecondLookMockEngine(FaceEngine):
    """Simulates first-pass miss, tile-seam face, tiny face, and low-score manual face."""

    @property
    def model_version(self) -> str:
        return "second_look_mock_v1"

    def detect_faces(
        self, bgr_image: np.ndarray, score_threshold: float | None = None
    ) -> list[DetectedFace]:
        h, w = bgr_image.shape[:2]
        # Reference portrait (small square 300x300)
        if w == 300 and h == 300:
            return [DetectedFace((40, 40, 120, 120), 0.96, np.zeros(15, dtype=np.float32))]

        # Full group photo first pass (1000x800): finds only face #0 on left
        if w == 1000 and h == 800:
            return [DetectedFace((80, 120, 110, 110), 0.95, np.zeros(15, dtype=np.float32))]

        # Manual box crop with relaxed score_threshold <= 0.40
        if score_threshold is not None and score_threshold <= 0.40:
            # Dark region center -> no face
            ch, cw = h, w
            center = bgr_image[ch // 4 : (3 * ch) // 4, cw // 4 : (3 * cw) // 4]
            if float(center.mean()) < 30.0:
                return []
            # Otherwise detect a low-score (0.42 < 0.65) face inside crop
            return [DetectedFace((cw // 4, ch // 4, cw // 2, ch // 2), 0.42, np.zeros(15, dtype=np.float32))]

        # Tiled pass (each 2x2 tile on 1000x800 is 556x444, upscaled or native):
        # Return a face sitting right on the vertical seam (around x=500) AND a tiny face (20x20)
        # Note: tiles with x offset 0 vs 444 both cover x=500!
        # We check brightness in the tile: only return seam face relative to tile coords
        results = []
        # Tile ~556x444 (top-left or top-right): simulate seam face at global x=480, y=140, size 80x80
        # Identify tile by checking pixel marker channel 2 encoded in top-left of image? Simpler:
        # Encode global coordinates in image color markers!
        # If tile contains green marker at (485, 145) -> return seam box relative to tile!
        ys, xs = np.where((bgr_image[:, :, 1] > 230) & (bgr_image[:, :, 0] < 30))
        if len(xs) > 0:
            mx, my = int(xs.min()), int(ys.min())
            results.append(DetectedFace((mx, my, 80, 80), 0.88, np.zeros(15, dtype=np.float32)))

        # If tile contains red marker -> tiny 20x20 face
        rys, rxs = np.where((bgr_image[:, :, 2] > 230) & (bgr_image[:, :, 0] < 30))
        if len(rxs) > 0:
            rx, ry = int(rxs.min()), int(rys.min())
            results.append(DetectedFace((rx, ry, 20, 20), 0.85, np.zeros(15, dtype=np.float32)))

        return results

    def extract_embedding(self, bgr_image: np.ndarray, face: DetectedFace) -> np.ndarray:
        vec = np.zeros(128, dtype=np.float32)
        vec[0] = 1.0  # Matches Priya Sharma
        return vec


def _make_group_scene() -> bytes:
    rng = np.random.default_rng(9)
    arr = rng.integers(80, 150, size=(800, 1000, 3), dtype=np.uint8)
    # Green marker block at seam (x=480..540, y=140..200) -> visible in BOTH left and right tiles!
    # RGB format: R=10, G=245, B=10 (with texture so blur passes)
    arr[140:200, 480:540, 0] = 10
    arr[140:200, 480:540, 1] = 245
    arr[140:200, 480:540, 2] = 10
    arr[145:195:4, 485:535:4, 1] = 120

    # Red marker block at (x=760..780, y=600..620) -> tiny 20x20 face
    arr[600:620, 760:780, 0] = 245
    arr[600:620, 760:780, 1] = 10
    arr[600:620, 760:780, 2] = 10

    # Dark empty region at bottom-left (x=40..160, y=620..740) -> no face for manual box
    arr[620:740, 40:160, :] = 5

    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _make_ref_portrait() -> bytes:
    rng = np.random.default_rng(1)
    arr = rng.integers(80, 180, size=(300, 300, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _enroll_priya(client: TestClient) -> None:
    resp = client.post(
        "/api/enroll",
        files=[("files", ("Priya Sharma.jpg", _make_ref_portrait(), "image/jpeg"))],
    )
    job_id = resp.json()["job_id"]
    for _ in range(40):
        st = client.get(f"/api/enroll/{job_id}").json()
        if st["status"] == "completed":
            return
        time.sleep(0.05)
    raise AssertionError("Enroll timed out")


def test_tile_windows_and_dedup_rules():
    grid_std, win_std = compute_tile_windows(1000, 800, overlap_ratio=0.20)
    assert grid_std == "2x2"
    assert len(win_std) == 4
    # 556px wide -> overlap = 556 + 556 - 1000 = 112px (~20.1% of 556)
    assert win_std[0][2] == 556

    grid_wide, win_wide = compute_tile_windows(2000, 1000, overlap_ratio=0.20)
    assert grid_wide == "3x2"
    assert len(win_wide) == 6
    # 2000 / 2.6 = 769px wide (~38.5% of image, 20% tile overlap)
    assert win_wide[0][2] == 769

    # IoU > 0.40 dedupes
    assert is_duplicate_box((100, 100, 80, 80), (110, 105, 80, 80)) is True
    # Center-inside with VERY different area (20x20 inside 200x200) must NOT dedupe
    assert is_duplicate_box((100, 100, 20, 20), (20, 20, 200, 200)) is False
    # Center-inside with similar area (60x60 inside 80x80) DOES dedupe
    assert is_duplicate_box((110, 110, 60, 60), (100, 100, 80, 80)) is True


def test_second_look_seam_stability_idempotence_tiny_reject_manual_and_low_score(tmp_path):
    db = RollCallDB(db_path=tmp_path / "sl_test.db")
    app = create_app(db=db, engine=SecondLookMockEngine())
    client = TestClient(app)
    _enroll_priya(client)

    grp = client.post(
        "/api/identify",
        files={"file": ("scene.png", _make_group_scene(), "image/png")},
    )
    assert grp.status_code == 200
    g_data = grp.json()
    sid = g_data["session_id"]
    assert len(g_data["faces"]) == 1  # Only face #0 found on first pass

    # Reviewer rejects face #0 to unknown; Second Look must leave face #0 untouched!
    rej = client.post(
        f"/api/sessions/{sid}/correct",
        json={"face_index": 0, "action": "mark_unknown"},
    )
    assert rej.status_code == 200

    # 1. Run Second Look: finds seam face ONCE (80x80 -> Confirmed) + tiny 20x20 face (rejected by quality gate -> unknown)
    sl1 = client.post(f"/api/sessions/{sid}/second-look")
    assert sl1.status_code == 200
    d1 = sl1.json()
    assert d1["second_look"]["first_pass_input_size"] == [1000, 800]
    assert d1["second_look"]["tile_grid"] == "2x2"
    assert d1["second_look"]["new_faces_found"] == 2
    assert len(d1["faces"]) == 3

    # Existing face #0 stayed untouched at index 0
    assert d1["faces"][0]["face_index"] == 0
    assert d1["faces"][0]["status"] == "unknown"
    assert d1["faces"][0]["manually_corrected"] is True

    # New faces appended at indices 1 and 2
    assert d1["faces"][1]["face_index"] == 1
    assert d1["faces"][1]["source"] == "second_look"
    assert d1["faces"][2]["face_index"] == 2
    assert d1["faces"][2]["source"] == "second_look"

    # Tiny face (20x20) rejected by original-pixel quality gate -> unknown
    tiny_face = next(f for f in d1["faces"][1:] if f["quality"]["min_side_px"] < 28)
    assert tiny_face["status"] == "unknown"
    assert "too small" in tiny_face["decision_reason"].lower()

    seam_face = next(f for f in d1["faces"][1:] if f["quality"]["min_side_px"] >= 28)
    assert seam_face["status"] == "Confirmed"

    # 2. Running Second Look a second time adds NOTHING
    sl2 = client.post(f"/api/sessions/{sid}/second-look")
    assert sl2.status_code == 200
    assert sl2.json()["second_look"]["new_faces_found"] == 0
    assert len(sl2.json()["faces"]) == 3

    # 3. Manual box on dark empty region -> no face found, friendly message
    mb_empty = client.post(
        f"/api/sessions/{sid}/manual-box",
        json={"x": 45, "y": 625, "w": 110, "h": 110},
    )
    assert mb_empty.status_code == 200
    assert mb_empty.json()["manual_box"]["found"] is False
    assert "No aligned face detected" in mb_empty.json()["manual_box"]["message"]
    assert len(mb_empty.json()["faces"]) == 3

    # 4. Manual box on textured region -> low detector score (0.42 < 0.65) caps at Likely, never Confirmed!
    mb_low = client.post(
        f"/api/sessions/{sid}/manual-box",
        json={"x": 220, "y": 350, "w": 160, "h": 160},
    )
    assert mb_low.status_code == 200
    m_data = mb_low.json()
    assert m_data["manual_box"]["found"] is True
    added_idx = m_data["manual_box"]["added_face_index"]
    assert added_idx == 3
    manual_face = m_data["faces"][3]
    assert manual_face["source"] == "manual_box"
    assert manual_face["status"] == "Likely"  # Never Confirmed!
    assert "low detector score" in manual_face["decision_reason"].lower()

    # 5. Removing manual box marks removed=True (not deleted) and logs to trail
    rem = client.post(f"/api/sessions/{sid}/faces/{added_idx}/remove")
    assert rem.status_code == 200
    rem_data = rem.json()
    assert len(rem_data["faces"]) == 4
    assert rem_data["faces"][3]["removed"] is True
    assert rem_data["faces"][3]["status"] == "Removed"
    assert rem_data["summary"]["total_detected"] == 3  # Active faces only
    actions = [t["action"] for t in rem_data["decision_trail"]]
    assert "second_look" in actions
    assert "manual_box" in actions
    assert "manual_box_removed" in actions
