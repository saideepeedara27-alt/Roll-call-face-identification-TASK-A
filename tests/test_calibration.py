"""Tests for folder-level threshold calibration (Differentiator A)."""

import numpy as np

from rollcall.calibration import (
    SFACE_FLOOR,
    T_LOW_CAP,
    calibrate_from_exemplars,
    calibrate_from_pair_similarities,
)
from rollcall.image_utils import FaceQualityReport
from rollcall.matcher import FaceMatchInput, STATUS_CONFIRMED, STATUS_LIKELY, match_group_faces


def test_n50_synthetic_similarity_matrix_calibrated_between_floor_and_cap():
    # Build 50 unit-ish synthetic identities and simulate 1,225 pair scores
    # where the 99.5th percentile lands cleanly around ~0.41 (between 0.363 and 0.50)
    rng = np.random.default_rng(123)
    n = 50
    pair_count = n * (n - 1) // 2
    assert pair_count == 1225

    sims = rng.normal(loc=0.18, scale=0.07, size=pair_count)
    # Place top 1% tail around 0.40 - 0.44
    sims[-12:] = np.linspace(0.38, 0.44, 12)

    cal = calibrate_from_pair_similarities(n, sims)
    expected_raw = round(float(np.percentile(sims, 99.5)), 4)

    assert cal.enrolled_identities == 50
    assert cal.pair_count == 1225
    assert cal.percentile_used == "99.5th"
    assert cal.fallback_used is False
    assert cal.floor_bound is False
    assert cal.cap_hit is False
    assert cal.raw_value == expected_raw
    assert cal.final_t_low == expected_raw
    assert SFACE_FLOOR < cal.final_t_low <= T_LOW_CAP
    assert cal.warning is None
    assert len(cal.limitations) == 3


def test_n_under_15_triggers_fallback_to_sface_floor():
    # N = 10 -> 45 pairs, even with high noisy pair scores
    sims = np.linspace(0.10, 0.52, 45)
    cal = calibrate_from_pair_similarities(10, sims)

    assert cal.enrolled_identities == 10
    assert cal.pair_count == 45
    assert cal.fallback_used is True
    assert cal.final_t_low == SFACE_FLOOR
    assert cal.warning is not None
    assert "N=10 < 15" in cal.warning


def test_floor_binding_at_0363():
    # N = 50 with very clean distinct embeddings (99.5th percentile ~ 0.25 < 0.363)
    rng = np.random.default_rng(7)
    sims = rng.uniform(-0.10, 0.26, size=1225)
    cal = calibrate_from_pair_similarities(50, sims)

    assert cal.enrolled_identities == 50
    assert cal.pair_count == 1225
    assert cal.raw_value < SFACE_FLOOR
    assert cal.floor_bound is True
    assert cal.cap_hit is False
    assert cal.final_t_low == SFACE_FLOOR


def test_cap_binding_at_050_with_warning():
    # N = 50 where top 1% impostor pairs exceed 0.50 -> T_low must cap at 0.50 with warning
    sims = np.full(1225, 0.20, dtype=np.float64)
    sims[-15:] = 0.56
    cal = calibrate_from_pair_similarities(50, sims)

    assert cal.enrolled_identities == 50
    assert cal.pair_count == 1225
    assert cal.raw_value == 0.56
    assert cal.cap_hit is True
    assert cal.final_t_low == T_LOW_CAP
    assert cal.final_t_high == 0.56
    assert cal.final_t_high > cal.final_t_low
    assert cal.warning is not None
    assert "capped at 0.50" in cal.warning


def test_t_high_tracks_above_t_low_050_in_matcher():
    sims = np.full(1225, 0.20, dtype=np.float64)
    sims[-15:] = 0.54
    cal = calibrate_from_pair_similarities(50, sims)
    assert cal.final_t_low == 0.50
    assert cal.final_t_high == 0.56

    thresholds = cal.to_thresholds()
    assert thresholds.similarity_low == 0.50
    assert thresholds.similarity_high == 0.56

    e_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    # Probe 1 has cosine sim 0.52 to Identity A (>= T_low=0.50, < T_high=0.56 -> Likely)
    p1 = np.array([0.52, 0.0, np.sqrt(1.0 - 0.52**2)], dtype=np.float32)
    p2 = np.array([0.0, 0.58, np.sqrt(1.0 - 0.58**2)], dtype=np.float32)

    q_ok = FaceQualityReport(
        width_px=80,
        height_px=80,
        min_side_px=80,
        blur_laplacian=120.0,
        brightness_mean=120.0,
        detector_score=0.95,
        passed=True,
        reasons=[],
    )
    results = match_group_faces(
        faces=[
            FaceMatchInput(face_index=1, bbox=(0, 0, 80, 80), quality=q_ok, embedding=p1, crop_b64=""),
            FaceMatchInput(face_index=2, bbox=(100, 0, 80, 80), quality=q_ok, embedding=p2, crop_b64=""),
        ],
        exemplars=[
            {"identity_id": "id_a", "display_name": "Alice", "photo_id": 1, "original_filename": "Alice.jpg", "embedding": e_a},
            {"identity_id": "id_b", "display_name": "Bob", "photo_id": 2, "original_filename": "Bob.jpg", "embedding": e_b},
        ],
        thresholds=thresholds,
    )
    assert results[0].status == STATUS_LIKELY
    assert results[0].display_name == "Alice"
    assert results[1].status == STATUS_CONFIRMED
    assert results[1].display_name == "Bob"


def test_calibrate_from_exemplars_multi_photo_grouping():
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    exemplars = [
        {"identity_id": "a", "embedding": v1},
        {"identity_id": "a", "embedding": v1},
        {"identity_id": "b", "embedding": v2},
    ]
    cal = calibrate_from_exemplars(exemplars)
    # 2 distinct identities -> 1 pair (max sim = 0.0), N < 15 fallback
    assert cal.enrolled_identities == 2
    assert cal.pair_count == 1
    assert cal.raw_value == 0.0
    assert cal.fallback_used is True
    assert cal.final_t_low == SFACE_FLOOR
