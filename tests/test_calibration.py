"""Tests for folder-level threshold calibration (Differentiator A)."""

import numpy as np

from rollcall.calibration import (
    SFACE_FLOOR,
    T_LOW_CAP,
    calibrate_from_exemplars,
    calibrate_from_pair_similarities,
)


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
    assert cal.warning is not None
    assert "capped at 0.50" in cal.warning


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
