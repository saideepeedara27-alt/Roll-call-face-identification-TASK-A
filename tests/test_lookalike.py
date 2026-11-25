"""Tests for Lookalike Guard and Possible Duplicate Identity separation (Step 4)."""

import numpy as np

from rollcall.calibration import calibrate_from_exemplars
from rollcall.config import DEFAULT_THRESHOLDS, RecognitionThresholds
from rollcall.image_utils import FaceQualityReport
from rollcall.matcher import (
    STATUS_CONFIRMED,
    STATUS_LIKELY,
    STATUS_UNKNOWN,
    FaceMatchInput,
    match_group_faces,
)


def _unit_from(a: float, b: float, c: float) -> np.ndarray:
    v = np.zeros(128, dtype=np.float32)
    v[0], v[1], v[2] = a, b, c
    return v / np.linalg.norm(v)


def _good_quality() -> FaceQualityReport:
    return FaceQualityReport(
        width_px=90,
        height_px=90,
        min_side_px=90,
        blur_laplacian=150.0,
        brightness_mean=130.0,
        detector_score=0.95,
        passed=True,
        reasons=[],
    )


def _bad_quality() -> FaceQualityReport:
    return FaceQualityReport(
        width_px=18,
        height_px=18,
        min_side_px=18,
        blur_laplacian=10.0,
        brightness_mean=130.0,
        detector_score=0.90,
        passed=False,
        reasons=["Face too small (18x18 px < 28 px)"],
    )


def test_duplicate_vs_lookalike_pair_separation():
    # Pair A<->B has similarity ~0.60 (>= 0.55 possible duplicate)
    # Pair A<->C has similarity ~0.48 (0.42 <= sim < 0.55 lookalike)
    # Identity D is completely orthogonal
    v_a = _unit_from(1.0, 0.0, 0.0)
    v_b = _unit_from(0.60, 0.80, 0.0)
    v_c = _unit_from(0.48, 0.0, float(np.sqrt(1 - 0.48**2)))
    v_d = np.zeros(128, dtype=np.float32)
    v_d[10] = 1.0

    exemplars = [
        {"photo_id": 1, "identity_id": "id_a", "display_name": "Alice", "original_filename": "Alice.jpg", "embedding": v_a},
        {"photo_id": 2, "identity_id": "id_b", "display_name": "Alicia", "original_filename": "Alicia.jpg", "embedding": v_b},
        {"photo_id": 3, "identity_id": "id_c", "display_name": "Alina", "original_filename": "Alina.jpg", "embedding": v_c},
        {"photo_id": 4, "identity_id": "id_d", "display_name": "David", "original_filename": "David.jpg", "embedding": v_d},
    ]

    cal = calibrate_from_exemplars(exemplars)
    assert len(cal.possible_duplicate_pairs) == 1
    assert cal.possible_duplicate_pairs[0]["identity_a_name"] == "Alice"
    assert cal.possible_duplicate_pairs[0]["identity_b_name"] == "Alicia"
    assert cal.possible_duplicate_pairs[0]["category"] == "possible_duplicate"

    assert len(cal.lookalike_pairs) == 1
    assert cal.lookalike_pairs[0]["identity_a_name"] == "Alice"
    assert cal.lookalike_pairs[0]["identity_b_name"] == "Alina"
    assert cal.lookalike_pairs[0]["category"] == "lookalike"


def test_lookalike_guard_sends_ambiguous_probe_to_likely_leaving_normal_confirmed():
    # Two lookalike identities with sim ~0.48 between them
    v_a = _unit_from(1.0, 0.0, 0.0)
    v_b = _unit_from(0.48, float(np.sqrt(1 - 0.48**2)), 0.0)
    # Normal unrelated identity
    v_norm = np.zeros(128, dtype=np.float32)
    v_norm[5] = 1.0

    exemplars = [
        {"photo_id": 1, "identity_id": "id_a", "display_name": "Alice", "original_filename": "Alice.jpg", "embedding": v_a},
        {"photo_id": 2, "identity_id": "id_b", "display_name": "Alina", "original_filename": "Alina.jpg", "embedding": v_b},
        {"photo_id": 3, "identity_id": "id_norm", "display_name": "David", "original_filename": "David.jpg", "embedding": v_norm},
    ]
    cal = calibrate_from_exemplars(exemplars)
    flagged = cal.flagged_pair_set()
    assert frozenset(("id_a", "id_b")) in flagged
    assert DEFAULT_THRESHOLDS.runner_up_margin == 0.05
    assert DEFAULT_THRESHOLDS.lookalike_margin == 0.10

    # 1. Probe equally close to Alice and Alina -> margin ~0.0 -> MUST land in Likely
    v_equal = (v_a + v_b) / np.linalg.norm(v_a + v_b)
    # 2. Probe slightly closer to Alice (margin ~0.07: passes base 0.05, fails lookalike 0.10) -> Likely
    v_close = (0.55 * v_a + 0.48 * v_b)
    v_close = v_close / np.linalg.norm(v_close)
    # 3. Normal match for David (sim=0.52, runner-up ~0.0, margin=0.52 >= 0.05) -> Confirmed
    v_other = np.zeros(128, dtype=np.float32)
    v_other[6] = 1.0
    v_david_probe = 0.52 * v_norm + float(np.sqrt(1 - 0.52**2)) * v_other
    # 4. Borderline probe for David (sim=0.39 between 0.363 and 0.44) -> stays Likely
    v_borderline = 0.39 * v_norm + float(np.sqrt(1 - 0.39**2)) * v_other
    # 5. Low-quality probe with exact Alice vector -> stays unknown (quality gate never overridden)

    thresholds = RecognitionThresholds(
        similarity_high=0.44,
        similarity_low=0.363,
        runner_up_margin=0.05,
        lookalike_extra_margin=0.05,
    )
    results = match_group_faces(
        [
            FaceMatchInput(0, (0, 0, 90, 90), _good_quality(), v_equal, ""),
            FaceMatchInput(1, (100, 0, 90, 90), _good_quality(), v_close, ""),
            FaceMatchInput(2, (200, 0, 90, 90), _good_quality(), v_david_probe, ""),
            FaceMatchInput(3, (300, 0, 90, 90), _good_quality(), v_borderline, ""),
            FaceMatchInput(4, (400, 0, 18, 18), _bad_quality(), v_a, ""),
        ],
        exemplars,
        thresholds=thresholds,
        lookalike_pairs=flagged,
    )

    assert results[0].status == STATUS_LIKELY
    assert "Lookalike guard active" in results[0].decision_reason
    assert results[1].status == STATUS_LIKELY
    assert "Lookalike guard active" in results[1].decision_reason
    assert results[2].status == STATUS_CONFIRMED
    assert results[2].display_name == "David"
    assert results[3].status == STATUS_LIKELY
    assert results[4].status == STATUS_UNKNOWN


def test_lookalike_guard_applies_to_possible_duplicate_pair_over_055():
    # Two identities with >= 0.55 similarity ("possible duplicate" class)
    v_a = _unit_from(1.0, 0.0, 0.0)
    v_b = _unit_from(0.60, 0.80, 0.0)
    exemplars = [
        {"photo_id": 1, "identity_id": "id_a", "display_name": "Alice", "original_filename": "Alice.jpg", "embedding": v_a},
        {"photo_id": 2, "identity_id": "id_b", "display_name": "Alicia", "original_filename": "Alicia.jpg", "embedding": v_b},
    ]
    cal = calibrate_from_exemplars(exemplars)
    assert len(cal.possible_duplicate_pairs) == 1
    flagged = cal.flagged_pair_set()
    assert frozenset(("id_a", "id_b")) in flagged

    # Probe equally close to both identities in the >= 0.55 duplicate pair MUST land in Likely
    v_equal = (v_a + v_b) / np.linalg.norm(v_a + v_b)
    thresholds = RecognitionThresholds(
        similarity_high=0.44,
        similarity_low=0.363,
        runner_up_margin=0.05,
        lookalike_extra_margin=0.05,
    )
    results = match_group_faces(
        [FaceMatchInput(0, (0, 0, 90, 90), _good_quality(), v_equal, "")],
        exemplars,
        thresholds=thresholds,
        lookalike_pairs=flagged,
    )
    assert results[0].status == STATUS_LIKELY
    assert "Lookalike guard active" in results[0].decision_reason
