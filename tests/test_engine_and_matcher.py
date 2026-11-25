"""Tests for OpenCVYuNetSFaceEngine and open-set matching logic."""

import numpy as np
import pytest

from rollcall.config import RecognitionThresholds
from rollcall.engine import ModelMissingError, OpenCVYuNetSFaceEngine
from rollcall.image_utils import FaceQualityReport
from rollcall.matcher import (
    STATUS_CONFIRMED,
    STATUS_LIKELY,
    STATUS_UNKNOWN,
    FaceMatchInput,
    match_group_faces,
)


def _unit_vec(seed: int, dim: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _good_quality() -> FaceQualityReport:
    return FaceQualityReport(
        width_px=80,
        height_px=80,
        min_side_px=80,
        blur_laplacian=120.0,
        brightness_mean=128.0,
        detector_score=0.95,
        passed=True,
        reasons=[],
    )


def _bad_quality() -> FaceQualityReport:
    return FaceQualityReport(
        width_px=18,
        height_px=18,
        min_side_px=18,
        blur_laplacian=12.0,
        brightness_mean=128.0,
        detector_score=0.90,
        passed=False,
        reasons=["Face too small (18x18 px < 28 px)"],
    )


def test_engine_initializes_from_local_models():
    engine = OpenCVYuNetSFaceEngine()
    assert "yunet" in engine.model_version
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    assert engine.detect_faces(blank) == []


def test_engine_fails_fast_when_models_missing(tmp_path):
    with pytest.raises(ModelMissingError):
        OpenCVYuNetSFaceEngine(models_dir=tmp_path)


def test_open_set_confirmed_likely_and_unknown():
    v_priya = _unit_vec(1)
    v_rahul = _unit_vec(2)
    v_stranger = _unit_vec(99)

    exemplars = [
        {
            "photo_id": 1,
            "identity_id": "priya_sharma",
            "display_name": "Priya Sharma",
            "original_filename": "Priya Sharma.jpg",
            "embedding": v_priya,
        },
        {
            "photo_id": 2,
            "identity_id": "rahul_verma",
            "display_name": "Rahul Verma",
            "original_filename": "Rahul Verma.jpg",
            "embedding": v_rahul,
        },
    ]

    # Create a borderline vector (~0.36 similarity to Rahul)
    v_borderline = 0.36 * v_rahul + np.sqrt(1 - 0.36**2) * _unit_vec(50)
    v_borderline = v_borderline / np.linalg.norm(v_borderline)

    faces = [
        FaceMatchInput(0, (0, 0, 80, 80), _good_quality(), v_priya, ""),
        FaceMatchInput(1, (100, 0, 80, 80), _good_quality(), v_borderline, ""),
        FaceMatchInput(2, (200, 0, 80, 80), _good_quality(), v_stranger, ""),
        FaceMatchInput(3, (300, 0, 18, 18), _bad_quality(), v_priya, ""),
    ]

    thresholds = RecognitionThresholds(
        similarity_high=0.45,
        similarity_low=0.30,
        runner_up_margin=0.05,
        enable_hungarian=False,
    )
    results = match_group_faces(faces, exemplars, thresholds)

    assert results[0].status == STATUS_CONFIRMED
    assert results[0].display_name == "Priya Sharma"
    assert results[1].status == STATUS_LIKELY
    assert results[1].display_name == "Rahul Verma"
    assert results[2].status == STATUS_UNKNOWN
    assert results[2].display_name == "unknown"
    assert results[3].status == STATUS_UNKNOWN  # Quality gate blocks even exact embedding
    assert results[3].display_name == "unknown"


def test_hungarian_never_upgrades_weak_or_likely_match():
    v_priya = _unit_vec(10)
    exemplars = [
        {
            "photo_id": 1,
            "identity_id": "priya_sharma",
            "display_name": "Priya Sharma",
            "original_filename": "Priya Sharma.jpg",
            "embedding": v_priya,
        }
    ]
    v_borderline = 0.35 * v_priya + np.sqrt(1 - 0.35**2) * _unit_vec(77)
    v_borderline = v_borderline / np.linalg.norm(v_borderline)

    faces = [
        FaceMatchInput(0, (0, 0, 80, 80), _good_quality(), v_priya, ""),
        FaceMatchInput(1, (100, 0, 80, 80), _good_quality(), v_borderline, ""),
    ]
    thresholds = RecognitionThresholds(
        similarity_high=0.45,
        similarity_low=0.30,
        runner_up_margin=0.05,
        enable_hungarian=True,
    )
    results = match_group_faces(faces, exemplars, thresholds)
    assert results[0].status == STATUS_CONFIRMED
    assert results[1].status == STATUS_UNKNOWN  # Collision resolved to dummy unknown


def test_unknown_below_threshold_keeps_runner_up_distinct_from_top1():
    v1 = np.zeros(128, dtype=np.float32)
    v2 = np.zeros(128, dtype=np.float32)
    v3 = np.zeros(128, dtype=np.float32)
    v1[0], v2[1], v3[2] = 1.0, 1.0, 1.0
    # Probe has sim 0.2413 to Alice (top1) and 0.1574 to Bob (top2), both below T_low=0.50 -> margin = 0.0839
    probe = 0.2413 * v1 + 0.1574 * v2 + np.sqrt(max(0.0, 1.0 - 0.2413**2 - 0.1574**2)) * v3
    probe = probe / np.linalg.norm(probe)

    exemplars = [
        {"photo_id": 1, "identity_id": "alice", "display_name": "Alice", "original_filename": "Alice.jpg", "embedding": v1},
        {"photo_id": 2, "identity_id": "bob", "display_name": "Bob", "original_filename": "Bob.jpg", "embedding": v2},
    ]
    thresholds = RecognitionThresholds(similarity_low=0.50, similarity_high=0.56)
    res = match_group_faces([FaceMatchInput(0, (0, 0, 80, 80), _good_quality(), probe, "")], exemplars, thresholds)[0]

    assert res.status == STATUS_UNKNOWN
    assert res.display_name == STATUS_UNKNOWN
    assert abs(res.similarity - 0.2413) < 1e-3
    assert res.runner_up_name == "Bob"
    assert abs(res.runner_up_similarity - 0.1574) < 1e-3
    assert abs((res.similarity - res.runner_up_similarity) - res.runner_up_margin) < 1e-3
    assert "closest: Alice" in res.decision_reason
