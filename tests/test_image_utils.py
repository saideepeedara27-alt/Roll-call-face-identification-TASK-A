"""Tests for image loading, EXIF normalization, resizing, and quality diagnostics."""

from io import BytesIO
import numpy as np
import pytest
from PIL import Image

from rollcall.image_utils import (
    ImageLoadError,
    assess_face_quality,
    load_image_from_bytes,
)


def _make_jpeg_bytes(width: int, height: int, color=(120, 140, 160)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_load_valid_image_and_resize():
    raw = _make_jpeg_bytes(3200, 1600)
    loaded = load_image_from_bytes(raw, "group.jpg", max_dimension=1600)
    assert loaded.was_resized is True
    assert loaded.width == 1600
    assert loaded.height == 800
    assert loaded.bgr.shape == (800, 1600, 3)


def test_reject_unsupported_extension():
    with pytest.raises(ImageLoadError):
        load_image_from_bytes(b"123", "notes.txt")


def test_reject_corrupt_image():
    with pytest.raises(ImageLoadError):
        load_image_from_bytes(b"not-an-image", "broken.jpg")


def test_quality_gate_rejects_small_and_flat_face():
    blank = np.full((100, 100, 3), 128, dtype=np.uint8)
    report = assess_face_quality(
        bgr_image=blank,
        bbox=(10, 10, 20, 20),
        detector_score=0.90,
        min_face_size_px=28,
        min_blur_laplacian=35.0,
        min_detector_score=0.65,
    )
    assert report.passed is False
    assert any("too small" in r.lower() for r in report.reasons)
    assert any("blurry" in r.lower() for r in report.reasons)
