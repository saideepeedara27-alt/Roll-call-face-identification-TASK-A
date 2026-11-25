"""Image loading, EXIF normalization, safe resizing, and quality diagnostics."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from rollcall.config import ALLOWED_EXTENSIONS, MAX_IMAGE_DIMENSION, MAX_UPLOAD_BYTES


class ImageLoadError(ValueError):
    """Raised when an uploaded image is invalid, unsupported, or corrupt."""


@dataclass(frozen=True)
class LoadedImage:
    """Normalized BGR image ready for OpenCV detection and encoding."""

    bgr: np.ndarray
    width: int
    height: int
    sha256: str
    was_resized: bool


@dataclass(frozen=True)
class FaceQualityReport:
    """Objective quality measurements for a detected face."""

    width_px: int
    height_px: int
    min_side_px: int
    blur_laplacian: float
    brightness_mean: float
    detector_score: float
    passed: bool
    reasons: list[str]


def validate_image_filename(filename: str) -> None:
    """Ensure the filename has a supported image extension."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported image format '{ext or 'none'}' for '{Path(filename).name}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )


def load_image_from_bytes(
    raw_bytes: bytes,
    filename: str = "image.jpg",
    max_dimension: int = MAX_IMAGE_DIMENSION,
) -> LoadedImage:
    """Load image bytes with EXIF orientation correction and memory-safe resizing."""
    validate_image_filename(filename)
    if not raw_bytes:
        raise ImageLoadError(f"File '{Path(filename).name}' is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        mb = len(raw_bytes) / (1024 * 1024)
        raise ImageLoadError(
            f"File '{Path(filename).name}' ({mb:.1f} MB) exceeds the 25 MB limit."
        )

    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    try:
        with Image.open(BytesIO(raw_bytes)) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            rgb_img = pil_img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(
            f"Could not decode '{Path(filename).name}' as a valid image."
        ) from exc

    width, height = rgb_img.size
    was_resized = False
    if max(width, height) > max_dimension:
        scale = max_dimension / float(max(width, height))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        rgb_img = rgb_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        width, height = new_w, new_h
        was_resized = True

    rgb_arr = np.asarray(rgb_img, dtype=np.uint8)
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    return LoadedImage(
        bgr=bgr_arr,
        width=width,
        height=height,
        sha256=sha256,
        was_resized=was_resized,
    )


def assess_face_quality(
    bgr_image: np.ndarray,
    bbox: tuple[int, int, int, int],
    detector_score: float,
    min_face_size_px: int,
    min_blur_laplacian: float,
    min_detector_score: float,
) -> FaceQualityReport:
    """Compute size, blur, lighting, and detector confidence metrics for a face."""
    x, y, w, h = bbox
    img_h, img_w = bgr_image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + max(1, w))
    y2 = min(img_h, y + max(1, h))

    crop = bgr_image[y1:y2, x1:x2]
    if crop.size == 0:
        return FaceQualityReport(
            width_px=0,
            height_px=0,
            min_side_px=0,
            blur_laplacian=0.0,
            brightness_mean=0.0,
            detector_score=detector_score,
            passed=False,
            reasons=["Empty face crop"],
        )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_mean = float(gray.mean())
    min_side = min(w, h)

    reasons: list[str] = []
    if min_side < min_face_size_px:
        reasons.append(f"Face too small ({w}x{h} px < {min_face_size_px} px)")
    if blur_laplacian < min_blur_laplacian:
        reasons.append(f"Blurry face (sharpness {blur_laplacian:.1f} < {min_blur_laplacian:.1f})")
    if detector_score < 0.35:
        reasons.append(f"Very low detector score ({detector_score:.2f} < 0.35)")
    if brightness_mean < 25.0:
        reasons.append(f"Severely underexposed (brightness {brightness_mean:.0f}/255)")
    elif brightness_mean > 240.0:
        reasons.append(f"Severely overexposed (brightness {brightness_mean:.0f}/255)")

    passed = len(reasons) == 0
    if detector_score < min_detector_score and detector_score >= 0.35:
        reasons.append(f"low detector score ({detector_score:.2f} < {min_detector_score:.2f})")

    return FaceQualityReport(
        width_px=w,
        height_px=h,
        min_side_px=min_side,
        blur_laplacian=round(blur_laplacian, 2),
        brightness_mean=round(brightness_mean, 1),
        detector_score=round(detector_score, 4),
        passed=passed,
        reasons=reasons,
    )


def encode_jpeg_bytes(bgr_image: np.ndarray, quality: int = 90) -> bytes:
    """Encode a BGR NumPy image into JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ImageLoadError("Failed to encode image crop as JPEG.")
    return buf.tobytes()


def crop_face_with_margin(
    bgr_image: np.ndarray, bbox: tuple[int, int, int, int], margin_ratio: float = 0.25
) -> np.ndarray:
    """Return a padded face crop for UI explainability display."""
    x, y, w, h = bbox
    img_h, img_w = bgr_image.shape[:2]
    pad_x = int(w * margin_ratio)
    pad_y = int(h * margin_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)
    return bgr_image[y1:y2, x1:x2]
