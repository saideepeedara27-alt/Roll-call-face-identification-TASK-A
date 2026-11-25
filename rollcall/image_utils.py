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
    original_width: int = 0
    original_height: int = 0
    scale_factor: float = 1.0


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
    pose_asymmetry: float = 0.0


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

    orig_w, orig_h = rgb_img.size
    width, height = orig_w, orig_h
    was_resized = False
    scale_factor = 1.0
    if max(orig_w, orig_h) > max_dimension:
        scale = max_dimension / float(max(orig_w, orig_h))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        rgb_img = rgb_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        width, height = new_w, new_h
        was_resized = True
        scale_factor = new_w / float(max(1, orig_w))

    rgb_arr = np.asarray(rgb_img, dtype=np.uint8)
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    return LoadedImage(
        bgr=bgr_arr,
        width=width,
        height=height,
        sha256=sha256,
        was_resized=was_resized,
        original_width=orig_w,
        original_height=orig_h,
        scale_factor=scale_factor,
    )


def assess_face_quality(
    bgr_image: np.ndarray,
    bbox: tuple[int, int, int, int],
    detector_score: float,
    min_face_size_px: int,
    min_blur_laplacian: float,
    min_detector_score: float,
    size_scale: float = 1.0,
    landmarks: np.ndarray | None = None,
) -> FaceQualityReport:
    """Compute size (in original pixels when scaled), blur, lighting, pose, and confidence metrics."""
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
    orig_w_px = max(1, int(round(w / size_scale))) if 0.0 < size_scale < 1.0 else w
    orig_h_px = max(1, int(round(h / size_scale))) if 0.0 < size_scale < 1.0 else h
    min_side = min(orig_w_px, orig_h_px)

    reasons: list[str] = []
    if min_side < min_face_size_px:
        reasons.append(f"Face too small ({orig_w_px}x{orig_h_px} px < {min_face_size_px} px)")
    if blur_laplacian < min_blur_laplacian:
        reasons.append(f"Blurry face (sharpness {blur_laplacian:.1f} < {min_blur_laplacian:.1f})")
    if detector_score < 0.35:
        reasons.append(f"Very low detector score / severe occlusion ({detector_score:.2f} < 0.35)")
    if brightness_mean < 25.0:
        reasons.append(f"Severely underexposed (brightness {brightness_mean:.0f}/255)")
    elif brightness_mean > 240.0:
        reasons.append(f"Severely overexposed (brightness {brightness_mean:.0f}/255)")

    passed = len(reasons) == 0
    if 0.35 <= detector_score < min_detector_score:
        reasons.append(
            f"Low detector confidence / possible occlusion ({detector_score:.2f} < {min_detector_score:.2f})"
        )

    pose_asymmetry = 0.0
    if landmarks is not None and len(landmarks) >= 6:
        rx, ry, lx, ly, nx, ny = [float(v) for v in landmarks[:6]]
        inter_eye = float(np.hypot(lx - rx, ly - ry))
        if inter_eye >= 4.0:
            d_right = float(np.hypot(nx - rx, ny - ry))
            d_left = float(np.hypot(nx - lx, ny - ly))
            pose_asymmetry = abs(d_right - d_left) / max(d_right, d_left, 1e-6)
            eye_min_x = min(rx, lx)
            eye_max_x = max(rx, lx)
            if pose_asymmetry >= 0.42 or nx < eye_min_x or nx > eye_max_x:
                reasons.append(f"Profile angle or strong head turn (asymmetry {pose_asymmetry:.2f})")

    return FaceQualityReport(
        width_px=orig_w_px,
        height_px=orig_h_px,
        min_side_px=min_side,
        blur_laplacian=round(blur_laplacian, 2),
        brightness_mean=round(brightness_mean, 1),
        detector_score=round(detector_score, 4),
        passed=passed,
        reasons=reasons,
        pose_asymmetry=round(pose_asymmetry, 3),
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
