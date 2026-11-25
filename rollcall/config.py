"""Centralized configuration and paths for RollCall."""

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
PRACTICE_DATA_DIR = DATA_DIR / "practice"
RUNTIME_DIR = BASE_DIR / "runtime_data"
DB_PATH = RUNTIME_DIR / "rollcall.db"

YUNET_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_NAME = "face_recognition_sface_2021dec.onnx"
MODEL_VERSION_TAG = "yunet_2023mar+sface_2021dec"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB per file
MAX_IMAGE_DIMENSION = 2400  # Cap longest side to protect 8 GB unified RAM


@dataclass(frozen=True)
class RecognitionThresholds:
    """Operating thresholds for quality gating and open-set matching.

    Note: Initial values are starting hypotheses; final operating points
    must be calibrated against validation data in Checkpoint 5.
    """

    similarity_high: float = 0.42
    similarity_low: float = 0.30
    runner_up_margin: float = 0.05
    min_face_size_px: int = 28
    min_blur_laplacian: float = 35.0
    min_detector_score: float = 0.65
    enable_hungarian: bool = False


DEFAULT_THRESHOLDS = RecognitionThresholds()
