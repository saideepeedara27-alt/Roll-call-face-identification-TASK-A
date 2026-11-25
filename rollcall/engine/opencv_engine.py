"""OpenCV YuNet (detection) + SFace (recognition) implementation of FaceEngine."""

from pathlib import Path
import threading
import cv2
import numpy as np

from rollcall.config import (
    MODELS_DIR,
    MODEL_VERSION_TAG,
    SFACE_MODEL_NAME,
    YUNET_MODEL_NAME,
)
from rollcall.engine.base import DetectedFace, FaceEngine, ModelMissingError


def _compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / float(union) if union > 0 else 0.0


class OpenCVYuNetSFaceEngine(FaceEngine):
    """Offline YuNet + SFace engine using pre-downloaded OpenCV Zoo ONNX weights."""

    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        score_threshold: float = 0.50,
        nms_threshold: float = 0.30,
        top_k: int = 500,
    ) -> None:
        self._yunet_path = models_dir / YUNET_MODEL_NAME
        self._sface_path = models_dir / SFACE_MODEL_NAME

        missing = [p.name for p in (self._yunet_path, self._sface_path) if not p.is_file()]
        if missing:
            raise ModelMissingError(
                f"Missing required offline model file(s): {', '.join(missing)} in {models_dir}. "
                "Run `python scripts/setup_models.py` while connected to the internet first."
            )

        self._default_score_threshold = score_threshold
        self._lock = threading.Lock()
        self._detector = cv2.FaceDetectorYN.create(
            str(self._yunet_path),
            "",
            (320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(self._sface_path), "")

    @property
    def model_version(self) -> str:
        return MODEL_VERSION_TAG

    def detect_faces(
        self, bgr_image: np.ndarray, score_threshold: float | None = None
    ) -> list[DetectedFace]:
        if bgr_image is None or bgr_image.size == 0:
            return []

        height, width = bgr_image.shape[:2]
        with self._lock:
            if score_threshold is not None:
                self._detector.setScoreThreshold(float(score_threshold))
            try:
                self._detector.setInputSize((width, height))
                _, raw_faces = self._detector.detect(bgr_image)
            finally:
                if score_threshold is not None:
                    self._detector.setScoreThreshold(float(self._default_score_threshold))

        if raw_faces is None or len(raw_faces) == 0:
            return []

        candidates: list[DetectedFace] = []
        for row in raw_faces:
            x, y, w, h = (int(round(v)) for v in row[:4])
            score = float(row[14])
            # Clamp bounding box to image bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + max(1, w))
            y2 = min(height, y + max(1, h))
            cw = max(1, x2 - x1)
            ch = max(1, y2 - y1)
            candidates.append(
                DetectedFace(
                    bbox=(x1, y1, cw, ch),
                    score=score,
                    raw_row=np.array(row, dtype=np.float32),
                )
            )

        # Sort by confidence descending and deduplicate overlapping boxes (IoU > 0.45)
        candidates.sort(key=lambda f: f.score, reverse=True)
        deduped: list[DetectedFace] = []
        for cand in candidates:
            if any(_compute_iou(cand.bbox, kept.bbox) > 0.45 for kept in deduped):
                continue
            deduped.append(cand)

        # Order left-to-right for predictable UI numbering
        deduped.sort(key=lambda f: (f.bbox[0], f.bbox[1]))
        return deduped

    def extract_embedding(
        self, bgr_image: np.ndarray, face: DetectedFace
    ) -> np.ndarray:
        with self._lock:
            aligned = self._recognizer.alignCrop(bgr_image, face.raw_row)
            feat = self._recognizer.feature(aligned)

        vec = np.asarray(feat, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return vec
