"""Abstract base interface for modular face detection and recognition engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


class ModelMissingError(RuntimeError):
    """Raised when required local model weights are missing."""


@dataclass(frozen=True)
class DetectedFace:
    """Single detected face with bounding box, landmarks, and detector confidence."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    score: float
    raw_row: np.ndarray  # Engine-specific landmark/detection row for alignment


class FaceEngine(ABC):
    """Pluggable face detection + alignment + embedding interface."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Unique identifier for the detector + recognizer model pair."""

    @abstractmethod
    def detect_faces(self, bgr_image: np.ndarray) -> list[DetectedFace]:
        """Detect all faces in a BGR image."""

    @abstractmethod
    def extract_embedding(
        self, bgr_image: np.ndarray, face: DetectedFace
    ) -> np.ndarray:
        """Align the detected face and return a 1D L2-normalized float32 embedding."""

    @staticmethod
    def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Compute cosine similarity between two L2-normalized embeddings."""
        denom = float(np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(emb_a, emb_b) / denom)
