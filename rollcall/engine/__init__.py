"""Pluggable face detection and recognition engine package."""

from rollcall.engine.base import DetectedFace, FaceEngine, ModelMissingError
from rollcall.engine.opencv_engine import OpenCVYuNetSFaceEngine

__all__ = [
    "DetectedFace",
    "FaceEngine",
    "ModelMissingError",
    "OpenCVYuNetSFaceEngine",
]
