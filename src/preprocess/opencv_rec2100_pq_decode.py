"""Strictly validated private OpenCV RGB16 Rec.2100-PQ decode adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np

from .png_stream import sha256_rec2100_pq_rgb16_png_samples

DecodeMode = Literal["file", "memory"]


class OpenCvRec2100PqDecodeError(RuntimeError):
    """Raised when strict semantics or OpenCV RGB16 decode differs."""


def decode_validated_rec2100_pq_rgb16_png_opencv_v1(
    path: Path,
    *,
    width: int,
    height: int,
    mode: DecodeMode,
) -> tuple[np.ndarray, str]:
    """Validate cICP/PNG structure first, then decode exact RGB16 with OpenCV."""

    source = Path(path)
    if mode not in {"file", "memory"}:
        raise OpenCvRec2100PqDecodeError("mode must be file or memory")
    try:
        strict_sample_sha256 = sha256_rec2100_pq_rgb16_png_samples(
            source, width=width, height=height
        )
    except (OSError, ValueError) as exc:
        raise OpenCvRec2100PqDecodeError(
            f"strict Rec.2100-PQ validation failed: {exc}"
        ) from exc

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - fixed audit runtime
        raise OpenCvRec2100PqDecodeError("OpenCV runtime is unavailable") from exc
    if cv2.__version__ != "4.13.0":
        raise OpenCvRec2100PqDecodeError("OpenCV runtime identity differs")
    if mode == "file":
        decoded = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    else:
        encoded = np.frombuffer(source.read_bytes(), dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if (
        decoded is None
        or decoded.dtype != np.uint16
        or decoded.shape != (height, width, 3)
    ):
        raise OpenCvRec2100PqDecodeError(
            "OpenCV decode must be exact uint16 HxWx3 BGR"
        )
    rgb = np.ascontiguousarray(decoded[:, :, ::-1])
    if hashlib.sha256(rgb.tobytes()).hexdigest() != strict_sample_sha256:
        raise OpenCvRec2100PqDecodeError(
            "OpenCV RGB16 samples differ from strict PNG samples"
        )
    return rgb, strict_sample_sha256


__all__ = [
    "DecodeMode",
    "OpenCvRec2100PqDecodeError",
    "decode_validated_rec2100_pq_rgb16_png_opencv_v1",
]
