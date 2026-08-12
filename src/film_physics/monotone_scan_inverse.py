"""Explicit profile-bound monotone scan-linear inverse."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MonotoneScanInverseV1:
    scan_knots_rgb: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    source_knots: tuple[float, ...]

    def __post_init__(self) -> None:
        source = np.asarray(self.source_knots, dtype=np.float64)
        scans = tuple(np.asarray(row, dtype=np.float64) for row in self.scan_knots_rgb)
        if (
            source.ndim != 1
            or source.size < 2
            or len(scans) != 3
            or any(row.shape != source.shape for row in scans)
            or not np.all(np.isfinite(source))
            or any(not np.all(np.isfinite(row)) for row in scans)
            or np.any(np.diff(source) <= 0.0)
            or any(np.any(np.diff(row) <= 0.0) for row in scans)
            or source[0] != 0.0
            or source[-1] != 1.0
        ):
            raise ValueError("invalid monotone scan inverse profile")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "neuro-film.monotone-scan-inverse.v1",
            "scan_knots_rgb": [list(row) for row in self.scan_knots_rgb],
            "source_knots": list(self.source_knots),
            "extrapolation_allowed": False,
            "hard_clipping_allowed": False,
        }

    def identity(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).hexdigest()

    def apply(self, scan_linear_rgb: np.ndarray) -> np.ndarray:
        values = np.asarray(scan_linear_rgb, dtype=np.float64)
        if (
            values.ndim < 1
            or values.shape[-1] != 3
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("invalid scan-linear inverse input")
        output = np.empty_like(values)
        for channel, knots in enumerate(self.scan_knots_rgb):
            x = np.asarray(knots, dtype=np.float64)
            samples = values[..., channel]
            if np.any(samples < x[0]) or np.any(samples > x[-1]):
                raise ValueError("scan-linear input requires forbidden extrapolation")
            output[..., channel] = np.interp(samples, x, self.source_knots)
        result = np.ascontiguousarray(output, dtype=np.float32)
        if np.any(result < 0.0) or np.any(result > 1.0):
            raise RuntimeError("monotone scan inverse escaped its output domain")
        return result


__all__ = ["MonotoneScanInverseV1"]
