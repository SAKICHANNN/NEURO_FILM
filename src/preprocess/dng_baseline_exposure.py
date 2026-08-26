"""Private DNG baseline-exposure stage for the exact P98 raster path.

This product includes DNG technology under license by Adobe.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .dng_forward_raster import (
    DngForwardRasterError,
    load_dng_forward_working_image,
)
from .types import DecodeWarning, SourceProfile, WorkingImage


class DngBaselineExposureError(ValueError):
    """Raised when the private DNG exposure contract cannot be satisfied."""


@dataclass(frozen=True)
class DngBaselineExposureFacts:
    """Resolved standard exposure values and their provenance."""

    baseline_exposure_ev: float
    baseline_exposure_provenance: str
    baseline_exposure_offset_ev: float
    baseline_exposure_offset_provenance: str
    total_ev: float
    scale: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "baseline_exposure_ev": self.baseline_exposure_ev,
            "baseline_exposure_provenance": self.baseline_exposure_provenance,
            "baseline_exposure_offset_ev": self.baseline_exposure_offset_ev,
            "baseline_exposure_offset_provenance": self.baseline_exposure_offset_provenance,
            "scale": self.scale,
            "total_ev": self.total_ev,
        }


_BASELINE_EXPOSURE = 50730
_BASELINE_EXPOSURE_OFFSET = 51109
_DEFAULT_EV = 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_rational(value: object, *, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size != 2:
        raise DngBaselineExposureError(f"{name} must contain one rational")
    numerator = float(raw[0])
    denominator = float(raw[1])
    if denominator == 0.0:
        raise DngBaselineExposureError(f"{name} has a zero denominator")
    result = numerator / denominator
    if not np.isfinite(result):
        raise DngBaselineExposureError(f"{name} is non-finite")
    return result


def _read_baseline_exposure(path: Path) -> DngBaselineExposureFacts:
    try:
        with tifffile.TiffFile(path) as document:
            tags = document.pages[0].tags
            exposure_tag = tags.get(_BASELINE_EXPOSURE)
            offset_tag = tags.get(_BASELINE_EXPOSURE_OFFSET)
            exposure = (
                _DEFAULT_EV
                if exposure_tag is None
                else _single_rational(exposure_tag.value, name="BaselineExposure")
            )
            offset = (
                _DEFAULT_EV
                if offset_tag is None
                else _single_rational(offset_tag.value, name="BaselineExposureOffset")
            )
    except DngBaselineExposureError:
        raise
    except (OSError, TypeError, ValueError, tifffile.TiffFileError) as exc:
        raise DngBaselineExposureError(
            f"invalid or unsupported DNG exposure metadata: {exc}"
        ) from exc
    total = exposure + offset
    if not np.isfinite(total) or total < -8.0 or total > 8.0:
        raise DngBaselineExposureError(
            "total DNG baseline exposure is outside [-8, 8] EV"
        )
    scale = float(np.exp2(np.float64(total)))
    if not np.isfinite(scale):
        raise DngBaselineExposureError("DNG baseline exposure scale is non-finite")
    return DngBaselineExposureFacts(
        baseline_exposure_ev=exposure,
        baseline_exposure_provenance=(
            "dng_standard_default" if exposure_tag is None else "explicit_ifd_tag"
        ),
        baseline_exposure_offset_ev=offset,
        baseline_exposure_offset_provenance=(
            "dng_standard_default" if offset_tag is None else "explicit_ifd_tag"
        ),
        total_ev=total,
        scale=scale,
    )


def _apply_baseline_exposure(
    pixels: np.ndarray,
    scale: float,
    *,
    row_block: int = 128,
) -> np.ndarray:
    if pixels.dtype != np.float32 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise DngBaselineExposureError("pixels must be float32 HxWx3")
    if pixels.size == 0 or not np.all(np.isfinite(pixels)):
        raise DngBaselineExposureError("pixels must be non-empty and finite")
    if not np.isfinite(scale) or scale < 2.0**-8 or scale > 2.0**8:
        raise DngBaselineExposureError("scale is outside the frozen exposure range")
    if not isinstance(row_block, int) or isinstance(row_block, bool) or row_block <= 0:
        raise DngBaselineExposureError("row_block must be a positive integer")
    output = np.empty_like(pixels)
    for start in range(0, pixels.shape[0], row_block):
        stop = min(start + row_block, pixels.shape[0])
        exposed = pixels[start:stop].astype(np.float64) * scale
        if not np.all(np.isfinite(exposed)):
            raise DngBaselineExposureError(
                "exposure adjustment produced non-finite values"
            )
        output[start:stop] = exposed.astype(np.float32)
    return output


def load_dng_baseline_exposed_working_image(
    path: str | Path,
    *,
    expected_source_bytes: int | None = None,
    expected_source_sha256: str | None = None,
) -> WorkingImage:
    """Apply standard DNG baseline exposure to the private P98 raster."""

    source = Path(path)
    if source.suffix.lower() != ".dng" or not source.is_file():
        raise DngBaselineExposureError("source must be an existing .dng file")
    if (
        expected_source_bytes is not None
        and source.stat().st_size != expected_source_bytes
    ):
        raise DngBaselineExposureError("source byte count mismatch")
    before_sha = _sha256(source)
    if expected_source_sha256 is not None and before_sha != expected_source_sha256:
        raise DngBaselineExposureError("source SHA-256 mismatch")
    facts = _read_baseline_exposure(source)
    try:
        parent = load_dng_forward_working_image(
            source,
            expected_source_bytes=expected_source_bytes,
            expected_source_sha256=expected_source_sha256,
        )
    except DngForwardRasterError as exc:
        raise DngBaselineExposureError(str(exc)) from exc
    parent_frozen = parent.pixels.copy()
    pixels = _apply_baseline_exposure(parent.pixels, facts.scale)
    if not np.array_equal(parent.pixels, parent_frozen):
        raise DngBaselineExposureError("P98 parent pixels changed during exposure")
    if _sha256(source) != before_sha:
        raise DngBaselineExposureError("source bytes changed during exposure decode")
    metadata: dict[str, Any] = dict(parent.hdr_metadata)
    metadata["dng_baseline_exposure"] = facts.to_dict()
    return WorkingImage(
        pixels=pixels,
        working_space=parent.working_space,
        transfer_state=parent.transfer_state,
        source_transfer_state=parent.source_transfer_state,
        source_profile=SourceProfile(
            parent.source_profile.kind,
            f"{parent.source_profile.description}; DNG baseline exposure applied",
            parent.source_profile.bytes_length,
        ),
        hdr_metadata=metadata,
        orientation_applied=parent.orientation_applied,
        alpha_policy=parent.alpha_policy,
        bit_depth_in=parent.bit_depth_in,
        source_path=parent.source_path,
        warnings=[
            *parent.warnings,
            DecodeWarning(
                "private_dng_baseline_exposure",
                "DNG BaselineExposure and BaselineExposureOffset were applied as a scene-linear EV multiplier without a tone map.",
            ),
        ],
    )


__all__ = [
    "DngBaselineExposureError",
    "DngBaselineExposureFacts",
    "load_dng_baseline_exposed_working_image",
]
