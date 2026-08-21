"""Opt-in DNG ForwardMatrix raster decode to linear Rec.2020.

This product includes DNG technology under license by Adobe.

The module is deliberately private and narrow.  It combines a camera-linear
LibRaw demosaic with the P94 DNG ForwardMatrix construction and the P97
D50-PCS-to-linear-Rec.2020 primitive.  It is not used by the default loader.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .dng_forward_matrix import build_dual_illuminant_camera_to_pcs
from .prophoto_icc import d50_xyz_to_linear_rec2020
from .types import DecodeWarning, SourceProfile, WorkingImage


class DngForwardRasterError(ValueError):
    """Raised when the opt-in DNG raster contract cannot be satisfied."""


_TAGS = {
    "color_matrix1": 50721,
    "color_matrix2": 50722,
    "camera_calibration1": 50723,
    "camera_calibration2": 50724,
    "analog_balance": 50727,
    "as_shot_neutral": 50728,
    "calibration_illuminant1": 50778,
    "calibration_illuminant2": 50779,
    "camera_calibration_signature": 50931,
    "profile_calibration_signature": 50932,
    "forward_matrix1": 50964,
    "forward_matrix2": 50965,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rational_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size % 2:
        raise DngForwardRasterError(f"{name} is not an interleaved rational array")
    numerator = raw[0::2].astype(np.float64)
    denominator = raw[1::2].astype(np.float64)
    if np.any(denominator == 0.0):
        raise DngForwardRasterError(f"{name} has a zero denominator")
    result = numerator / denominator
    if not np.all(np.isfinite(result)):
        raise DngForwardRasterError(f"{name} is non-finite")
    return result


def _read_profile_tags(path: Path) -> dict[str, Any]:
    try:
        with tifffile.TiffFile(path) as document:
            tags = document.pages[0].tags
            required = (
                "color_matrix1",
                "color_matrix2",
                "as_shot_neutral",
                "calibration_illuminant1",
                "calibration_illuminant2",
                "forward_matrix1",
                "forward_matrix2",
            )
            missing = [name for name in required if _TAGS[name] not in tags]
            if missing:
                raise DngForwardRasterError(
                    f"missing required DNG profile tags: {', '.join(missing)}"
                )
            signatures = []
            for name in (
                "camera_calibration_signature",
                "profile_calibration_signature",
            ):
                code = _TAGS[name]
                signatures.append("" if code not in tags else str(tags[code].value))
            if signatures[0] != signatures[1]:
                raise DngForwardRasterError("DNG calibration signatures do not match")

            values: dict[str, Any] = {}
            for name in (
                "color_matrix1",
                "color_matrix2",
                "as_shot_neutral",
                "forward_matrix1",
                "forward_matrix2",
            ):
                values[name] = _rational_array(tags[_TAGS[name]].value, name=name)
            for name in ("calibration_illuminant1", "calibration_illuminant2"):
                values[name] = int(tags[_TAGS[name]].value)
            for name in ("camera_calibration1", "camera_calibration2"):
                code = _TAGS[name]
                values[name] = (
                    _rational_array(tags[code].value, name=name)
                    if code in tags
                    else None
                )
            code = _TAGS["analog_balance"]
            values["analog_balance"] = (
                _rational_array(tags[code].value, name="analog_balance")
                if code in tags
                else None
            )
    except DngForwardRasterError:
        raise
    except (OSError, TypeError, ValueError, tifffile.TiffFileError) as exc:
        raise DngForwardRasterError(
            f"invalid or unsupported DNG profile: {exc}"
        ) from exc

    for name in (
        "color_matrix1",
        "color_matrix2",
        "camera_calibration1",
        "camera_calibration2",
        "forward_matrix1",
        "forward_matrix2",
    ):
        if values[name] is not None:
            values[name] = values[name].reshape(3, 3)
    return values


def _decode_camera_linear_dng(path: Path) -> np.ndarray:
    try:
        import rawpy
    except Exception as exc:  # pragma: no cover - environment failure
        raise DngForwardRasterError("rawpy is required for DNG raster decode") from exc
    try:
        with rawpy.imread(str(path)) as raw:
            camera = raw.postprocess(
                use_camera_wb=False,
                use_auto_wb=False,
                user_wb=[1.0, 1.0, 1.0, 1.0],
                no_auto_bright=True,
                output_bps=16,
                output_color=rawpy.ColorSpace.raw,
                gamma=(1.0, 1.0),
                user_flip=None,
            )
    except Exception as exc:
        raise DngForwardRasterError(
            f"LibRaw camera-raster decode failed: {exc}"
        ) from exc
    if camera.dtype != np.uint16 or camera.ndim != 3 or camera.shape[2] != 3:
        raise DngForwardRasterError("camera raster must be non-empty uint16 HxWx3")
    if camera.size == 0:
        raise DngForwardRasterError("camera raster is empty")
    return camera


def _apply_camera_to_rec2020(
    camera: np.ndarray,
    camera_to_pcs: np.ndarray,
    *,
    row_block: int = 128,
) -> np.ndarray:
    if camera.dtype != np.uint16 or camera.ndim != 3 or camera.shape[2] != 3:
        raise DngForwardRasterError("camera must be uint16 HxWx3")
    matrix = np.asarray(camera_to_pcs, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise DngForwardRasterError("camera_to_pcs must be a finite 3x3 matrix")
    if not isinstance(row_block, int) or isinstance(row_block, bool) or row_block <= 0:
        raise DngForwardRasterError("row_block must be a positive integer")
    output = np.empty(camera.shape, dtype=np.float32)
    for start in range(0, camera.shape[0], row_block):
        stop = min(start + row_block, camera.shape[0])
        normalized = camera[start:stop].astype(np.float64) / 65535.0
        pcs = normalized @ matrix.T
        converted = d50_xyz_to_linear_rec2020(pcs)
        if not np.all(np.isfinite(converted)):
            raise DngForwardRasterError(
                "DNG colour conversion produced non-finite values"
            )
        output[start:stop] = converted.astype(np.float32)
    return output


def load_dng_forward_working_image(
    path: str | Path,
    *,
    expected_source_bytes: int | None = None,
    expected_source_sha256: str | None = None,
) -> WorkingImage:
    """Decode one exact three-channel DNG through the opt-in ForwardMatrix path."""

    source = Path(path)
    if source.suffix.lower() != ".dng" or not source.is_file():
        raise DngForwardRasterError("source must be an existing .dng file")
    if (
        expected_source_bytes is not None
        and source.stat().st_size != expected_source_bytes
    ):
        raise DngForwardRasterError("source byte count mismatch")
    before_sha = _sha256(source)
    if expected_source_sha256 is not None and before_sha != expected_source_sha256:
        raise DngForwardRasterError("source SHA-256 mismatch")

    values = _read_profile_tags(source)
    camera_to_pcs = build_dual_illuminant_camera_to_pcs(**values).camera_to_pcs
    camera = _decode_camera_linear_dng(source)
    pixels = _apply_camera_to_rec2020(camera, camera_to_pcs)
    if _sha256(source) != before_sha:
        raise DngForwardRasterError("source bytes changed during decode")
    return WorkingImage(
        pixels=pixels,
        working_space="linear_rec2020",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile(
            "raw_metadata",
            "Opt-in DNG ForwardMatrix camera-to-D50-PCS to linear Rec.2020",
        ),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=source,
        warnings=[
            DecodeWarning(
                "private_dng_forward_raster",
                "Private exact-cohort DNG ForwardMatrix path; arbitrary DNG support is not established.",
            ),
            DecodeWarning(
                "unqualified_demosaic_and_calibration",
                "LibRaw demosaic and DNG profile mechanics are not vendor rendering, sensor calibration, or photographic-quality evidence.",
            ),
            DecodeWarning(
                "scene_linear_no_tone_map",
                "Linear Rec.2020 values are retained without clipping, gamut mapping, or a scene-to-display tone map.",
            ),
        ],
    )


__all__ = [
    "DngForwardRasterError",
    "load_dng_forward_working_image",
]
