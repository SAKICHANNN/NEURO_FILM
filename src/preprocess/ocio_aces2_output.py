"""Isolated official OpenColorIO ACES 2 output-transform runtime."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .types import WorkingImage

OCIO_VERSION = "2.5.2"
CONFIG_URI = "ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5"
CONFIG_CACHE_ID = "351c1452fc2bde6177841947c5fce086:6001c324468d497f99aa06d3014798d8"
SOURCE_SPACE = "ACEScg"

OutputTarget = Literal["sdr_rec709", "hdr_rec2020_pq"]

_TARGETS = {
    "sdr_rec709": (
        "sRGB - Display",
        "ACES 2.0 - SDR 100 nits (Rec.709)",
    ),
    "hdr_rec2020_pq": (
        "Rec.2100-PQ - Display",
        "ACES 2.0 - HDR 1000 nits (Rec.2020)",
    ),
}

_WORKING_SOURCE_SPACES = {
    "linear_srgb": "Linear Rec.709 (sRGB)",
    "linear_rec2020": "Linear Rec.2020",
}


class OcioAces2RuntimeError(RuntimeError):
    """Raised when the pinned OCIO/ACES 2 runtime contract is unavailable."""


def _ocio():
    try:
        import PyOpenColorIO as ocio
    except ImportError as exc:
        raise OcioAces2RuntimeError("opencolorio==2.5.2 is required") from exc
    if ocio.__version__ != OCIO_VERSION:
        raise OcioAces2RuntimeError(
            f"OpenColorIO version drift: expected {OCIO_VERSION}, got {ocio.__version__}"
        )
    return ocio


def load_aces2_config():
    """Load and validate the exact built-in ACES 2 CG config."""

    ocio = _ocio()
    try:
        config = ocio.Config.CreateFromFile(CONFIG_URI)
        config.validate()
    except Exception as exc:  # pragma: no cover - provider-specific exception hierarchy
        raise OcioAces2RuntimeError(f"unable to load pinned ACES 2 config: {exc}") from exc
    if config.getCacheID() != CONFIG_CACHE_ID:
        raise OcioAces2RuntimeError("built-in ACES 2 config cache identity drift")
    if config.getColorSpace(SOURCE_SPACE) is None:
        raise OcioAces2RuntimeError("pinned ACEScg source space is unavailable")
    for display, view in _TARGETS.values():
        if display not in list(config.getDisplays()) or view not in list(config.getViews(display)):
            raise OcioAces2RuntimeError(f"pinned display/view unavailable: {display} / {view}")
    return config


def build_aces2_numeric_fixture() -> np.ndarray:
    """Return the frozen 986-row float32 ACEScg cube plus neutral ramp."""

    levels = np.asarray([-0.125, 0.0, 0.001, 0.01, 0.18, 1.0, 4.0, 16.0, 64.0], dtype=np.float32)
    cube = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1).reshape(-1, 3)
    neutral_values = np.linspace(-0.125, 64.0, 257, dtype=np.float32)
    neutral = np.repeat(neutral_values[:, None], 3, axis=1)
    return np.ascontiguousarray(np.concatenate((cube, neutral), axis=0), dtype=np.float32)


def _processor(target: OutputTarget):
    if target not in _TARGETS:
        raise OcioAces2RuntimeError(f"unsupported ACES 2 output target: {target}")
    ocio = _ocio()
    config = load_aces2_config()
    display, view = _TARGETS[target]
    transform = ocio.DisplayViewTransform(src=SOURCE_SPACE, display=display, view=view)
    return config.getProcessor(transform).getDefaultCPUProcessor()


def apply_aces2_output_scalar(pixels: np.ndarray, target: OutputTarget) -> np.ndarray:
    """Apply the official CPU processor one RGB triplet at a time."""

    source = _validated_pixels(pixels)
    processor = _processor(target)
    output = np.empty_like(source)
    for index, row in enumerate(source):
        output[index] = processor.applyRGB([float(row[0]), float(row[1]), float(row[2])])
    if not np.isfinite(output).all():
        raise OcioAces2RuntimeError("official scalar ACES 2 output is non-finite")
    return output


def apply_aces2_output_packed(pixels: np.ndarray, target: OutputTarget) -> np.ndarray:
    """Apply the same official processor through PackedImageDesc."""

    source = _validated_pixels(pixels)
    output = np.ascontiguousarray(source.copy())
    ocio = _ocio()
    descriptor = ocio.PackedImageDesc(output, output.shape[0], 1, 3)
    _processor(target).apply(descriptor)
    if not np.isfinite(output).all():
        raise OcioAces2RuntimeError("official packed ACES 2 output is non-finite")
    return output


def convert_working_image_to_acescg(working: WorkingImage) -> np.ndarray:
    """Convert one supported scene-linear WorkingImage to ACEScg pixels."""

    if not isinstance(working, WorkingImage):
        raise OcioAces2RuntimeError("working must be a WorkingImage")
    if working.transfer_state != "scene_linear":
        raise OcioAces2RuntimeError("ACES 2 adapter requires scene_linear input")
    try:
        source_space = _WORKING_SOURCE_SPACES[working.working_space]
    except KeyError as exc:
        raise OcioAces2RuntimeError(
            "ACES 2 adapter requires linear_srgb or linear_rec2020 input"
        ) from exc
    source = np.ascontiguousarray(working.pixels.reshape(-1, 3).copy())
    ocio = _ocio()
    config = load_aces2_config()
    try:
        processor = config.getProcessor(source_space, SOURCE_SPACE).getDefaultCPUProcessor()
        descriptor = ocio.PackedImageDesc(source, source.shape[0], 1, 3)
        processor.apply(descriptor)
    except Exception as exc:  # pragma: no cover - provider-specific exception hierarchy
        raise OcioAces2RuntimeError(
            f"unable to convert {working.working_space} to ACEScg: {exc}"
        ) from exc
    if not np.isfinite(source).all():
        raise OcioAces2RuntimeError("official WorkingImage to ACEScg output is non-finite")
    return source.reshape(working.pixels.shape)


def apply_working_image_aces2_output(
    working: WorkingImage, target: OutputTarget
) -> np.ndarray:
    """Apply the pinned official ACES 2 output to an existing WorkingImage."""

    acescg = convert_working_image_to_acescg(working)
    output = apply_aces2_output_packed(acescg.reshape(-1, 3), target)
    return np.ascontiguousarray(output.reshape(working.pixels.shape))


def target_display_view(target: OutputTarget) -> tuple[str, str]:
    """Return the exact display/view pair for receipt reporting."""

    try:
        return _TARGETS[target]
    except KeyError as exc:
        raise OcioAces2RuntimeError(f"unsupported ACES 2 output target: {target}") from exc


def _validated_pixels(pixels: np.ndarray) -> np.ndarray:
    value = np.asarray(pixels)
    if value.dtype != np.float32 or value.ndim != 2 or value.shape[1:] != (3,):
        raise OcioAces2RuntimeError("ACES 2 numeric input must be finite Nx3 float32")
    if value.shape[0] == 0 or not np.isfinite(value).all():
        raise OcioAces2RuntimeError("ACES 2 numeric input must be finite and non-empty")
    return value
