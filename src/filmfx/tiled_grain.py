"""Exact legacy-grain staging with bounded RAM and private disk scratch."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.inference.tiled_render import TiledExecutionMetadata, plan_tile_windows

from .effects import luminance
from .fast_blur import gaussian_filter_safe
from .layers import FilmLayer


GRAIN_BLUR_SIGMA = 1.2
GRAIN_BLUR_HALO = 4


@dataclass(frozen=True)
class StagedGrainMetadata:
    """Auditable topology and temporary-storage bounds for one grain layer."""

    tiled: TiledExecutionMetadata
    row_chunk: int
    color: bool
    raw_bytes: int
    highpass_bytes: int
    peak_scratch_bytes: int
    scratch_files_remaining: int


def _integer(value: object, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _strength(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("strength must be a finite number in [0, 1]")
    resolved = float(value)
    if not np.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError("strength must be a finite number in [0, 1]")
    return resolved


def _scratch_root(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise ValueError("scratch_root must be an existing directory path")
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise ValueError("scratch_root must be an existing directory")
    return resolved


def _close_memmap(array: np.memmap | None) -> None:
    if array is None:
        return
    array.flush()
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def staged_grain_residual_layer(
    base_rgb: np.ndarray,
    *,
    scratch_root: str | Path,
    scratch_budget_bytes: int,
    tile_size: int,
    row_chunk: int,
    strength: float = 0.018,
    seed: int = 7,
    color: bool = True,
    name: str = "grain",
) -> tuple[FilmLayer, StagedGrainMetadata]:
    """Reproduce ``grain_residual_layer`` using temporary full-layout staging."""

    if not isinstance(base_rgb, np.ndarray) or base_rgb.ndim != 3 or base_rgb.shape[2] != 3:
        raise ValueError("base_rgb must be an HWC float32 RGB array")
    if any(size <= 0 for size in base_rgb.shape):
        raise ValueError("base_rgb dimensions must be non-zero")
    if base_rgb.dtype != np.float32 or not np.isfinite(base_rgb).all():
        raise ValueError("base_rgb must be a finite float32 array")
    resolved_strength = _strength(strength)
    resolved_seed = _integer(seed, "seed", 0, 2**63 - 1)
    resolved_tile_size = _integer(tile_size, "tile_size", 1, 2**31 - 1)
    resolved_row_chunk = _integer(row_chunk, "row_chunk", 1, 2**31 - 1)
    if not isinstance(color, bool):
        raise ValueError("color must be boolean")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    resolved_root = _scratch_root(scratch_root)

    shape = tuple(int(size) for size in base_rgb.shape)
    field_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    peak_scratch_bytes = 2 * field_bytes
    resolved_budget = _integer(scratch_budget_bytes, "scratch_budget_bytes", 0, 2**63 - 1)
    if resolved_budget < peak_scratch_bytes:
        raise ValueError(
            f"scratch_budget_bytes {resolved_budget} is below required peak {peak_scratch_bytes}"
        )

    windows = plan_tile_windows(
        shape[0],
        shape[1],
        tile_size=resolved_tile_size,
        halo=GRAIN_BLUR_HALO,
    )
    max_height = max(window.expanded_shape[0] for window in windows)
    max_width = max(window.expanded_shape[1] for window in windows)
    tiled = TiledExecutionMetadata(
        input_shape=shape,
        tile_size=resolved_tile_size,
        halo=GRAIN_BLUR_HALO,
        tile_count=len(windows),
        max_expanded_shape=(max_height, max_width, 3),
    )

    raw: np.memmap | None = None
    highpass: np.memmap | None = None
    residual: np.ndarray | None = None
    with tempfile.TemporaryDirectory(prefix="neuro_film_grain_", dir=resolved_root) as temporary:
        temporary_path = Path(temporary)
        raw_path = temporary_path / "raw.f32"
        highpass_path = temporary_path / "highpass.f32"
        try:
            raw = np.memmap(raw_path, dtype=np.float32, mode="w+", shape=shape)
            rng = np.random.default_rng(resolved_seed)
            channels = 3 if color else 1
            for y0 in range(0, shape[0], resolved_row_chunk):
                y1 = min(y0 + resolved_row_chunk, shape[0])
                noise = rng.normal(0.0, 1.0, size=(y1 - y0, shape[1], channels)).astype(np.float32)
                if not color:
                    noise = np.repeat(noise, 3, axis=2)
                raw[y0:y1] = noise
            raw.flush()

            highpass = np.memmap(highpass_path, dtype=np.float32, mode="w+", shape=shape)
            for window in windows:
                tile = raw[
                    window.expanded_y0 : window.expanded_y1,
                    window.expanded_x0 : window.expanded_x1,
                    :,
                ].view()
                tile.setflags(write=False)
                blurred = gaussian_filter_safe(tile, sigma=(GRAIN_BLUR_SIGMA, GRAIN_BLUR_SIGMA, 0.0))
                local_y0, local_y1, local_x0, local_x1 = window.core_offsets
                highpass[
                    window.core_y0 : window.core_y1,
                    window.core_x0 : window.core_x1,
                    :,
                ] = tile[local_y0:local_y1, local_x0:local_x1, :] - blurred[
                    local_y0:local_y1, local_x0:local_x1, :
                ]
            highpass.flush()
            _close_memmap(raw)
            raw = None
            raw_path.unlink()

            channel_mean = highpass.mean(axis=(0, 1), keepdims=True)
            for y0 in range(0, shape[0], resolved_row_chunk):
                y1 = min(y0 + resolved_row_chunk, shape[0])
                highpass[y0:y1] = highpass[y0:y1] - channel_mean
            highpass.flush()

            standard_deviation = max(float(highpass.std()), 1e-6)
            envelope = 0.45 + 0.75 * (1.0 - luminance(base_rgb))
            for y0 in range(0, shape[0], resolved_row_chunk):
                y1 = min(y0 + resolved_row_chunk, shape[0])
                highpass[y0:y1] = (
                    highpass[y0:y1]
                    / standard_deviation
                    * envelope[y0:y1, :, None]
                    * resolved_strength
                )
            highpass.flush()

            residual_mean = highpass.mean(axis=(0, 1), keepdims=True)
            for y0 in range(0, shape[0], resolved_row_chunk):
                y1 = min(y0 + resolved_row_chunk, shape[0])
                highpass[y0:y1] = highpass[y0:y1] - residual_mean
            highpass.flush()
            residual = np.array(highpass, dtype=np.float32, copy=True)
        finally:
            _close_memmap(raw)
            _close_memmap(highpass)
            raw = None
            highpass = None

    if residual is None:
        raise RuntimeError("grain staging ended without a residual")
    return FilmLayer(name=name, mode="residual", residual=residual), StagedGrainMetadata(
        tiled=tiled,
        row_chunk=resolved_row_chunk,
        color=color,
        raw_bytes=field_bytes,
        highpass_bytes=field_bytes,
        peak_scratch_bytes=peak_scratch_bytes,
        scratch_files_remaining=0,
    )
