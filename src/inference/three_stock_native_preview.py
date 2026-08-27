"""Opt-in exact native colour backend for the three-stock preview path."""

from __future__ import annotations

import ctypes
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from skimage.color import rgb2lab

from scripts.pipeline_color_baseline import (
    apply_output_margin,
    compress_to_srgb_gamut_parallel,
    lab_to_rgb_no_clip,
)
from src.color_engine.safe_lab import (
    apply_tone_rolloff,
    preserve_luma_detail,
    safe_lab_context_from_lab,
)
from src.eval.native_safe_lab_pointwise_v3 import (
    apply_native_safe_lab_pointwise_v3,
    build_native_safe_lab_pointwise_v3,
    load_native_safe_lab_pointwise_v3,
)

from .style_safe_engine import StyleSafeEngineError
from .three_stock_look import (
    THREE_STOCK_LOOK_CATALOG,
    resolve_three_stock_look_parameters,
)


@dataclass
class NativeThreeStockPreviewBackend:
    """One explicitly built native runtime and its reproducible identity."""

    library: ctypes.CDLL
    dll_sha256: str
    source_sha256: str
    header_sha256: str
    toolchain: str
    thread_count: int = 8
    gamut_workers: int = 1
    closed: bool = False

    def close(self) -> None:
        """Release the Windows DLL so its owned build directory can be removed."""

        if self.closed:
            return
        free_library = ctypes.windll.kernel32.FreeLibrary
        free_library.argtypes = [ctypes.c_void_p]
        free_library.restype = ctypes.c_int
        if not free_library(self.library._handle):
            raise RuntimeError("failed to unload native three-stock preview backend")
        self.closed = True

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("native three-stock preview backend is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def build_native_three_stock_preview_backend(
    *, root: Path, output_directory: Path, thread_count: int = 8, gamut_workers: int = 1
) -> NativeThreeStockPreviewBackend:
    """Build and load the exact U7.6J pointwise runtime."""

    for value, label in ((thread_count, "thread_count"), (gamut_workers, "gamut_workers")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    build = build_native_safe_lab_pointwise_v3(
        root=Path(root), output_dir=Path(output_directory)
    )
    return NativeThreeStockPreviewBackend(
        library=load_native_safe_lab_pointwise_v3(Path(build["dll_path"])),
        dll_sha256=str(build["dll_sha256"]),
        source_sha256=str(build["source_sha256"]),
        header_sha256=str(build["header_sha256"]),
        toolchain=str(build["toolchain"]),
        thread_count=thread_count,
        gamut_workers=gamut_workers,
    )


def _native_output_rgb(
    source_lab: np.ndarray,
    styled_lab: np.ndarray,
    *,
    dither: float,
    output_margin: int,
    seed: int,
    gamut_workers: int,
) -> np.ndarray:
    compressed = compress_to_srgb_gamut_parallel(
        source_lab, styled_lab, workers=gamut_workers
    )
    result = lab_to_rgb_no_clip(compressed)
    if dither > 0.0:
        rng = np.random.default_rng(seed + 1009)
        noise = rng.uniform(-0.5, 0.5, size=result.shape).astype(np.float32)
        result = np.clip(result + noise * (dither / 255.0), 0.0, 1.0)
    result = apply_output_margin(result, output_margin)
    return np.asarray(np.clip(result, 0.0, 1.0), dtype=np.float32)


def iter_three_stock_look_rgb_native(
    encoded_srgb: np.ndarray,
    *,
    backend: NativeThreeStockPreviewBackend,
    profile: Mapping[str, Any],
    look_amount: float,
    style_statistics: Mapping[str, Mapping[str, Any]],
    guardrails: Mapping[str, Mapping[str, Any]],
    seed: int,
) -> Iterator[tuple[dict[str, str], np.ndarray]]:
    """Yield exact current three-stock outputs through an explicit native runtime."""

    if not isinstance(backend, NativeThreeStockPreviewBackend) or backend.closed:
        raise StyleSafeEngineError("native three-stock preview backend is unavailable")
    source = np.asarray(encoded_srgb)
    if (
        source.dtype != np.float32
        or source.ndim != 3
        or source.shape[2] != 3
        or source.size == 0
        or not source.flags.c_contiguous
        or not np.isfinite(source).all()
        or np.any((source < 0.0) | (source > 1.0))
    ):
        raise StyleSafeEngineError(
            "encoded_srgb must be finite bounded C-contiguous HxWx3 float32"
        )
    source_lab = np.ascontiguousarray(rgb2lab(source), dtype=np.float32)
    context = safe_lab_context_from_lab(source_lab)
    for catalog_row in THREE_STOCK_LOOK_CATALOG:
        style, parameters = resolve_three_stock_look_parameters(
            profile,
            film_stock_id=catalog_row["film_stock_id"],
            look_amount=look_amount,
        )
        if style not in style_statistics or style not in guardrails:
            raise StyleSafeEngineError(f"missing three-stock inputs for style: {style}")
        if look_amount == 0.0:
            yield dict(catalog_row), np.ascontiguousarray(source.copy())
            continue
        if (
            float(parameters["grain"]) != 0.0
            or parameters["gamut_mode"] != "source"
            or not bool(parameters["use_guardrails"])
        ):
            raise StyleSafeEngineError("native preview backend rejects unsupported profile semantics")
        statistics = style_statistics[style]
        guard = guardrails[style]
        pointwise = apply_native_safe_lab_pointwise_v3(
            backend.library,
            source_lab,
            source_context=context,
            destination_mean=np.asarray(statistics["mean"], dtype=np.float32),
            destination_std=np.asarray(statistics["std"], dtype=np.float32),
            strength=float(parameters["strength"]),
            luma_strength=float(parameters["luma_strength"]),
            chroma_curve_strength=float(parameters["chroma_curve_strength"]),
            neutral_protect=float(guard["neutral_protect"]),
            skin_protect=float(guard["skin_protect"]),
            max_chroma_gain=float(guard["max_chroma_gain"]),
            max_chroma_boost=float(guard["max_chroma_boost"]),
            max_chroma_absolute=guard.get("max_chroma_absolute"),
            thread_count=backend.thread_count,
        )
        detailed = preserve_luma_detail(
            source_lab, pointwise, float(parameters["preserve_luma_detail"])
        )
        styled = apply_tone_rolloff(
            detailed,
            float(parameters["tone_rolloff"]),
            float(parameters["shadow_floor_l"]),
            float(parameters["highlight_ceiling_l"]),
        )
        yield dict(catalog_row), _native_output_rgb(
            source_lab,
            styled,
            dither=float(parameters["dither"]),
            output_margin=int(parameters["output_margin"]),
            seed=seed,
            gamut_workers=backend.gamut_workers,
        )


__all__ = [
    "NativeThreeStockPreviewBackend",
    "build_native_three_stock_preview_backend",
    "iter_three_stock_look_rgb_native",
]
