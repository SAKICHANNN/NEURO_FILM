"""Exact-semantics v2 native-assisted ProPhoto-to-Rec.2020 Velvia rendering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.color_engine.safe_lab import apply_tone_rolloff, preserve_luma_detail
from src.eval.native_rec2020_lab_source_compress_v2 import (
    apply_native_rec2020_lab_compress_v2,
    build_native_rec2020_lab_compress_v2,
    load_native_rec2020_lab_compress_v2,
)
from src.eval.native_rec2020_oklab_interior import (
    apply_native_rec2020_interior,
    build_native_rec2020_interior,
    load_native_rec2020_interior,
)
from src.eval.native_safe_lab_pointwise_v2 import (
    apply_native_safe_lab_pointwise_v2,
    build_native_safe_lab_pointwise_v2,
    load_native_safe_lab_pointwise_v2,
)

from .romm_rec2020_velvia_staged import (
    render_supported_prophoto_velvia_rec2020_staged,
)


@dataclass(frozen=True)
class NativeStagedVelviaV2Runtime:
    ingress_library: Any
    pointwise_library: Any
    compress_library: Any
    build_receipt: dict[str, Any]


def build_native_staged_velvia_v2_runtime(
    *, root: Path, build_dir: Path
) -> NativeStagedVelviaV2Runtime:
    ingress_build = build_native_rec2020_interior(
        root=root, output_dir=Path(build_dir) / "ingress"
    )
    pointwise_build = build_native_safe_lab_pointwise_v2(
        root=root, output_dir=Path(build_dir) / "pointwise"
    )
    compress_build = build_native_rec2020_lab_compress_v2(
        root=root, output_dir=Path(build_dir) / "compress"
    )
    return NativeStagedVelviaV2Runtime(
        ingress_library=load_native_rec2020_interior(
            Path(ingress_build["dll_path"])
        ),
        pointwise_library=load_native_safe_lab_pointwise_v2(
            Path(pointwise_build["dll_path"])
        ),
        compress_library=load_native_rec2020_lab_compress_v2(
            Path(compress_build["dll_path"])
        ),
        build_receipt={
            "ingress_dll_sha256": ingress_build["dll_sha256"],
            "pointwise_dll_sha256": pointwise_build["dll_sha256"],
            "compress_dll_sha256": compress_build["dll_sha256"],
            "toolchain": pointwise_build["toolchain"],
        },
    )


def render_supported_prophoto_velvia_rec2020_staged_native_v2(
    input_path: Path,
    output_path: Path,
    *,
    profile_path: Path,
    root: Path,
    scratch_dir: Path,
    build_dir: Path,
    row_chunk: int = 128,
    thread_count: int = 8,
    runtime: NativeStagedVelviaV2Runtime | None = None,
    _output_writer_factory: Callable[[Path, int, int], Any] | None = None,
    _postcolor_mapper: Callable[
        [np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]
    ]
    | None = None,
    _in_memory_staging: bool = False,
) -> dict[str, Any]:
    """Render through the retained C20/C22 components and Python remainder."""

    runtime = runtime or build_native_staged_velvia_v2_runtime(
        root=root, build_dir=build_dir
    )

    def ingress_mapper(decoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mapped = np.empty_like(decoded)
        scale = np.empty(decoded.shape[:2], dtype=np.float32)
        return apply_native_rec2020_interior(
            runtime.ingress_library,
            decoded,
            output=mapped,
            chroma_scale=scale,
            thread_count=thread_count,
        )

    def style_mapper(
        source_lab: np.ndarray,
        *,
        source_context,
        destination_mean: np.ndarray,
        destination_std: np.ndarray,
        style: str,
        strength: float,
        luma_strength: float,
        tone_rolloff: float,
        shadow_floor_l: float,
        highlight_ceiling_l: float,
        preserve_luma_detail_strength: float,
        chroma_curve_strength: float,
        neutral_protect: float,
        skin_protect: float,
        max_chroma_gain: float | None,
        max_chroma_boost: float | None,
        max_chroma_absolute: float | None,
    ) -> np.ndarray:
        if style != "velvia_50" or max_chroma_gain is None or max_chroma_boost is None:
            raise ValueError("native staged renderer requires the frozen Velvia profile")
        pointwise = apply_native_safe_lab_pointwise_v2(
            runtime.pointwise_library,
            np.ascontiguousarray(source_lab),
            source_context=source_context,
            destination_mean=destination_mean,
            destination_std=destination_std,
            strength=strength,
            luma_strength=luma_strength,
            chroma_curve_strength=chroma_curve_strength,
            neutral_protect=neutral_protect,
            skin_protect=skin_protect,
            max_chroma_gain=max_chroma_gain,
            max_chroma_boost=max_chroma_boost,
            max_chroma_absolute=max_chroma_absolute,
            thread_count=thread_count,
        )
        detailed = preserve_luma_detail(
            source_lab, pointwise, preserve_luma_detail_strength
        )
        return apply_tone_rolloff(
            detailed,
            tone_rolloff,
            shadow_floor_l,
            highlight_ceiling_l,
        )

    def gamut_mapper(
        source_lab: np.ndarray, styled_lab: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        output_lab, output_rgb, _ = apply_native_rec2020_lab_compress_v2(
            runtime.compress_library,
            np.ascontiguousarray(source_lab),
            np.ascontiguousarray(styled_lab),
            thread_count=thread_count,
        )
        return output_lab, output_rgb

    return render_supported_prophoto_velvia_rec2020_staged(
        input_path,
        output_path,
        profile_path=profile_path,
        root=root,
        scratch_dir=scratch_dir,
        row_chunk=row_chunk,
        _ingress_mapper=ingress_mapper,
        _style_mapper=style_mapper,
        _gamut_mapper=gamut_mapper,
        _output_writer_factory=_output_writer_factory,
        _postcolor_mapper=_postcolor_mapper,
        _in_memory_staging=_in_memory_staging,
    )


__all__ = [
    "NativeStagedVelviaV2Runtime",
    "build_native_staged_velvia_v2_runtime",
    "render_supported_prophoto_velvia_rec2020_staged_native_v2",
]
