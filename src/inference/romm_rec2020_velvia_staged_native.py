"""Isolated native-assisted exact ProPhoto-to-Rec.2020 Velvia rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.color_engine.safe_lab import apply_tone_rolloff, preserve_luma_detail
from src.eval.native_rec2020_oklab_interior import (
    apply_native_rec2020_interior,
    build_native_rec2020_interior,
    load_native_rec2020_interior,
)
from src.eval.native_safe_lab_pointwise import (
    apply_native_safe_lab_pointwise,
    build_native_safe_lab_pointwise,
    load_native_safe_lab_pointwise,
)

from .romm_rec2020_velvia_staged import (
    render_supported_prophoto_velvia_rec2020_staged,
)


def render_supported_prophoto_velvia_rec2020_staged_native(
    input_path: Path,
    output_path: Path,
    *,
    profile_path: Path,
    root: Path,
    scratch_dir: Path,
    build_dir: Path,
    row_chunk: int = 128,
    thread_count: int = 8,
) -> dict[str, Any]:
    """Render through the retained C20/C22 components and Python remainder."""

    ingress_build = build_native_rec2020_interior(
        root=root, output_dir=Path(build_dir) / "ingress"
    )
    ingress_library = load_native_rec2020_interior(Path(ingress_build["dll_path"]))
    pointwise_build = build_native_safe_lab_pointwise(
        root=root, output_dir=Path(build_dir) / "pointwise"
    )
    pointwise_library = load_native_safe_lab_pointwise(
        Path(pointwise_build["dll_path"])
    )

    def ingress_mapper(decoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mapped = np.empty_like(decoded)
        scale = np.empty(decoded.shape[:2], dtype=np.float32)
        return apply_native_rec2020_interior(
            ingress_library,
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
        pointwise = apply_native_safe_lab_pointwise(
            pointwise_library,
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

    return render_supported_prophoto_velvia_rec2020_staged(
        input_path,
        output_path,
        profile_path=profile_path,
        root=root,
        scratch_dir=scratch_dir,
        row_chunk=row_chunk,
        _ingress_mapper=ingress_mapper,
        _style_mapper=style_mapper,
    )


__all__ = ["render_supported_prophoto_velvia_rec2020_staged_native"]
