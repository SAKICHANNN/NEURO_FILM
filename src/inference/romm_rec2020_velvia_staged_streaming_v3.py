"""Exact-sample streaming-output v3 ProPhoto-to-Rec.2020 Velvia rendering."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_rec2020_postcolor_rgb16_v1 import (
    apply_native_rec2020_postcolor_v1,
    build_exact_rec2020_rgb16_buckets,
    build_exact_rec2020_rgb16_thresholds,
    build_native_rec2020_postcolor_v1,
    load_native_rec2020_postcolor_v1,
)
from src.preprocess.png_stream import StreamingRec2020PngWriter

from .romm_rec2020_velvia_staged_native_v2 import (
    NativeStagedVelviaV2Runtime,
    build_native_staged_velvia_v2_runtime,
    render_supported_prophoto_velvia_rec2020_staged_native_v2,
)


@dataclass(frozen=True)
class NativeStagedVelviaV3Runtime:
    color: NativeStagedVelviaV2Runtime
    postcolor_library: Any
    quantization_thresholds: np.ndarray
    quantization_buckets: np.ndarray
    build_receipt: dict[str, Any]


def build_native_staged_velvia_v3_runtime(
    *, root: Path, build_dir: Path
) -> NativeStagedVelviaV3Runtime:
    color = build_native_staged_velvia_v2_runtime(
        root=root, build_dir=Path(build_dir) / "color"
    )
    postcolor = build_native_rec2020_postcolor_v1(
        root=root, output_dir=Path(build_dir) / "postcolor"
    )
    thresholds = build_exact_rec2020_rgb16_thresholds()
    buckets = build_exact_rec2020_rgb16_buckets(thresholds)
    return NativeStagedVelviaV3Runtime(
        color=color,
        postcolor_library=load_native_rec2020_postcolor_v1(
            Path(postcolor["dll_path"])
        ),
        quantization_thresholds=thresholds,
        quantization_buckets=buckets,
        build_receipt={
            **color.build_receipt,
            "postcolor_dll_sha256": postcolor["dll_sha256"],
            "quantization_thresholds_sha256": hashlib.sha256(
                thresholds.tobytes()
            ).hexdigest(),
            "quantization_buckets_sha256": hashlib.sha256(
                buckets.tobytes()
            ).hexdigest(),
        },
    )


def render_supported_prophoto_velvia_rec2020_staged_streaming_v3(
    input_path: Path,
    output_path: Path,
    *,
    profile_path: Path,
    root: Path,
    scratch_dir: Path,
    build_dir: Path,
    row_chunk: int = 128,
    thread_count: int = 8,
    runtime: NativeStagedVelviaV3Runtime | None = None,
) -> dict[str, Any]:
    """Render exact v2 samples into a deterministic uncompressed CICP PNG."""

    runtime = runtime or build_native_staged_velvia_v3_runtime(
        root=root, build_dir=build_dir
    )

    def writer_factory(path: Path, width: int, height: int) -> StreamingRec2020PngWriter:
        return StreamingRec2020PngWriter(
            path,
            width=width,
            height=height,
            bit_depth=16,
            compression_level=0,
        )

    def postcolor_mapper(
        source: np.ndarray, candidate: np.ndarray, margin: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return apply_native_rec2020_postcolor_v1(
            runtime.postcolor_library,
            np.ascontiguousarray(source),
            np.ascontiguousarray(candidate),
            margin,
            thread_count=thread_count,
            thresholds=runtime.quantization_thresholds,
            buckets=runtime.quantization_buckets,
        )

    return render_supported_prophoto_velvia_rec2020_staged_native_v2(
        input_path,
        output_path,
        profile_path=profile_path,
        root=root,
        scratch_dir=scratch_dir,
        build_dir=build_dir,
        row_chunk=row_chunk,
        thread_count=thread_count,
        runtime=runtime.color,
        _output_writer_factory=writer_factory,
        _postcolor_mapper=postcolor_mapper,
        _in_memory_staging=True,
        _preprocess_workers=2,
        _spill_mapped_for_context=True,
    )


__all__ = [
    "NativeStagedVelviaV3Runtime",
    "build_native_staged_velvia_v3_runtime",
    "render_supported_prophoto_velvia_rec2020_staged_streaming_v3",
]
