"""Direct low-resolution previews for the three Look Approximation baselines."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.pipeline_color_baseline import load_guardrail_config
from src.preprocess import (
    inspect_input,
    load_jpeg_preview_working_image,
    load_working_image,
    save_srgb8,
    working_image_to_srgb_float,
)

from .render_contract import atomic_write_json, load_render_profile, sha256_file
from .three_stock_look import iter_three_stock_look_rgb_shared_context


class ThreeStockPreviewError(ValueError):
    """Raised when a direct three-stock preview cannot be produced."""


def preview_dimensions(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """Return deterministic aspect-preserving dimensions without upsampling."""

    for value, label in ((width, "width"), (height, "height"), (max_pixels, "max_pixels")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ThreeStockPreviewError(f"{label} must be a positive integer")
    if width * height <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / float(width * height))
    resized_width = max(1, math.floor(width * scale))
    resized_height = max(1, math.floor(height * scale))
    while resized_width * resized_height > max_pixels:
        if resized_width >= resized_height:
            resized_width -= 1
        else:
            resized_height -= 1
    return resized_width, resized_height


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ThreeStockPreviewError(f"{label} must be a positive integer")
    return value


def preview_fidelity_metrics(
    candidate: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    """Measure a bounded PNG8 preview against a same-size full-render reference."""

    left = np.asarray(candidate)
    right = np.asarray(reference)
    if (
        left.dtype != np.float32
        or right.dtype != np.float32
        or left.shape != right.shape
        or left.ndim != 3
        or left.shape[2] != 3
        or left.size == 0
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.any((left < 0.0) | (left > 1.0))
        or np.any((right < 0.0) | (right > 1.0))
    ):
        raise ThreeStockPreviewError(
            "candidate and reference must be same-shape bounded float32 RGB arrays"
        )
    absolute = np.abs(left.astype(np.float64) - right.astype(np.float64))
    reference_interior = (right > (1.0 / 255.0)) & (right < (254.0 / 255.0))
    candidate_boundary = (left <= 0.0) | (left >= 1.0)
    return {
        "rgb_rmse": float(np.sqrt(np.mean(np.square(absolute), dtype=np.float64))),
        "rgb_absolute_error_p95": float(np.percentile(absolute, 95.0)),
        "new_boundary_fraction": float(np.mean(candidate_boundary & reference_interior)),
    }


def render_three_stock_previews_to_directory(
    input_path: Path,
    output_directory: Path,
    *,
    root: Path,
    profile_path: Path,
    statistics_path: Path,
    guardrails_path: Path,
    max_preview_pixels: int = 1_000_000,
    look_amount: float = 1.0,
    seed: int = 7,
    tile_size: int = 256,
    tile_workers: int = 1,
    png_compression: int = 6,
    jpeg_scaled_decode: bool = False,
) -> dict[str, Any]:
    """Render all three previews after one linear-light area downsample."""

    input_path = Path(input_path)
    output_directory = Path(output_directory)
    root = Path(root)
    max_preview_pixels = _positive_integer(max_preview_pixels, "max_preview_pixels")
    tile_size = _positive_integer(tile_size, "tile_size")
    tile_workers = _positive_integer(tile_workers, "tile_workers")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ThreeStockPreviewError("seed must be an integer")
    if (
        isinstance(png_compression, bool)
        or not isinstance(png_compression, int)
        or not 0 <= png_compression <= 9
    ):
        raise ThreeStockPreviewError("png_compression must be an integer in [0, 9]")
    if not isinstance(jpeg_scaled_decode, bool):
        raise ThreeStockPreviewError("jpeg_scaled_decode must be a boolean")
    if not input_path.is_file():
        raise ThreeStockPreviewError("input_path must be an existing file")
    if output_directory.exists():
        raise ThreeStockPreviewError("output_directory must not already exist")

    profile = load_render_profile(profile_path, root=root)
    statistics_payload = json.loads(statistics_path.read_text(encoding="utf-8"))
    statistics_by_style: dict[str, Mapping[str, Any]] = {}
    guardrails_by_style: dict[str, Mapping[str, Any]] = {}
    for style in ("velvia_50", "portra_400", "ektar_100"):
        statistics_by_style[style] = statistics_payload["styles"][style]
        guardrails_by_style[style] = load_guardrail_config(guardrails_path, style)

    inspection = inspect_input(input_path)
    source_width = inspection.width
    source_height = inspection.height
    if source_width < 1 or source_height < 1:
        raise ThreeStockPreviewError("input dimensions could not be inspected")
    preview_width, preview_height = preview_dimensions(
        source_width, source_height, max_preview_pixels
    )
    use_scaled_decode = jpeg_scaled_decode and (
        (preview_width, preview_height) != (source_width, source_height)
    )
    if use_scaled_decode:
        try:
            working = load_jpeg_preview_working_image(
                input_path,
                target_width=preview_width,
                target_height=preview_height,
            )
        except ValueError as exc:
            raise ThreeStockPreviewError(str(exc)) from exc
    else:
        working = load_working_image(input_path)
    decoded_height, decoded_width = working.pixels.shape[:2]
    if (preview_width, preview_height) == (source_width, source_height):
        preview_linear = np.ascontiguousarray(working.pixels.copy())
    else:
        preview_linear = np.ascontiguousarray(
            cv2.resize(
                working.pixels,
                (preview_width, preview_height),
                interpolation=cv2.INTER_AREA,
            ),
            dtype=np.float32,
        )
    preview_working = replace(working, pixels=preview_linear)
    del working, preview_linear
    source = working_image_to_srgb_float(preview_working)
    del preview_working

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    stage = output_directory.with_name(
        f".{output_directory.name}.{os.getpid()}.{uuid.uuid4().hex}.stage"
    )
    stage.mkdir()
    rows: list[dict[str, Any]] = []
    try:
        for catalog_row, output in iter_three_stock_look_rgb_shared_context(
            source,
            profile=profile,
            look_amount=look_amount,
            style_statistics=statistics_by_style,
            guardrails=guardrails_by_style,
            seed=seed,
            tile_size=tile_size,
            tile_workers=tile_workers,
        ):
            style = catalog_row["style_id"]
            filename = f"{style}.preview.png"
            staged_output = stage / filename
            save_srgb8(output, staged_output)
            del output
            rows.append(
                {
                    "film_stock_id": catalog_row["film_stock_id"],
                    "style_id": style,
                    "output_path": str((output_directory / filename).resolve()),
                    "output_sha256": sha256_file(staged_output),
                }
            )
        manifest = {
            "schema_version": "neuro-film.three-stock-direct-preview.v1",
            "input_path": str(input_path.resolve()),
            "input_sha256": sha256_file(input_path),
            "source_width": source_width,
            "source_height": source_height,
            "preview_width": preview_width,
            "preview_height": preview_height,
            "preview_pixels": preview_width * preview_height,
            "max_preview_pixels": max_preview_pixels,
            "decoded_width": decoded_width,
            "decoded_height": decoded_height,
            "jpeg_scaled_decode": use_scaled_decode,
            "look_amount": float(look_amount),
            "preview_basis": (
                "libjpeg scaled decode then linear-light INTER_AREA resize before "
                "shared-context render"
                if use_scaled_decode
                else "linear-light INTER_AREA resize before shared-context render"
            ),
            "rows": rows,
            "claim_ceiling": (
                "Direct previews of three non-calibrated Look Approximations; "
                "not final export, calibrated stock response or stock distinguishability."
            ),
        }
        atomic_write_json(stage / "preview.json", manifest)
        os.rename(stage, output_directory)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = [
    "ThreeStockPreviewError",
    "preview_dimensions",
    "preview_fidelity_metrics",
    "render_three_stock_previews_to_directory",
]
