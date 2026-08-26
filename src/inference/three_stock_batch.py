"""One-decode file export for the three explicit stock Look Approximations."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from scripts.pipeline_color_baseline import load_guardrail_config
from src.preprocess import (
    StreamingSrgbPngWriter,
    load_working_image,
    resolve_look_approximation_claim,
    srgb_icc_profile_fingerprint_sha256,
    working_image_to_srgb_float,
)

from .render_contract import (
    atomic_write_json,
    build_render_recipe,
    load_render_profile,
    sha256_file,
    validate_render_recipe,
)
from .three_stock_look import (
    iter_three_stock_look_rgb_shared_context,
    resolve_three_stock_look_parameters,
)


class ThreeStockBatchError(ValueError):
    """Raised when a three-stock file batch cannot be completed."""


THREE_STOCK_PNG_ENCODER_ID = "kmcfm.streaming-srgb-rgb16-png.v1"
THREE_STOCK_PNG_ROW_COUNT = 128


def _save_srgb16_png_streaming(
    rgb: np.ndarray,
    path: Path,
    *,
    compression_level: int,
    row_count: int = THREE_STOCK_PNG_ROW_COUNT,
) -> str:
    """Quantize and publish one RGB16 PNG without a full-frame encode buffer."""

    values = np.asarray(rgb)
    if (
        values.dtype != np.float32
        or values.ndim != 3
        or values.shape[2] != 3
        or values.shape[0] <= 0
        or values.shape[1] <= 0
        or not np.isfinite(values).all()
    ):
        raise ThreeStockBatchError("streaming PNG input must be finite HxWx3 float32")
    row_count = _integer(row_count, "row_count", 1)
    height, width, _ = values.shape
    with StreamingSrgbPngWriter(
        path,
        width=int(width),
        height=int(height),
        bit_depth=16,
        compression_level=compression_level,
    ) as writer:
        for row_start in range(0, height, row_count):
            rows = np.rint(
                np.clip(values[row_start : row_start + row_count], 0.0, 1.0)
                * np.float32(65535.0)
            ).astype(np.uint16)
            writer.write_rows(row_start, np.ascontiguousarray(rows))
        writer.finish()
    return "PNG"


def _integer(value: object, label: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ThreeStockBatchError(f"{label} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ThreeStockBatchError(f"{label} must be <= {maximum}")
    return value


def render_three_stock_batch_to_directory(
    input_path: Path,
    output_directory: Path,
    *,
    root: Path,
    profile_path: Path,
    statistics_path: Path,
    guardrails_path: Path,
    look_amount: float = 1.0,
    seed: int = 31,
    tile_size: int = 512,
    tile_workers: int = 1,
    png_compression: int = 0,
) -> dict[str, Any]:
    """Render all three PNG16 outputs and recipes, then publish one directory."""

    input_path = Path(input_path)
    output_directory = Path(output_directory)
    root = Path(root)
    tile_size = _integer(tile_size, "tile_size", 1)
    tile_workers = _integer(tile_workers, "tile_workers", 1)
    png_compression = _integer(png_compression, "png_compression", 0, 9)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ThreeStockBatchError("seed must be an integer")
    if not input_path.is_file():
        raise ThreeStockBatchError("input_path must be an existing file")
    if output_directory.exists():
        raise ThreeStockBatchError("output_directory must not already exist")

    profile = load_render_profile(profile_path, root=root)
    statistics_payload = json.loads(statistics_path.read_text(encoding="utf-8"))
    statistics_by_style: dict[str, Mapping[str, Any]] = {}
    guardrails_by_style: dict[str, Mapping[str, Any]] = {}
    for style in ("velvia_50", "portra_400", "ektar_100"):
        statistics_by_style[style] = statistics_payload["styles"][style]
        guardrails_by_style[style] = load_guardrail_config(guardrails_path, style)

    working = load_working_image(input_path)
    source = working_image_to_srgb_float(working)
    output_claim = resolve_look_approximation_claim(working)
    input_metadata = {
        "color_state": working.source_transfer_state,
        "working_space": working.working_space,
        "source_profile_kind": working.source_profile.kind,
        "bit_depth": working.bit_depth_in,
        "warnings": [warning.__dict__ for warning in working.warnings],
        "source_profile_fingerprint_sha256": None,
    }
    # ``source`` owns its encoded float32 pixels.  None of the remaining
    # render or recipe stages needs the decoded linear WorkingImage, whose
    # full-resolution pixel array would otherwise overlap the shared Lab
    # context and every stock output.
    del working
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, encoding="utf-8"
    ).strip()

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
            _, color_parameters = resolve_three_stock_look_parameters(
                profile,
                film_stock_id=catalog_row["film_stock_id"],
                look_amount=look_amount,
            )
            filename = f"{style}.png"
            staged_output = stage / filename
            output_format = _save_srgb16_png_streaming(
                output,
                staged_output,
                compression_level=png_compression,
            )
            del output
            final_output = output_directory / filename
            recipe = build_render_recipe(
                profile_path=profile_path,
                profile=profile,
                input_path=input_path,
                input_metadata=input_metadata,
                render_metadata={
                    "engine_id": "safe_lab_v1",
                    "preset": "safe-rich",
                    "style": style,
                    "seed": seed,
                    "color_parameters": color_parameters,
                    "effects": {
                        "grain": {"strength": 0.0, "seed": seed, "color": True},
                        "halation": {
                            "strength": 0.0,
                            "model": "simple",
                            "preset": None,
                            "control_mode": "locked",
                            "resolved_parameters": None,
                        },
                        "dust": {"strength": 0.0, "seed": seed + 17},
                    },
                    "look_amount": float(look_amount),
                },
                output_path=staged_output,
                output_format=output_format,
                output_bit_depth=16,
                output_icc_fingerprint_sha256=srgb_icc_profile_fingerprint_sha256(),
                output_claim=output_claim,
                software_commit=commit,
                output_png_compression=png_compression,
            )
            recipe["output"]["path"] = str(final_output.resolve())
            validate_render_recipe(recipe)
            recipe_name = f"{style}.recipe.json"
            recipe_sha256 = atomic_write_json(stage / recipe_name, recipe)
            rows.append(
                {
                    "film_stock_id": catalog_row["film_stock_id"],
                    "style_id": style,
                    "output_path": str(final_output.resolve()),
                    "output_sha256": sha256_file(staged_output),
                    "recipe_path": str((output_directory / recipe_name).resolve()),
                    "recipe_sha256": recipe_sha256,
                }
            )
        manifest = {
            "schema_version": "neuro-film.three-stock-file-batch.v1",
            "input_path": str(input_path.resolve()),
            "input_sha256": sha256_file(input_path),
            "profile_sha256": sha256_file(profile_path),
            "look_amount": float(look_amount),
            "png_compression": png_compression,
            "png_encoder_id": THREE_STOCK_PNG_ENCODER_ID,
            "png_row_count": THREE_STOCK_PNG_ROW_COUNT,
            "rows": rows,
            "claim_ceiling": (
                "Three deterministic non-calibrated Look Approximations; "
                "stock separation and calibrated response are not established."
            ),
        }
        atomic_write_json(stage / "batch.json", manifest)
        os.rename(stage, output_directory)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = [
    "THREE_STOCK_PNG_ENCODER_ID",
    "THREE_STOCK_PNG_ROW_COUNT",
    "ThreeStockBatchError",
    "render_three_stock_batch_to_directory",
]
