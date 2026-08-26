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
from src.color_engine.safe_lab import SafeLabSourceContext
from src.color_engine.safe_lab_rgb_context import build_safe_lab_source_context
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
from .style_safe_engine import stream_resolved_safe_lab_rgb_rows
from .three_stock_look import (
    list_three_stock_looks,
    resolve_three_stock_look_parameters,
)


class ThreeStockBatchError(ValueError):
    """Raised when a three-stock file batch cannot be completed."""


THREE_STOCK_RENDER_EXECUTION_ID = (
    "row-striped-safe-lab-plus-streaming-srgb16-png.v1"
)


def _stream_three_stock_png(
    source: np.ndarray,
    path: Path,
    *,
    style: str,
    style_statistics: Mapping[str, Any],
    style_parameters: Mapping[str, Any],
    guardrails: Mapping[str, Any],
    seed: int,
    tile_size: int,
    tile_workers: int,
    source_context: SafeLabSourceContext | None,
    compression_level: int,
) -> str:
    """Render one look into PNG rows without a full-resolution output array."""

    with StreamingSrgbPngWriter(
        path,
        width=int(source.shape[1]),
        height=int(source.shape[0]),
        bit_depth=16,
        compression_level=compression_level,
    ) as writer:

        def write_rows(row_start: int, rows: np.ndarray) -> None:
            encoded = np.rint(
                rows * np.float32(65535.0)
            ).astype(np.uint16)
            writer.write_rows(row_start, np.ascontiguousarray(encoded))

        if source_context is None:
            for row_start in range(0, source.shape[0], tile_size):
                write_rows(row_start, source[row_start : row_start + tile_size])
        else:
            stream_resolved_safe_lab_rgb_rows(
                source,
                style=style,
                style_statistics=style_statistics,
                style_parameters=style_parameters,
                guardrails=guardrails,
                seed=seed,
                tile_size=tile_size,
                source_context=source_context,
                consumer=write_rows,
                tile_workers=tile_workers,
            )
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
        source_context = (
            None if look_amount == 0.0 else build_safe_lab_source_context(source)
        )
        for catalog_row in list_three_stock_looks():
            style = catalog_row["style_id"]
            _, color_parameters = resolve_three_stock_look_parameters(
                profile,
                film_stock_id=catalog_row["film_stock_id"],
                look_amount=look_amount,
            )
            filename = f"{style}.png"
            staged_output = stage / filename
            output_format = _stream_three_stock_png(
                source,
                staged_output,
                style=style,
                style_statistics=statistics_by_style[style],
                style_parameters=color_parameters,
                guardrails=guardrails_by_style[style],
                seed=seed,
                tile_size=tile_size,
                tile_workers=tile_workers,
                source_context=source_context,
                compression_level=png_compression,
            )
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
            "render_execution_id": THREE_STOCK_RENDER_EXECUTION_ID,
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
    "THREE_STOCK_RENDER_EXECUTION_ID",
    "ThreeStockBatchError",
    "render_three_stock_batch_to_directory",
]
