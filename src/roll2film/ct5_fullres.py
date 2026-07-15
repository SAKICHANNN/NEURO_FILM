"""Full-resolution CT5 diagnostics and deterministic review sheets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from skimage.color import deltaE_ciede2000, rgb2lab

from src.filmcase.diagnostics import chroma_speckle_diagnostics


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    rgb = np.asarray(values, dtype=np.float64)
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(np.maximum(rgb, 0.0), 1.0 / 2.4) - 0.055,
    )


def linear_to_u8(values: np.ndarray) -> np.ndarray:
    encoded = np.clip(linear_to_srgb(values), 0.0, 1.0)
    return np.rint(encoded * 255.0).astype(np.uint8)


def full_resolution_diagnostics(
    source_linear: np.ndarray,
    target_linear: np.ndarray,
    output_linear: np.ndarray,
) -> dict[str, Any]:
    source = np.asarray(source_linear, dtype=np.float64)
    target = np.asarray(target_linear, dtype=np.float64)
    output = np.asarray(output_linear, dtype=np.float64)
    if source.shape != target.shape or source.shape != output.shape or source.ndim != 3:
        raise ValueError("full-resolution source, target, and output must share HxWx3")
    if not np.all(np.isfinite(output)):
        raise ValueError("full-resolution output contains non-finite values")
    source_u8 = linear_to_u8(source)
    output_u8 = linear_to_u8(output)
    target_u8 = linear_to_u8(target)
    target_encoded = linear_to_srgb(target)
    output_encoded = linear_to_srgb(output)
    source_lab = rgb2lab(linear_to_srgb(source))
    target_lab = rgb2lab(target_encoded)
    output_lab = rgb2lab(output_encoded)
    target_delta = deltaE_ciede2000(target_lab, output_lab)
    style_delta = deltaE_ciede2000(source_lab, output_lab)
    luminance_u8 = np.rint(np.clip(output_lab[..., 0] / 100.0 * 255.0, 0, 255)).astype(np.uint8)
    hist = np.bincount(luminance_u8.ravel(), minlength=256)
    low, high = np.percentile(luminance_u8, [1, 99]).astype(int)
    active = hist[low : high + 1]
    red_cyan = (
        (np.abs(output[..., 0] - output[..., 1]) > 0.25)
        | (np.abs(output[..., 0] - output[..., 2]) > 0.25)
    ) & (np.max(output, axis=-1) > 0.75)
    lower_excursion = np.maximum(-output, 0.0)
    upper_excursion = np.maximum(output - 1.0, 0.0)
    excursion = np.maximum(lower_excursion, upper_excursion)
    nonzero_excursion = excursion[excursion > 0.0]
    source_clip_pixels = np.any((source_u8 == 0) | (source_u8 == 255), axis=-1)
    target_clip_pixels = np.any((target_u8 == 0) | (target_u8 == 255), axis=-1)
    output_clip_pixels = np.any((output_u8 == 0) | (output_u8 == 255), axis=-1)
    return {
        "mean_delta_e00_to_target": float(np.mean(target_delta)),
        "p95_delta_e00_to_target": float(np.percentile(target_delta, 95)),
        "median_delta_e00_from_input": float(np.median(style_delta)),
        "linear_rgb_rmse_to_target": float(np.sqrt(np.mean((output - target) ** 2))),
        "raw_out_of_range_fraction": float(np.mean((output < 0.0) | (output > 1.0))),
        "raw_out_of_range_pixel_fraction": float(
            np.mean(np.any((output < 0.0) | (output > 1.0), axis=-1))
        ),
        "raw_undershoot_channel_fraction": float(np.mean(output < 0.0)),
        "raw_overshoot_channel_fraction": float(np.mean(output > 1.0)),
        "raw_excursion_mean_among_oob_channels": float(
            np.mean(nonzero_excursion) if nonzero_excursion.size else 0.0
        ),
        "raw_excursion_p95_among_oob_channels": float(
            np.percentile(nonzero_excursion, 95) if nonzero_excursion.size else 0.0
        ),
        "raw_excursion_max": float(np.max(excursion)),
        "display_clip_channel_fraction": float(
            np.mean((output_u8 == 0) | (output_u8 == 255))
        ),
        "display_clip_pixel_fraction": float(np.mean(output_clip_pixels)),
        "source_display_clip_pixel_fraction": float(np.mean(source_clip_pixels)),
        "target_display_clip_pixel_fraction": float(np.mean(target_clip_pixels)),
        "new_display_clip_pixel_fraction_vs_source": float(
            np.mean(output_clip_pixels & ~source_clip_pixels)
        ),
        "new_display_clip_pixel_fraction_vs_target": float(
            np.mean(output_clip_pixels & ~target_clip_pixels)
        ),
        "luma_empty_bin_fraction": float(1.0 - np.count_nonzero(active) / max(len(active), 1)),
        "luma_max_bin_fraction": float(active.max() / max(active.sum(), 1)),
        "red_cyan_boundary_occupancy": float(np.mean(red_cyan)),
        "chroma_speckle": chroma_speckle_diagnostics(source_u8, output_u8),
    }


def make_review_sheet(
    rows: list[dict[str, Path | str]],
    output_path: Path,
    *,
    tile_size: int = 256,
) -> None:
    columns = ("input", "target", "best_basic", "candidate")
    header = 28
    sheet = Image.new("RGB", (tile_size * len(columns), header + tile_size * len(rows)), "#151515")
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(columns):
        draw.text((column * tile_size + 8, 7), label, fill="white")
    for row_index, row in enumerate(rows):
        for column, label in enumerate(columns):
            with Image.open(Path(row[label])) as image:
                tile = image.convert("RGB")
                tile.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (tile_size, tile_size), "black")
                canvas.paste(tile, ((tile_size - tile.width) // 2, (tile_size - tile.height) // 2))
            sheet.paste(canvas, (column * tile_size, header + row_index * tile_size))
        draw.text((5, header + row_index * tile_size + 5), str(row["content_id"]), fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, "JPEG", quality=92, subsampling=0)
