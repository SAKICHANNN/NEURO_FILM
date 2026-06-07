#!/usr/bin/env python3
"""Mine candidate halation patches and display-level metrics from real photos."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import find_objects, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine real-photo halation patches from a source manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/eval/halation_real_photo_v1"))
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--max-patches", type=int, default=36)
    parser.add_argument("--max-patches-per-source", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=0.30)
    parser.add_argument("--threshold-percentile", type=float, default=99.55)
    return parser.parse_args()


def load_rgb(path: Path, max_side: int = 1800) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path, "PNG")


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def annulus_masks(size: int, center: tuple[float, float], source_radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dist = np.hypot(xx - center[0], yy - center[1])
    source = dist <= source_radius
    inner = (dist > source_radius * 1.25) & (dist <= source_radius * 3.0)
    outer = (dist > source_radius * 3.0) & (dist <= source_radius * 6.5)
    return source, inner, outer


def metric_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def patch_metrics(patch: np.ndarray, source_center: tuple[float, float], source_radius: float) -> dict[str, float]:
    lum = luminance(patch)
    source_mask, inner_mask, outer_mask = annulus_masks(patch.shape[0], source_center, source_radius)
    background_mask = outer_mask if outer_mask.any() else ~source_mask
    background_luma = metric_mean(lum[background_mask])
    source_peak = float(np.percentile(lum[source_mask], 98.0)) if source_mask.any() else float(lum.max())
    red_excess_map = np.maximum(patch[..., 0] - np.maximum(patch[..., 1], patch[..., 2]), 0.0)
    halo_signal = np.maximum(lum - background_luma, 0.0) * 0.45 + red_excess_map * 0.55
    yy_grid, xx_grid = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]].astype(np.float32)
    radial = np.hypot(xx_grid - source_center[0], yy_grid - source_center[1])
    visible_mask = radial > source_radius * 1.2
    if visible_mask.any():
        threshold = max(0.012, float(np.percentile(halo_signal[visible_mask], 90.0)) * 0.35)
        visible = visible_mask & (halo_signal > threshold)
        if visible.any():
            yy, xx = np.nonzero(visible)
            visible_radius = float(np.percentile(np.hypot(xx - source_center[0], yy - source_center[1]), 85.0))
        else:
            visible_radius = 0.0
    else:
        visible_radius = 0.0
    eps = 1e-5
    inner_rgb = patch[inner_mask] if inner_mask.any() else patch.reshape(-1, 3)
    outer_rgb = patch[outer_mask] if outer_mask.any() else patch.reshape(-1, 3)
    source_rgb = patch[source_mask] if source_mask.any() else patch.reshape(-1, 3)
    source_rg = float((source_rgb[:, 0].mean() + eps) / (source_rgb[:, 1].mean() + eps))
    source_blue_leakage = float(source_rgb[:, 2].mean() / max(source_rgb[:, :2].mean(), eps))
    inner_rg = float((inner_rgb[:, 0].mean() + eps) / (inner_rgb[:, 1].mean() + eps))
    outer_rg = float((outer_rgb[:, 0].mean() + eps) / (outer_rgb[:, 1].mean() + eps))
    blue_leakage = float(outer_rgb[:, 2].mean() / max(outer_rgb[:, :2].mean(), eps))
    orange_core = float(
        ((inner_rgb[:, 0].mean() + inner_rgb[:, 1].mean()) * 0.5 - inner_rgb[:, 2].mean())
        * (inner_rg / max(outer_rg, eps))
    )
    contrast = source_peak - background_luma
    red_excess = max(0.0, outer_rg - 1.0)
    dark_bg_score = max(0.0, min(1.0, (0.42 - background_luma) / 0.42))
    blue_penalty = max(0.0, blue_leakage - 0.85)
    huge_radius_penalty = max(0.0, visible_radius / patch.shape[0] - 0.36)
    halo_confidence = float(
        np.clip(
            0.34 * contrast
            + 0.30 * dark_bg_score
            + 0.26 * red_excess
            + 0.10 * min(1.0, visible_radius / 45.0)
            - 0.18 * blue_penalty
            - 0.25 * huge_radius_penalty,
            0.0,
            1.0,
        )
    )
    return {
        "source_luma_peak": source_peak,
        "background_luma": background_luma,
        "visible_radius_px": visible_radius,
        "radius_normalized": float(visible_radius / max(source_radius, 1.0)),
        "red_green_ratio_inner": inner_rg,
        "red_green_ratio_outer": outer_rg,
        "source_red_green_ratio": source_rg,
        "source_blue_leakage": source_blue_leakage,
        "blue_leakage": blue_leakage,
        "orange_core_score": orange_core,
        "dark_side_ratio": dark_bg_score,
        "halo_confidence": halo_confidence,
    }


def crop_patch(rgb: np.ndarray, cx: float, cy: float, size: int) -> tuple[np.ndarray, tuple[float, float]] | None:
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + size
    y1 = y0 + size
    if x0 < 0 or y0 < 0 or x1 > rgb.shape[1] or y1 > rgb.shape[0]:
        return None
    return rgb[y0:y1, x0:x1].copy(), (cx - x0, cy - y0)


def candidates_from_image(rgb: np.ndarray, threshold_percentile: float) -> list[dict]:
    lum = luminance(rgb)
    threshold = float(np.percentile(lum, threshold_percentile))
    bright = lum >= max(threshold, 0.72)
    labels, count = label(bright)
    objects = find_objects(labels)
    candidates: list[dict] = []
    for component_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        ys, xs = slc
        area = int((labels[slc] == component_id).sum())
        if area < 3 or area > max(800, lum.size * 0.004):
            continue
        height = ys.stop - ys.start
        width = xs.stop - xs.start
        if height > 140 or width > 140:
            continue
        mask = labels[slc] == component_id
        yy, xx = np.nonzero(mask)
        weights = lum[slc][mask] + 1e-4
        cx = float((xx + xs.start) @ weights / weights.sum())
        cy = float((yy + ys.start) @ weights / weights.sum())
        source_radius = float(max(2.0, math.sqrt(area / math.pi)))
        candidates.append({"cx": cx, "cy": cy, "area": area, "source_radius": source_radius, "peak": float(lum[slc][mask].max())})
    candidates.sort(key=lambda item: item["peak"], reverse=True)
    deduped: list[dict] = []
    for candidate in candidates:
        if all(np.hypot(candidate["cx"] - other["cx"], candidate["cy"] - other["cy"]) > 80.0 for other in deduped):
            deduped.append(candidate)
    return deduped


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_contact_sheet(metrics: list[dict], output: Path) -> None:
    font = ImageFont.load_default()
    tiles: list[Image.Image] = []
    for row in metrics:
        image = Image.open(row["patch_path"]).convert("RGB")
        image.thumbnail((180, 180), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (180, 222), "white")
        tile.paste(image, ((180 - image.width) // 2, 0))
        draw = ImageDraw.Draw(tile)
        label_text = f"{row['source_id'][:26]}\nr={row['visible_radius_px']:.1f} rg={row['red_green_ratio_outer']:.2f}\nconf={row['halo_confidence']:.2f}"
        draw.text((5, 184), label_text, fill="black", font=font)
        tiles.append(tile)
    if not tiles:
        return
    cols = min(4, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * 180, rows * 222 + 28), "white")
    ImageDraw.Draw(sheet).text((8, 8), "real-photo candidate halation patches", fill="black", font=font)
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % cols) * 180, 28 + (index // cols) * 222))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "PNG")


def main() -> int:
    args = parse_args()
    patch_dir = args.output_root / "patches" / "real"
    metrics_dir = args.output_root / "metrics"
    contact_dir = args.output_root / "contact_sheets"
    patch_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict] = []
    for row in read_manifest(args.manifest):
        local_path = row.get("local_path") or ""
        if not local_path:
            continue
        rgb = load_rgb(Path(local_path))
        accepted_for_source = 0
        for index, candidate in enumerate(candidates_from_image(rgb, args.threshold_percentile), start=1):
            cropped = crop_patch(rgb, candidate["cx"], candidate["cy"], args.patch_size)
            if cropped is None:
                continue
            patch, center = cropped
            item = patch_metrics(patch, center, candidate["source_radius"])
            plausible_background = item["background_luma"] <= 0.52
            plausible_color = item["red_green_ratio_outer"] >= 1.08 or item["orange_core_score"] >= 0.08
            plausible_source = item["source_red_green_ratio"] >= 1.02 or item["orange_core_score"] >= 0.12
            plausible_radius = item["visible_radius_px"] <= args.patch_size * 0.65
            if item["halo_confidence"] < args.min_confidence or not (
                plausible_background and plausible_color and plausible_source and plausible_radius
            ):
                continue
            patch_id = f"{row['source_id']}_p{index:02d}"
            patch_path = patch_dir / f"{patch_id}.png"
            save_rgb(patch, patch_path)
            item.update(
                {
                    "patch_id": patch_id,
                    "source_id": row["source_id"],
                    "patch_path": str(patch_path.as_posix()),
                    "source_page_url": row.get("page_url", ""),
                    "license": row.get("license", ""),
                    "source_radius_px": float(candidate["source_radius"]),
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                }
            )
            metrics.append(item)
            accepted_for_source += 1
            if accepted_for_source >= args.max_patches_per_source:
                break
            if len(metrics) >= args.max_patches:
                break
        if len(metrics) >= args.max_patches:
            break
    metrics.sort(key=lambda item: item["halo_confidence"], reverse=True)
    metrics_path = metrics_dir / "real_patch_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    make_contact_sheet(metrics, contact_dir / "real_patches_contact_sheet.png")
    print(metrics_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
