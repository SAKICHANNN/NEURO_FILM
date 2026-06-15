#!/usr/bin/env python3
"""Extract response stats from high-precision FiveK freeze-pack TIFFs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract high-precision FiveK freeze response stats.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs" / "fivek_auto_optimize" / "freeze_v1" / "manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "fivek_auto_optimize" / "freeze_v1" / "response_stats_filtered_srgb16",
    )
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--sample-stride", type=int, default=3)
    return parser.parse_args()


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_tiff(path: Path, stride: int) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.dtype != np.uint16:
        raise ValueError(f"Expected uint16 TIFF: {path}, got {arr.dtype}")
    rgb = arr[..., :3].astype(np.float32) / 65535.0
    if stride > 1:
        rgb = rgb[::stride, ::stride]
    return rgb


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def chroma(rgb: np.ndarray) -> np.ndarray:
    return np.sqrt(((rgb - rgb.mean(axis=2, keepdims=True)) ** 2).sum(axis=2))


def response_curves(rows: list[dict[str, str]], bins: int, stride: int) -> tuple[dict, list[dict], dict[str, np.ndarray]]:
    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
    centers = (edges[:-1] + edges[1:]) / 2.0
    luma_sum = np.zeros(bins, dtype=np.float64)
    luma_delta_sum = np.zeros(bins, dtype=np.float64)
    chroma_sum = np.zeros(bins, dtype=np.float64)
    chroma_delta_sum = np.zeros(bins, dtype=np.float64)
    rgb_delta_sum = np.zeros((bins, 3), dtype=np.float64)
    counts = np.zeros(bins, dtype=np.float64)
    per_image: list[dict] = []

    for row in rows:
        raw = load_tiff(ROOT / row["raw_default_srgb16"], stride)
        target = load_tiff(ROOT / row["filtered_target_srgb16"], stride)
        raw_l = luma(raw)
        target_l = luma(target)
        raw_c = chroma(raw)
        target_c = chroma(target)
        idx = np.clip(np.digitize(raw_l.reshape(-1), edges) - 1, 0, bins - 1)
        target_l_flat = target_l.reshape(-1)
        raw_l_flat = raw_l.reshape(-1)
        raw_c_flat = raw_c.reshape(-1)
        target_c_flat = target_c.reshape(-1)
        delta_rgb = (target - raw).reshape(-1, 3)

        for bin_index in range(bins):
            mask = idx == bin_index
            if not mask.any():
                continue
            n = int(mask.sum())
            counts[bin_index] += n
            luma_sum[bin_index] += float(target_l_flat[mask].sum())
            luma_delta_sum[bin_index] += float((target_l_flat[mask] - raw_l_flat[mask]).sum())
            chroma_sum[bin_index] += float(target_c_flat[mask].sum())
            chroma_delta_sum[bin_index] += float((target_c_flat[mask] - raw_c_flat[mask]).sum())
            rgb_delta_sum[bin_index] += delta_rgb[mask].sum(axis=0)

        per_image.append(
            {
                "id": row["id"],
                "raw_luma_mean": float(raw_l.mean()),
                "target_luma_mean": float(target_l.mean()),
                "luma_delta_mean": float((target_l - raw_l).mean()),
                "raw_chroma_mean": float(raw_c.mean()),
                "target_chroma_mean": float(target_c.mean()),
                "chroma_delta_mean": float((target_c - raw_c).mean()),
                "rgb_delta_mean_r": float(delta_rgb[:, 0].mean()),
                "rgb_delta_mean_g": float(delta_rgb[:, 1].mean()),
                "rgb_delta_mean_b": float(delta_rgb[:, 2].mean()),
            }
        )

    safe_counts = np.maximum(counts, 1.0)
    curves = {
        "bin_centers": centers.tolist(),
        "counts": counts.astype(int).tolist(),
        "target_luma_by_raw_luma": (luma_sum / safe_counts).tolist(),
        "luma_delta_by_raw_luma": (luma_delta_sum / safe_counts).tolist(),
        "target_chroma_by_raw_luma": (chroma_sum / safe_counts).tolist(),
        "chroma_delta_by_raw_luma": (chroma_delta_sum / safe_counts).tolist(),
        "rgb_delta_by_raw_luma": (rgb_delta_sum / safe_counts[:, None]).tolist(),
    }
    arrays = {
        "bin_centers": centers.astype(np.float32),
        "counts": counts.astype(np.float32),
        "target_luma_by_raw_luma": (luma_sum / safe_counts).astype(np.float32),
        "luma_delta_by_raw_luma": (luma_delta_sum / safe_counts).astype(np.float32),
        "target_chroma_by_raw_luma": (chroma_sum / safe_counts).astype(np.float32),
        "chroma_delta_by_raw_luma": (chroma_delta_sum / safe_counts).astype(np.float32),
        "rgb_delta_by_raw_luma": (rgb_delta_sum / safe_counts[:, None]).astype(np.float32),
    }
    return curves, per_image, arrays


def make_diagnostic_plot(curves: dict, output: Path) -> None:
    width, height = 900, 520
    margin = 52
    plot_w = width - margin * 2
    plot_h = height - margin * 2
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((margin, 16), "FiveK freeze response stats: raw sRGB16 -> filtered target sRGB16", fill="black", font=font)
    draw.rectangle((margin, margin, margin + plot_w, margin + plot_h), outline=(180, 180, 180))
    centers = np.asarray(curves["bin_centers"], dtype=np.float32)
    target_l = np.asarray(curves["target_luma_by_raw_luma"], dtype=np.float32)
    delta_l = np.asarray(curves["luma_delta_by_raw_luma"], dtype=np.float32)
    chroma_delta = np.asarray(curves["chroma_delta_by_raw_luma"], dtype=np.float32)

    def xy(x: float, y: float) -> tuple[int, int]:
        return (
            int(margin + np.clip(x, 0, 1) * plot_w),
            int(margin + (1.0 - np.clip(y, 0, 1)) * plot_h),
        )

    identity = [xy(float(v), float(v)) for v in centers]
    target = [xy(float(x), float(y)) for x, y in zip(centers, target_l)]
    delta = [xy(float(x), float(0.5 + y * 2.0)) for x, y in zip(centers, delta_l)]
    chroma = [xy(float(x), float(0.5 + y * 5.0)) for x, y in zip(centers, chroma_delta)]
    if len(identity) > 1:
        draw.line(identity, fill=(160, 160, 160), width=2)
        draw.line(target, fill=(20, 90, 210), width=3)
        draw.line(delta, fill=(210, 90, 20), width=2)
        draw.line(chroma, fill=(30, 150, 80), width=2)
    draw.text((margin, height - 34), "gray: identity, blue: target luma, orange: luma delta x2 + 0.5, green: chroma delta x5 + 0.5", fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG")


def main() -> int:
    args = parse_args()
    rows = read_manifest(args.manifest)
    if not rows:
        raise ValueError(f"No rows in manifest: {args.manifest}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    curves, per_image, arrays = response_curves(rows, args.bins, args.sample_stride)
    (output_dir / "response_stats.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source_manifest": repo_path(args.manifest),
                "row_count": len(rows),
                "bins": args.bins,
                "sample_stride": args.sample_stride,
                "curves": curves,
                "note": "High-precision response stats fit from freeze-pack uint16 sRGB TIFFs.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "per_image_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)
    np.savez(output_dir / "response_curves.npz", **arrays)
    make_diagnostic_plot(curves, output_dir / "response_curves.png")
    print(f"rows={len(rows)} bins={args.bins}")
    print(repo_path(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
