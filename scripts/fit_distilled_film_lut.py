#!/usr/bin/env python3
"""Fit a fast distilled SepLUT from existing film-response teacher outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.neural_film_lut.distilled_seplut import (  # noqa: E402
    DISTILLED_STYLE_NAMES,
    apply_1d_lut_np,
    apply_3d_residual_np,
    image_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit distilled per-stock SepLUT.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "outputs" / "color_baseline" / "velvia50_rawpixls20_s0p50_gamutsafe" / "manifest.json",
    )
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=ROOT / "outputs" / "eval" / "color_engine_challenge" / "film_response_v1_s1p0",
    )
    parser.add_argument("--styles", default=",".join(DISTILLED_STYLE_NAMES))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-side", type=int, default=256)
    parser.add_argument("--lut1d-size", type=int, default=33)
    parser.add_argument("--lut3d-size", type=int, default=17)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "neural_film_lut_v2" / "scheme_a_distilled")
    return parser.parse_args()


def source_paths(path: Path, limit: int) -> list[Path]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if limit > 0:
        rows = rows[:limit]
    return [Path(row.get("before") or row.get("input")) for row in rows if Path(row.get("before") or row.get("input")).exists()]


def load_rgb(path: Path, max_side: int) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def fit_1d_lut(source: np.ndarray, target: np.ndarray, size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    lut = np.stack([axis, axis, axis], axis=0)
    for channel in range(3):
        coords = np.clip(np.rint(source[:, channel] * (size - 1)).astype(np.int32), 0, size - 1)
        sums = np.bincount(coords, weights=target[:, channel], minlength=size).astype(np.float32)
        counts = np.bincount(coords, minlength=size).astype(np.float32)
        known = counts > 0
        if known.sum() >= 2:
            values = sums[known] / counts[known]
            lut[channel] = np.interp(axis, axis[known], values).astype(np.float32)
        elif known.sum() == 1:
            lut[channel] = sums[known][0] / counts[known][0]
        lut[channel] = gaussian_filter(lut[channel], sigma=0.65)
        lut[channel] = np.clip(lut[channel], 0.0, 1.0)
    return lut


def fit_3d_residual(source_after_1d: np.ndarray, target: np.ndarray, size: int) -> np.ndarray:
    residual_sum = np.zeros((size, size, size, 3), dtype=np.float32)
    count = np.zeros((size, size, size, 1), dtype=np.float32)
    coords = np.clip(np.rint(source_after_1d * (size - 1)).astype(np.int32), 0, size - 1)
    residual = target - source_after_1d
    np.add.at(residual_sum, (coords[:, 0], coords[:, 1], coords[:, 2]), residual)
    np.add.at(count, (coords[:, 0], coords[:, 1], coords[:, 2]), 1.0)
    residual_grid = np.divide(residual_sum, np.maximum(count, 1.0), out=np.zeros_like(residual_sum), where=count > 0)
    for channel in range(3):
        residual_grid[..., channel] = gaussian_filter(residual_grid[..., channel], sigma=0.85)
    return np.clip(residual_grid, -0.22, 0.22).astype(np.float32)


def fit_gate_weights(features: list[np.ndarray], desired: list[float]) -> np.ndarray:
    x = np.stack(features).astype(np.float32)
    y = np.asarray(desired, dtype=np.float32)
    ridge = np.eye(x.shape[1], dtype=np.float32) * 0.05
    weights = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    return weights.astype(np.float32)


def optimal_gate(source: np.ndarray, base: np.ndarray, target: np.ndarray) -> float:
    delta = (base - source).reshape(-1)
    want = (target - source).reshape(-1)
    denom = float(np.dot(delta, delta))
    if denom <= 1e-8:
        return 1.0
    return float(np.clip(np.dot(delta, want) / denom, 0.45, 1.65))


def main() -> int:
    args = parse_args()
    styles = [item.strip() for item in args.styles.split(",") if item.strip()]
    paths = source_paths(args.source_manifest, args.limit)
    lut1d_all = []
    residual3d_all = []
    gate_weights_all = []
    report = {"styles": {}, "teacher_root": str(args.teacher_root), "image_count": len(paths)}

    for style in styles:
        source_pixels = []
        target_pixels = []
        per_image = []
        for index, input_path in enumerate(paths, start=1):
            teacher_path = args.teacher_root / style / "after" / f"{index:02d}_{style}_film_response.png"
            if not teacher_path.exists():
                continue
            source = load_rgb(input_path, args.max_side)
            target = load_rgb(teacher_path, args.max_side)
            height = min(source.shape[0], target.shape[0])
            width = min(source.shape[1], target.shape[1])
            source = source[:height, :width]
            target = target[:height, :width]
            source_pixels.append(source[:: args.sample_stride, :: args.sample_stride].reshape(-1, 3))
            target_pixels.append(target[:: args.sample_stride, :: args.sample_stride].reshape(-1, 3))
            per_image.append((source, target))
        if not source_pixels:
            raise ValueError(f"No teacher pairs for {style}")
        source_flat = np.concatenate(source_pixels, axis=0)
        target_flat = np.concatenate(target_pixels, axis=0)
        lut1d = fit_1d_lut(source_flat, target_flat, args.lut1d_size)
        after_1d_flat = apply_1d_lut_np(source_flat.reshape(-1, 1, 3), lut1d).reshape(-1, 3)
        residual3d = fit_3d_residual(after_1d_flat, target_flat, args.lut3d_size)

        features = []
        desired = []
        for source, target in per_image:
            after_1d = apply_1d_lut_np(source, lut1d)
            base = np.clip(after_1d + apply_3d_residual_np(after_1d, residual3d), 0.0, 1.0)
            features.append(image_features(source))
            desired.append(optimal_gate(source, base, target))
        weights = fit_gate_weights(features, desired)

        lut1d_all.append(lut1d)
        residual3d_all.append(residual3d)
        gate_weights_all.append(weights)
        report["styles"][style] = {
            "pixel_count": int(source_flat.shape[0]),
            "gate_min": float(min(desired)),
            "gate_max": float(max(desired)),
            "gate_mean": float(np.mean(desired)),
        }
        print(f"{style} pixels={source_flat.shape[0]} gate={np.mean(desired):.3f}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "model.npz",
        styles=np.asarray(styles),
        lut1d=np.stack(lut1d_all),
        residual3d=np.stack(residual3d_all),
        gate_weights=np.stack(gate_weights_all),
        lut1d_size=np.asarray([args.lut1d_size]),
        lut3d_size=np.asarray([args.lut3d_size]),
    )
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_dir / "model.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
