#!/usr/bin/env python3
"""Fit distilled NILUT and context-4D variants from teacher outputs."""

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

from src.models.neural_film_lut.distilled_seplut import DISTILLED_STYLE_NAMES, image_features  # noqa: E402
from src.models.neural_film_lut.distilled_variants import (  # noqa: E402
    CONTEXT_NAMES,
    apply_distilled_context4d,
    apply_distilled_nilut,
    context_weights,
    nilut_basis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit distilled neural LUT V2 variants.")
    parser.add_argument("--scheme", choices=["nilut", "context4d"], required=True)
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
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--lut3d-size", type=int, default=13)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def source_paths(path: Path, limit: int) -> list[Path]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if limit > 0:
        rows = rows[:limit]
    paths = []
    for row in rows:
        candidate = Path(row.get("before") or row.get("input") or "")
        if candidate.exists():
            paths.append(candidate)
    if not paths:
        raise ValueError(f"No source images found in {path}")
    return paths


def load_rgb(path: Path, max_side: int) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def fit_gate_weights(features: list[np.ndarray], desired: list[float]) -> np.ndarray:
    x = np.stack(features).astype(np.float32)
    y = np.asarray(desired, dtype=np.float32)
    ridge = np.eye(x.shape[1], dtype=np.float32) * 0.05
    return np.linalg.solve(x.T @ x + ridge, x.T @ y).astype(np.float32)


def optimal_gate(source: np.ndarray, base: np.ndarray, target: np.ndarray) -> float:
    delta = (base - source).reshape(-1)
    want = (target - source).reshape(-1)
    denom = float(np.dot(delta, delta))
    if denom <= 1e-8:
        return 1.0
    return float(np.clip(np.dot(delta, want) / denom, 0.45, 1.65))


def teacher_pairs(args: argparse.Namespace, style: str, paths: list[Path]) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for index, input_path in enumerate(paths, start=1):
        teacher_path = args.teacher_root / style / "after" / f"{index:02d}_{style}_film_response.png"
        if not teacher_path.exists():
            continue
        source = load_rgb(input_path, args.max_side)
        target = load_rgb(teacher_path, args.max_side)
        height = min(source.shape[0], target.shape[0])
        width = min(source.shape[1], target.shape[1])
        pairs.append((source[:height, :width], target[:height, :width]))
    if not pairs:
        raise ValueError(f"No teacher pairs found for {style}")
    return pairs


def fit_nilut(pairs: list[tuple[np.ndarray, np.ndarray]], ridge_value: float, sample_stride: int) -> tuple[np.ndarray, np.ndarray]:
    basis_parts = []
    residual_parts = []
    for source, target in pairs:
        source_sample = source[::sample_stride, ::sample_stride]
        target_sample = target[::sample_stride, ::sample_stride]
        basis_parts.append(nilut_basis(source_sample).reshape(-1, 13))
        residual_parts.append((target_sample - source_sample).reshape(-1, 3))
    x = np.concatenate(basis_parts, axis=0)
    y = np.concatenate(residual_parts, axis=0)
    ridge = np.eye(x.shape[1], dtype=np.float32) * float(ridge_value)
    coeffs = np.linalg.solve(x.T @ x + ridge, x.T @ y).astype(np.float32)

    features = []
    desired = []
    for source, target in pairs:
        residual = np.clip(nilut_basis(source) @ coeffs, -0.22, 0.22)
        base = np.clip(source + residual, 0.0, 1.0)
        features.append(image_features(source))
        desired.append(optimal_gate(source, base, target))
    return coeffs, fit_gate_weights(features, desired)


def fit_context4d(
    pairs: list[tuple[np.ndarray, np.ndarray]], size: int, sample_stride: int
) -> tuple[np.ndarray, np.ndarray]:
    residual_sum = np.zeros((len(CONTEXT_NAMES), size, size, size, 3), dtype=np.float32)
    count = np.zeros((len(CONTEXT_NAMES), size, size, size, 1), dtype=np.float32)
    for source, target in pairs:
        source_sample = source[::sample_stride, ::sample_stride]
        target_sample = target[::sample_stride, ::sample_stride]
        coords = np.clip(np.rint(source_sample * (size - 1)).astype(np.int32), 0, size - 1)
        residual = target_sample - source_sample
        weights = context_weights(source_sample)
        for context_index in range(len(CONTEXT_NAMES)):
            weight = weights[..., context_index : context_index + 1]
            np.add.at(
                residual_sum,
                (context_index, coords[..., 0], coords[..., 1], coords[..., 2]),
                residual * weight,
            )
            np.add.at(
                count,
                (context_index, coords[..., 0], coords[..., 1], coords[..., 2]),
                weight,
            )
    residual_grid = np.divide(residual_sum, np.maximum(count, 1e-6), out=np.zeros_like(residual_sum), where=count > 0)
    for context_index in range(len(CONTEXT_NAMES)):
        for channel in range(3):
            residual_grid[context_index, ..., channel] = gaussian_filter(
                residual_grid[context_index, ..., channel], sigma=0.90
            )
    residual_grid = np.clip(residual_grid, -0.22, 0.22).astype(np.float32)

    features = []
    desired = []
    for source, target in pairs:
        base, _gate = apply_distilled_context4d(
            source,
            style_index=0,
            residual4d=residual_grid[None],
            gate_weights=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            strength=1.0,
            output_margin=0,
        )
        features.append(image_features(source))
        desired.append(optimal_gate(source, base, target))
    return residual_grid, fit_gate_weights(features, desired)


def main() -> int:
    args = parse_args()
    styles = [item.strip() for item in args.styles.split(",") if item.strip()]
    paths = source_paths(args.source_manifest, args.limit)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT / "outputs" / "neural_film_lut_v2" / f"scheme_{'b' if args.scheme == 'nilut' else 'c'}_{args.scheme}_distilled_v1"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "scheme": args.scheme,
        "styles": {},
        "teacher_root": str(args.teacher_root),
        "image_count": len(paths),
        "context_names": CONTEXT_NAMES if args.scheme == "context4d" else [],
    }
    gate_weights = []

    if args.scheme == "nilut":
        coeffs_all = []
        for style in styles:
            pairs = teacher_pairs(args, style, paths)
            coeffs, weights = fit_nilut(pairs, args.ridge, args.sample_stride)
            coeffs_all.append(coeffs)
            gate_weights.append(weights)
            report["styles"][style] = {"pair_count": len(pairs), "basis_count": int(coeffs.shape[0])}
            print(f"{style} pairs={len(pairs)} coeff_norm={float(np.linalg.norm(coeffs)):.4f}")
        np.savez_compressed(
            output_dir / "model.npz",
            scheme=np.asarray(["nilut"]),
            styles=np.asarray(styles),
            coeffs=np.stack(coeffs_all),
            gate_weights=np.stack(gate_weights),
        )
    else:
        residual_all = []
        for style in styles:
            pairs = teacher_pairs(args, style, paths)
            residual_grid, weights = fit_context4d(pairs, args.lut3d_size, args.sample_stride)
            residual_all.append(residual_grid)
            gate_weights.append(weights)
            report["styles"][style] = {
                "pair_count": len(pairs),
                "lut3d_size": args.lut3d_size,
                "context_count": len(CONTEXT_NAMES),
            }
            print(f"{style} pairs={len(pairs)} residual_abs={float(np.abs(residual_grid).mean()):.5f}")
        np.savez_compressed(
            output_dir / "model.npz",
            scheme=np.asarray(["context4d"]),
            styles=np.asarray(styles),
            residual4d=np.stack(residual_all),
            gate_weights=np.stack(gate_weights),
            context_names=np.asarray(CONTEXT_NAMES),
            lut3d_size=np.asarray([args.lut3d_size]),
        )

    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_dir / "model.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
