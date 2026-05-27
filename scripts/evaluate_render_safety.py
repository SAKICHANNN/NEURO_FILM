#!/usr/bin/env python3
"""Evaluate content-preserving render safety for a before/after image pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure clipping, structure, and color artifacts.")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--diff-map", type=Path, default=None)
    parser.add_argument("--output-margin", type=int, default=0, help="Expected 8-bit headroom, e.g. 4 means [4, 251].")
    parser.add_argument("--fail-on-clip", action="store_true")
    parser.add_argument("--fail-on-l-ssim", action="store_true")
    parser.add_argument("--min-l-ssim", type=float, default=0.995)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def load_rgb_u8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image, dtype=np.uint8)


def clipped_mask(arr: np.ndarray) -> np.ndarray:
    return np.any((arr <= 0) | (arr >= 255), axis=2)


def percentage(mask: np.ndarray) -> float:
    return float(mask.mean() * 100.0)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def gradient_magnitude(luminance: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(luminance.astype(np.float32))
    return np.hypot(grad_x, grad_y)


def banding_metrics(luminance_u8: np.ndarray) -> dict[str, float | int]:
    values = luminance_u8.reshape(-1)
    low = int(np.percentile(values, 1))
    high = int(np.percentile(values, 99))
    if high <= low:
        return {"occupied_bins": 1, "empty_bin_fraction": 1.0, "max_histogram_bin_fraction": 1.0}
    hist = np.bincount(values, minlength=256)[low : high + 1]
    occupied = int(np.count_nonzero(hist))
    empty_fraction = 1.0 - occupied / float(len(hist))
    max_bin_fraction = float(hist.max() / max(1, hist.sum()))
    return {
        "occupied_bins": occupied,
        "empty_bin_fraction": float(empty_fraction),
        "max_histogram_bin_fraction": max_bin_fraction,
    }


def hue_chroma_metrics(before_lab: np.ndarray, after_lab: np.ndarray) -> dict[str, float]:
    before_ab = before_lab[..., 1:3]
    after_ab = after_lab[..., 1:3]
    before_chroma = np.linalg.norm(before_ab, axis=2)
    after_chroma = np.linalg.norm(after_ab, axis=2)
    chroma_delta = after_chroma - before_chroma

    neutral_mask = before_chroma < 8.0
    skin_mask = (
        (before_lab[..., 0] > 20.0)
        & (before_lab[..., 0] < 92.0)
        & (before_lab[..., 1] > 4.0)
        & (before_lab[..., 1] < 28.0)
        & (before_lab[..., 2] > 4.0)
        & (before_lab[..., 2] < 46.0)
    )
    neutral_contamination = neutral_mask & (after_chroma > 14.0)
    skin_chroma_delta = np.abs(chroma_delta[skin_mask]) if np.any(skin_mask) else np.asarray([], dtype=np.float32)

    return {
        "before_chroma_mean": float(before_chroma.mean()),
        "after_chroma_mean": float(after_chroma.mean()),
        "chroma_delta_mean": float(chroma_delta.mean()),
        "chroma_delta_p95_abs": percentile(np.abs(chroma_delta), 95),
        "neutral_pixel_percent": percentage(neutral_mask),
        "neutral_contaminated_percent": percentage(neutral_contamination),
        "skin_pixel_percent": percentage(skin_mask),
        "skin_chroma_delta_p95_abs": percentile(skin_chroma_delta, 95) if skin_chroma_delta.size else 0.0,
    }


def make_diff_map(before: np.ndarray, after: np.ndarray, path: Path) -> None:
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16)).astype(np.float32)
    diff = np.clip(diff * 4.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(diff, mode="RGB").save(path, "PNG")


def evaluate(before_path: Path, after_path: Path) -> dict[str, Any]:
    before = load_rgb_u8(before_path)
    after = load_rgb_u8(after_path)
    if before.shape != after.shape:
        raise ValueError(f"Image dimensions differ: before={before.shape} after={after.shape}")

    before_float = before.astype(np.float32) / 255.0
    after_float = after.astype(np.float32) / 255.0
    before_lab = rgb2lab(before_float)
    after_lab = rgb2lab(after_float)
    before_l = before_lab[..., 0]
    after_l = after_lab[..., 0]

    before_clip = clipped_mask(before)
    after_clip = clipped_mask(after)
    new_clip = after_clip & ~before_clip
    before_grad = gradient_magnitude(before_l)
    after_grad = gradient_magnitude(after_l)
    grad_abs_delta = np.abs(after_grad - before_grad)
    before_hf = before_l - gaussian_filter(before_l, sigma=1.2)
    after_hf = after_l - gaussian_filter(after_l, sigma=1.2)
    hf_abs_delta = np.abs(after_hf - before_hf)
    before_l_u8 = np.rint(np.clip(before_l / 100.0 * 255.0, 0, 255)).astype(np.uint8)
    after_l_u8 = np.rint(np.clip(after_l / 100.0 * 255.0, 0, 255)).astype(np.uint8)

    return {
        "before": str(before_path),
        "after": str(after_path),
        "shape": list(before.shape),
        "pixel_count": int(before.shape[0] * before.shape[1]),
        "clip": {
            "before_min": int(before.min()),
            "before_max": int(before.max()),
            "after_min": int(after.min()),
            "after_max": int(after.max()),
            "before_clipped_pixel_percent": percentage(before_clip),
            "after_clipped_pixel_percent": percentage(after_clip),
            "new_clipped_pixel_percent": percentage(new_clip),
            "new_clipped_pixel_count": int(new_clip.sum()),
            "after_zero_channel_count": int((after <= 0).sum()),
            "after_255_channel_count": int((after >= 255).sum()),
        },
        "structure": {
            "l_ssim": float(structural_similarity(before_l, after_l, data_range=100.0)),
            "gradient_delta_mean": float(grad_abs_delta.mean()),
            "gradient_delta_p95": percentile(grad_abs_delta, 95),
            "gradient_delta_max": float(grad_abs_delta.max()),
            "high_frequency_delta_mean": float(hf_abs_delta.mean()),
            "high_frequency_delta_p95": percentile(hf_abs_delta, 95),
            "high_frequency_energy_ratio": float(after_hf.std() / max(float(before_hf.std()), 1e-6)),
        },
        "banding": {
            "before": banding_metrics(before_l_u8),
            "after": banding_metrics(after_l_u8),
        },
        "color": hue_chroma_metrics(before_lab, after_lab),
    }


def main() -> int:
    args = parse_args()
    metrics = evaluate(args.before, args.after)
    failures: list[str] = []

    if args.diff_map:
        make_diff_map(load_rgb_u8(args.before), load_rgb_u8(args.after), args.diff_map)
        metrics["diff_map"] = str(args.diff_map)

    clip = metrics["clip"]
    if args.output_margin > 0:
        low = int(args.output_margin)
        high = 255 - low
        metrics["clip"]["output_margin"] = low
        metrics["clip"]["within_output_margin"] = bool(clip["after_min"] >= low and clip["after_max"] <= high)
        if args.fail_on_clip and not metrics["clip"]["within_output_margin"]:
            failures.append(f"output outside [{low}, {high}]")

    if args.fail_on_clip and clip["new_clipped_pixel_count"] > 0:
        failures.append(f"new clipped pixels: {clip['new_clipped_pixel_count']}")
    if args.fail_on_l_ssim and metrics["structure"]["l_ssim"] < args.min_l_ssim:
        failures.append(f"L_ssim {metrics['structure']['l_ssim']:.6f} < {args.min_l_ssim:.6f}")

    metrics["passed"] = not failures
    metrics["failures"] = failures

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2 if args.pretty else None))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
