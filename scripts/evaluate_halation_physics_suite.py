#!/usr/bin/env python3
"""Synthetic physics checks for physical-prior halation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx import composite_layers, physical_halation_layer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synthetic halation physics metrics.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "eval" / "halation_v2p1_physics")
    parser.add_argument("--profile", choices=("vision3_500t", "cinestill_800t"), default="cinestill_800t")
    parser.add_argument("--amplify", type=float, default=1.15)
    parser.add_argument("--impact", type=float, default=0.90)
    parser.add_argument("--source-limiter", type=float, default=1.8)
    parser.add_argument("--local-diffusion", type=float, default=1.25)
    parser.add_argument("--global-diffusion", type=float, default=0.18)
    parser.add_argument("--hue-green", type=float, default=0.34)
    parser.add_argument("--background-gain", type=float, default=1.45)
    parser.add_argument("--background-luma-target", type=float, default=0.20)
    parser.add_argument("--visible-alpha", type=float, default=0.01)
    return parser.parse_args()


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    display = linear / (1.0 + linear)
    return np.power(np.clip(display, 0.0, 1.0), 1.0 / 2.2).astype(np.float32)


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path, "PNG")


def layer_views(layer) -> tuple[np.ndarray, np.ndarray]:
    alpha = layer.alpha
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    black = np.clip(layer.rgb * alpha, 0.0, 1.0)
    white = np.clip(1.0 * (1.0 - alpha) + layer.rgb * alpha, 0.0, 1.0)
    return black, white


def point_light_scene(stop: float, *, size: int = 512, dark_side: bool = True) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    y, x = np.mgrid[0:size, 0:size]
    center = (size // 2, size // 2)
    radius = 8.0
    background = 0.012 if dark_side else 0.42
    source_linear = np.full((size, size, 3), background, dtype=np.float32)
    dist = np.hypot(x - center[0], y - center[1])
    core = dist <= radius
    # Stops are measured above middle gray in scene-linear space.
    exposure = 0.18 * (2.0 ** stop)
    source_linear[core] = exposure
    source_linear[..., 1] *= 0.98
    source_linear[..., 2] *= 0.94
    display = linear_to_srgb(source_linear)
    return display, source_linear, center


def visible_radius(alpha: np.ndarray, center: tuple[int, int], threshold: float) -> float:
    yy, xx = np.mgrid[0 : alpha.shape[0], 0 : alpha.shape[1]]
    dist = np.hypot(xx - center[0], yy - center[1])
    visible = alpha > threshold
    if not np.any(visible):
        return 0.0
    return float(np.percentile(dist[visible], 98.0))


def radial_profile(values: np.ndarray, center: tuple[int, int], max_radius: int = 140) -> list[float]:
    yy, xx = np.mgrid[0 : values.shape[0], 0 : values.shape[1]]
    dist = np.hypot(xx - center[0], yy - center[1]).astype(np.int32)
    profile = []
    for radius in range(max_radius + 1):
        mask = dist == radius
        profile.append(float(values[mask].mean()) if np.any(mask) else 0.0)
    return profile


def make_contact_sheet(rows: list[dict], output: Path, title: str) -> None:
    font = ImageFont.load_default()
    tiles = []
    for row in rows:
        images = [Image.open(row[key]).convert("RGB") for key in ("original", "layer_black", "layer_white", "combined")]
        for image in images:
            image.thumbnail((210, 160), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (840, 202), "white")
        draw = ImageDraw.Draw(tile)
        draw.text(
            (8, 8),
            f"{row['label']} radius={row['visible_radius_px']:.1f}px blue={row['blue_leakage']:.4f}",
            fill="black",
            font=font,
        )
        for index, image in enumerate(images):
            tile.paste(image, (index * 210 + (210 - image.width) // 2, 34))
        labels = ["original", "layer on black", "layer on white", "combined"]
        for index, label in enumerate(labels):
            draw.text((index * 210 + 8, 182), label, fill="black", font=font)
        tiles.append(tile)
    sheet = Image.new("RGB", (840, 34 + len(tiles) * 202), "white")
    ImageDraw.Draw(sheet).text((8, 10), title, fill="black", font=font)
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (0, 34 + index * 202))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "PNG")


def render_layer(base: np.ndarray, source_linear: np.ndarray, args: argparse.Namespace):
    return physical_halation_layer(
        base,
        source_linear_rgb=source_linear,
        source_normalization="none",
        profile=args.profile,
        amplify=args.amplify,
        impact=args.impact,
        source_limiter_stops=args.source_limiter,
        local_diffusion=args.local_diffusion,
        global_diffusion=args.global_diffusion,
        hue_green=args.hue_green,
        background_gain=args.background_gain,
        background_luma_target=args.background_luma_target,
        no_remjet=1.0 if args.profile == "cinestill_800t" else 0.42,
    )


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    stops = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    rows = []
    radii = []
    for stop in stops:
        base, source_linear, center = point_light_scene(stop, dark_side=True)
        layer = render_layer(base, source_linear, args)
        combined = composite_layers(base, [layer], output_margin=4)
        black, white = layer_views(layer)
        alpha = layer.alpha[..., 0] if layer.alpha.ndim == 3 else layer.alpha
        radius = visible_radius(alpha, center, args.visible_alpha)
        radii.append(radius)
        profile = radial_profile(alpha, center)
        center_mask = np.zeros(alpha.shape, dtype=bool)
        yy, xx = np.mgrid[0 : alpha.shape[0], 0 : alpha.shape[1]]
        dist = np.hypot(xx - center[0], yy - center[1])
        center_mask |= dist < 14
        outer_mask = (dist >= max(radius * 0.55, 18.0)) & (dist <= max(radius, 24.0))
        weighted = layer.rgb * alpha[..., None]
        red = weighted[..., 0]
        green = weighted[..., 1]
        blue = weighted[..., 2]
        row = {
            "label": f"stop_{stop:.1f}",
            "stop": stop,
            "visible_radius_px": radius,
            "alpha_max": float(alpha.max()),
            "alpha_mean": float(alpha.mean()),
            "center_green_red_ratio": float(green[center_mask].sum() / max(red[center_mask].sum(), 1e-8)),
            "outer_green_red_ratio": float(green[outer_mask].sum() / max(red[outer_mask].sum(), 1e-8)) if np.any(outer_mask) else 0.0,
            "blue_leakage": float(blue.sum() / max(red.sum() + green.sum(), 1e-8)),
            "tail_ratio_r48_r12": float(profile[48] / max(profile[12], 1e-8)),
        }
        run_dir = output_root / "exposure_radius"
        row["original"] = str(run_dir / f"{row['label']}_original.png")
        row["layer_black"] = str(run_dir / f"{row['label']}_layer_black.png")
        row["layer_white"] = str(run_dir / f"{row['label']}_layer_white.png")
        row["combined"] = str(run_dir / f"{row['label']}_combined.png")
        save_rgb(base, Path(row["original"]))
        save_rgb(black, Path(row["layer_black"]))
        save_rgb(white, Path(row["layer_white"]))
        save_rgb(combined, Path(row["combined"]))
        rows.append(row)

    radius_monotonic = all(r2 >= r1 for r1, r2 in zip(radii, radii[1:]))
    radius_gains = [float(r2 - r1) for r1, r2 in zip(radii, radii[1:])]

    dark_base, dark_source, dark_center = point_light_scene(4.0, dark_side=True)
    bright_base, bright_source, bright_center = point_light_scene(4.0, dark_side=False)
    dark_layer = render_layer(dark_base, dark_source, args)
    bright_layer = render_layer(bright_base, bright_source, args)
    dark_alpha = dark_layer.alpha[..., 0] if dark_layer.alpha.ndim == 3 else dark_layer.alpha
    bright_alpha = bright_layer.alpha[..., 0] if bright_layer.alpha.ndim == 3 else bright_layer.alpha
    dark_radius = visible_radius(dark_alpha, dark_center, args.visible_alpha)
    bright_radius = visible_radius(bright_alpha, bright_center, args.visible_alpha)
    background_suppression = {
        "dark_visible_radius_px": dark_radius,
        "bright_visible_radius_px": bright_radius,
        "dark_to_bright_alpha_sum_ratio": float(dark_alpha.sum() / max(bright_alpha.sum(), 1e-8)),
        "dark_to_bright_radius_ratio": float(dark_radius / max(bright_radius, 1e-8)),
    }

    summary = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "exposure_radius_rows": rows,
        "radius_monotonic": radius_monotonic,
        "radius_gains_px": radius_gains,
        "radius_gain_decelerates_after_threshold": bool(len(radius_gains) >= 3 and radius_gains[-1] <= max(radius_gains[0], 1e-6) * 1.35),
        "background_suppression": background_suppression,
        "blue_leakage_max": max(row["blue_leakage"] for row in rows),
        "center_green_ratio_gt_outer_count": sum(
            1 for row in rows if row["center_green_red_ratio"] >= row["outer_green_red_ratio"]
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "halation_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_contact_sheet(rows, output_root / "exposure_radius_contact_sheet.png", "Halation V2.1 Exposure Radius Check")
    print(output_root / "halation_eval.json")
    print(output_root / "exposure_radius_contact_sheet.png")
    print(json.dumps({k: summary[k] for k in ("radius_monotonic", "radius_gains_px", "background_suppression", "blue_leakage_max")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
