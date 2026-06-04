#!/usr/bin/env python3
"""Evaluate physical-prior halation V2 parameter sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx import composite_layers, layer_metrics, physical_halation_layer  # noqa: E402


DEFAULT_RUNS = [
    {
        "name": "vision3_restrained",
        "profile": "vision3_500t",
        "amplify": 0.55,
        "impact": 0.70,
        "source_limiter_stops": 2.8,
        "local_diffusion": 0.85,
        "global_diffusion": 0.08,
        "hue_green": 0.16,
        "background_gain": 1.00,
        "no_remjet": 0.35,
    },
    {
        "name": "vision3_standard",
        "profile": "vision3_500t",
        "amplify": 0.85,
        "impact": 0.85,
        "source_limiter_stops": 2.4,
        "local_diffusion": 1.00,
        "global_diffusion": 0.12,
        "hue_green": 0.22,
        "background_gain": 1.15,
        "no_remjet": 0.45,
    },
    {
        "name": "cinestill_no_remjet",
        "profile": "cinestill_800t",
        "amplify": 1.10,
        "impact": 0.85,
        "source_limiter_stops": 1.9,
        "local_diffusion": 1.25,
        "global_diffusion": 0.22,
        "hue_green": 0.32,
        "background_gain": 1.45,
        "no_remjet": 1.00,
    },
    {
        "name": "cinestill_aggressive",
        "profile": "cinestill_800t",
        "amplify": 1.45,
        "impact": 1.00,
        "source_limiter_stops": 1.6,
        "local_diffusion": 1.55,
        "global_diffusion": 0.34,
        "hue_green": 0.42,
        "background_gain": 1.70,
        "no_remjet": 1.15,
    },
    {
        "name": "impact_low",
        "profile": "cinestill_800t",
        "amplify": 1.35,
        "impact": 0.45,
        "source_limiter_stops": 1.8,
        "local_diffusion": 1.45,
        "global_diffusion": 0.28,
        "hue_green": 0.36,
        "background_gain": 1.55,
        "no_remjet": 1.10,
    },
    {
        "name": "impact_high",
        "profile": "cinestill_800t",
        "amplify": 1.35,
        "impact": 1.00,
        "source_limiter_stops": 1.8,
        "local_diffusion": 1.45,
        "global_diffusion": 0.28,
        "hue_green": 0.36,
        "background_gain": 1.55,
        "no_remjet": 1.10,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate physical halation V2 sweeps.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "outputs" / "color_baseline" / "velvia50_rawpixls20_s0p50_gamutsafe" / "manifest.json",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "eval" / "halation_v2")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--runs-json", type=Path, default=None)
    parser.add_argument("--output-margin", type=int, default=4)
    parser.add_argument("--include-diagnostics", action="store_true")
    return parser.parse_args()


def source_paths(path: Path, limit: int) -> list[Path]:
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    if limit > 0:
        rows = rows[:limit]
    paths = []
    for row in rows:
        raw = row.get("before") or row.get("input") or row.get("source")
        if raw and Path(raw).exists():
            paths.append(Path(raw))
    if not paths:
        raise ValueError(f"No source images found in {path}")
    return paths


def load_rgb(path: Path, max_side: int) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max_side > 0:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path, "PNG")


def diagnostic_sources(output_root: Path) -> list[Path]:
    """Create synthetic originals that expose halation behavior clearly."""
    source_dir = output_root / "_diagnostic_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save_image(image: Image.Image, name: str) -> None:
        path = source_dir / name
        image.save(path, "PNG")
        paths.append(path)

    font = ImageFont.load_default()
    img = Image.new("RGB", (900, 520), (5, 6, 8))
    draw = ImageDraw.Draw(img)
    for x, y, radius, color in [
        (170, 160, 34, (255, 250, 230)),
        (430, 130, 18, (255, 120, 45)),
        (670, 190, 52, (255, 255, 255)),
    ]:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw.text((95, 330), "POINT LIGHTS ON DARK BACKGROUND", fill=(255, 255, 255), font=font)
    save_image(img, "01_point_lights_dark.png")

    img = Image.new("RGB", (900, 520), (8, 8, 10))
    draw = ImageDraw.Draw(img)
    draw.rectangle((110, 145, 790, 265), fill=(245, 245, 230))
    draw.rectangle((130, 165, 770, 245), fill=(18, 20, 24))
    draw.text((168, 190), "WHITE SIGN / BLACK GAP / HIGH CONTRAST", fill=(255, 245, 215), font=font)
    save_image(img, "02_white_sign_black_gap.png")

    img = Image.new("RGB", (900, 520), (22, 16, 14))
    draw = ImageDraw.Draw(img)
    for i in range(6):
        x0 = 90 + i * 120
        draw.rectangle((x0, 150, x0 + 72, 260), fill=(255, 85 + i * 22, 35))
        draw.rectangle((x0 + 12, 170, x0 + 60, 240), fill=(255, 236, 180))
    draw.text((90, 330), "WARM NEON / ORANGE CORE CHECK", fill=(255, 210, 170), font=font)
    save_image(img, "03_warm_neon_blocks.png")

    grad = np.linspace(0.08, 0.92, 900, dtype=np.float32)[None, :, None]
    arr = np.repeat(grad, 520, axis=0)
    rgb = np.concatenate([arr, arr, arr * 1.02], axis=2)
    img = Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse((155, 200, 245, 290), fill=(255, 255, 248))
    draw.ellipse((640, 200, 730, 290), fill=(255, 255, 248))
    draw.text((105, 330), "same light on dark side", fill=(255, 255, 255), font=font)
    draw.text((590, 330), "same light on bright side", fill=(10, 10, 10), font=font)
    save_image(img, "04_background_suppression.png")
    return paths


def layer_view(layer) -> np.ndarray:
    alpha = layer.alpha
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    return np.clip(layer.rgb * alpha, 0.0, 1.0)


def make_contact_sheet(rows: list[dict], output: Path, title: str) -> None:
    font = ImageFont.load_default()
    tiles = []
    for row in rows:
        original = Image.open(row["original"]).convert("RGB")
        layer = Image.open(row["layer"]).convert("RGB")
        combined = Image.open(row["combined"]).convert("RGB")
        for image in (original, layer, combined):
            image.thumbnail((230, 170), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (690, 210), "white")
        draw = ImageDraw.Draw(tile)
        label = (
            f"{row['id']} alpha_max={row['alpha_max']:.4f} "
            f"visible={row['visible_affected_percent']:.2f}% bounds={row['min']}..{row['max']}"
        )
        draw.text((8, 8), label, fill="black", font=font)
        tile.paste(original, ((230 - original.width) // 2, 34))
        tile.paste(layer, (230 + (230 - layer.width) // 2, 34))
        tile.paste(combined, (460 + (230 - combined.width) // 2, 34))
        draw.text((8, 188), "original", fill="black", font=font)
        draw.text((238, 188), "halation layer on black", fill="black", font=font)
        draw.text((468, 188), "combined", fill="black", font=font)
        tiles.append(tile)
    sheet = Image.new("RGB", (690, 34 + len(tiles) * 210), "white")
    ImageDraw.Draw(sheet).text((8, 10), title, fill="black", font=font)
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (0, 34 + index * 210))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "PNG")


def run_configurations(args: argparse.Namespace) -> list[dict]:
    if args.runs_json:
        return json.loads(args.runs_json.read_text(encoding="utf-8"))
    return DEFAULT_RUNS


def main() -> int:
    args = parse_args()
    paths = source_paths(args.source_manifest, args.limit)
    output_root = args.output_root.resolve()
    if args.include_diagnostics:
        paths = diagnostic_sources(output_root) + paths
    all_summary = {"runs": []}
    for config in run_configurations(args):
        run_name = config["name"]
        run_dir = output_root / run_name
        rows = []
        for index, path in enumerate(paths, start=1):
            base = load_rgb(path, args.max_side)
            layer = physical_halation_layer(
                base,
                profile=config["profile"],
                amplify=float(config["amplify"]),
                impact=float(config["impact"]),
                source_limiter_stops=float(config["source_limiter_stops"]),
                local_diffusion=float(config["local_diffusion"]),
                global_diffusion=float(config["global_diffusion"]),
                hue_green=float(config["hue_green"]),
                background_gain=float(config["background_gain"]),
                no_remjet=float(config["no_remjet"]),
            )
            combined = composite_layers(base, [layer], output_margin=args.output_margin)
            view = layer_view(layer)
            original_path = run_dir / "original" / f"{index:02d}_original.png"
            layer_path = run_dir / "layers" / f"{index:02d}_halation_layer.png"
            combined_path = run_dir / "combined" / f"{index:02d}_combined.png"
            save_rgb(base, original_path)
            save_rgb(view, layer_path)
            save_rgb(combined, combined_path)
            combined_u8 = np.rint(combined * 255.0).astype(np.uint8)
            metrics = layer_metrics(layer)
            alpha = layer.alpha
            if alpha.ndim == 3:
                alpha = alpha[..., 0]
            row = {
                "id": f"{index:02d}",
                "source": str(path),
                "original": str(original_path),
                "layer": str(layer_path),
                "combined": str(combined_path),
                "min": int(combined_u8.min()),
                "max": int(combined_u8.max()),
                "alpha_max": float(metrics.get("alpha_max", 0.0)),
                "alpha_mean": float(metrics.get("alpha_mean", 0.0)),
                "affected_percent": float(metrics.get("affected_percent", 0.0)),
                "visible_affected_percent": float((alpha > 0.01).mean() * 100.0),
            }
            rows.append(row)
            print(
                f"{run_name} {index:02d} alpha_max={row['alpha_max']:.4f} "
                f"visible={row['visible_affected_percent']:.2f}% bounds={row['min']}..{row['max']}"
            )
        metrics_summary = {
            "config": config,
            "image_count": len(rows),
            "alpha_max_max": max(row["alpha_max"] for row in rows),
            "alpha_mean_mean": mean(row["alpha_mean"] for row in rows),
            "affected_percent_mean": mean(row["affected_percent"] for row in rows),
            "visible_affected_percent_mean": mean(row["visible_affected_percent"] for row in rows),
            "bounds_min": min(row["min"] for row in rows),
            "bounds_max": max(row["max"] for row in rows),
            "images": rows,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
        make_contact_sheet(rows, run_dir / "contact_sheet.png", f"Physical Halation V2 - {run_name}")
        all_summary["runs"].append(metrics_summary)
        print(run_dir / "contact_sheet.png")
    (output_root / "summary.json").write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
