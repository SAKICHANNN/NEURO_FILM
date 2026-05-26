#!/usr/bin/env python3
"""Content-safe deterministic film color baseline.

This pipeline never uses a generative model. It transfers robust Lab color
statistics from the film-domain dataset and optionally adds light grain.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.color import lab2rgb, rgb2lab


ROOT = Path(__file__).resolve().parents[1]
B_AND_W_STYLES = {"hp5", "tri_x_400"}


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply deterministic film color transfer.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--style", default="portra_800")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--stats", type=Path, default=ROOT / "configs" / "film_color_stats.json")
    parser.add_argument("--strength", type=float, default=0.55)
    parser.add_argument("--strengths", default=None, help="Comma-separated grid, e.g. 0.35,0.55,0.75")
    parser.add_argument("--luma-strength", type=float, default=0.35)
    parser.add_argument("--grain", type=float, default=0.012)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "color_baseline")
    parser.add_argument("--contact-sheet", action="store_true")
    return parser.parse_args()


def slug_float(value: float) -> str:
    return str(value).replace(".", "p")


def load_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def style_transfer(
    image: Image.Image,
    stats: dict,
    style: str,
    strength: float,
    luma_strength: float,
    grain: float,
    seed: int,
) -> Image.Image:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    lab = rgb2lab(rgb)
    src_mean = lab.reshape(-1, 3).mean(axis=0)
    src_std = np.maximum(lab.reshape(-1, 3).std(axis=0), 1e-3)
    dst_mean = np.asarray(stats["mean"], dtype=np.float32)
    dst_std = np.asarray(stats["std"], dtype=np.float32)

    transferred = (lab - src_mean) / src_std * dst_std + dst_mean
    out = lab.copy()
    out[..., 0] = lab[..., 0] + luma_strength * strength * (transferred[..., 0] - lab[..., 0])
    out[..., 1:] = lab[..., 1:] + strength * (transferred[..., 1:] - lab[..., 1:])

    if style in B_AND_W_STYLES:
        gray_strength = min(1.0, strength * 1.35)
        out[..., 1:] *= 1.0 - gray_strength
        contrast = 1.0 + 0.30 * strength
        out[..., 0] = np.clip((out[..., 0] - 50.0) * contrast + 50.0, 0.0, 100.0)

    out[..., 0] = np.clip(out[..., 0], 0.0, 100.0)
    out[..., 1:] = np.clip(out[..., 1:], -128.0, 127.0)
    result = lab2rgb(out)

    if grain > 0:
        rng = np.random.default_rng(seed)
        luminance = result.mean(axis=2, keepdims=True)
        noise_scale = grain * (0.55 + 0.9 * (1.0 - luminance))
        noise = rng.normal(0.0, noise_scale, size=result.shape).astype(np.float32)
        result = np.clip(result + noise, 0.0, 1.0)

    return Image.fromarray(np.clip(result * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def write_contact_sheet(rows: list[dict], output_dir: Path, label: str) -> Path:
    font = ImageFont.load_default()
    thumbs = []
    for row in rows:
        image = Image.open(row["output"]).convert("RGB")
        image.thumbnail((256, 256), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (256, 286), "white")
        canvas.paste(image, ((256 - image.width) // 2, 0))
        ImageDraw.Draw(canvas).text((8, 264), row["label"], fill="black", font=font)
        thumbs.append(canvas)

    cols = min(4, len(thumbs))
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 256, rows_count * 286 + 28), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), label, fill="black", font=font)
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 256, 28 + (idx // cols) * 286))
    path = output_dir / "contact_sheets" / f"{label}_contact_sheet.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)
    return path


def main() -> int:
    args = parse_args()
    stats_doc = json.loads(args.stats.read_text(encoding="utf-8"))
    styles = sorted(stats_doc["styles"]) if args.all else [args.style]
    strengths = parse_float_list(args.strengths) if args.strengths else [args.strength]
    image = load_image(args.input)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for style in styles:
        if style not in stats_doc["styles"]:
            raise ValueError(f"Style not found in stats: {style}")
        style_rows = []
        for strength in strengths:
            out = style_transfer(
                image,
                stats_doc["styles"][style],
                style,
                strength=strength,
                luma_strength=args.luma_strength,
                grain=args.grain,
                seed=args.seed,
            )
            if args.output and len(styles) == 1 and len(strengths) == 1:
                output_path = args.output
            else:
                output_path = output_dir / f"{args.input.stem}_{style}_s{slug_float(strength)}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out.save(output_path, "JPEG", quality=95)
            row = {
                "input": str(args.input.resolve()),
                "style": style,
                "strength": strength,
                "luma_strength": args.luma_strength,
                "grain": args.grain,
                "output": str(output_path),
                "label": f"{style} s={strength}",
            }
            rows.append(row)
            style_rows.append(row)
            print(output_path)
        if args.contact_sheet and len(style_rows) > 1:
            write_contact_sheet(style_rows, output_dir, style)

    if len(rows) > 1:
        csv_path = output_dir / f"{args.input.stem}_color_baseline_manifest.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output_dir / f"{args.input.stem}_color_baseline_manifest.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
