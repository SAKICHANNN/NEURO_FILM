#!/usr/bin/env python3
"""Run a low-strength SDXL LoRA img2img grid for content-preservation checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from pipeline import FILM_STYLES, get_device, load_pipeline, translate


ROOT = Path(__file__).resolve().parents[1]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search low-strength SDXL LoRA img2img settings.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--style", choices=list(FILM_STYLES) + ["all"], default="portra_400")
    parser.add_argument("--strengths", default="0.10,0.18,0.25,0.35")
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "sdxl_lora_grid")
    return parser.parse_args()


def fit_image(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    if max(width, height) <= max_side:
        return image
    scale = max_side / max(width, height)
    new_width = max(8, int(width * scale) // 8 * 8)
    new_height = max(8, int(height * scale) // 8 * 8)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def slug_float(value: float) -> str:
    return str(value).replace(".", "p")


def write_contact_sheet(style: str, rows: list[dict], output_dir: Path) -> Path:
    font = ImageFont.load_default()
    thumbs: list[Image.Image] = []
    for row in rows:
        image = Image.open(row["output"]).convert("RGB")
        image.thumbnail((256, 256), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (256, 286), "white")
        canvas.paste(image, ((256 - image.width) // 2, 0))
        label = f"strength={row['strength']} cfg={row['guidance']}"
        ImageDraw.Draw(canvas).text((8, 264), label, fill="black", font=font)
        thumbs.append(canvas)

    columns = min(4, len(thumbs))
    rows_count = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 256, rows_count * 286 + 28), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), style, fill="black", font=font)
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % columns) * 256, 28 + (idx // columns) * 286))

    path = output_dir / "contact_sheets" / f"{style}_contact_sheet.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)
    return path


def main() -> int:
    args = parse_args()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    strengths = parse_float_list(args.strengths)
    styles = list(FILM_STYLES) if args.style == "all" else [args.style]

    if args.device == "auto":
        device, dtype = get_device()
    else:
        device = torch.device(args.device)
        dtype = torch.float16 if args.device == "cuda" else torch.float32

    image = fit_image(Image.open(image_path).convert("RGB"), args.max_side)
    pipe = load_pipeline(device, dtype)

    rows: list[dict] = []
    try:
        for style in styles:
            style_rows: list[dict] = []
            for strength in strengths:
                generator = torch.Generator(device=device).manual_seed(args.seed)
                torch.manual_seed(args.seed)
                result = translate(
                    pipe,
                    image,
                    style,
                    strength=strength,
                    steps=args.steps,
                    guidance=args.guidance,
                    lora_scale=args.lora_scale,
                )
                output_path = output_dir / (
                    f"{image_path.stem}_{style}_s{slug_float(strength)}_g{slug_float(args.guidance)}.jpg"
                )
                result.save(output_path, "JPEG", quality=95)
                row = {
                    "input": str(image_path),
                    "style": style,
                    "strength": strength,
                    "guidance": args.guidance,
                    "steps": args.steps,
                    "lora_scale": args.lora_scale,
                    "seed": args.seed,
                    "output": str(output_path),
                }
                style_rows.append(row)
                rows.append(row)
                del generator
            write_contact_sheet(style, style_rows, output_dir)
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = output_dir / f"{image_path.stem}_sdxl_lora_grid_manifest.csv"
    json_path = output_dir / f"{image_path.stem}_sdxl_lora_grid_manifest.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} outputs")
    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
