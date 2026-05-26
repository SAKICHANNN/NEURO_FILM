#!/usr/bin/env python3
"""Grid search InstructPix2Pix guidance parameters for film translation.

Example:
  python scripts/grid_search_ip2p.py --image test.jpg --style portra_400
  python scripts/grid_search_ip2p.py --image test.jpg --style all --dry-run
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

STYLE_PROMPTS = {
    "ektar_100": "Apply a Kodak Ektar 100 film look with vivid saturated colors, crisp contrast, and ultra fine grain.",
    "hp5": "Apply an Ilford HP5 Plus black and white film look with classic grain and rich tonality.",
    "portra_400": "Apply a Kodak Portra 400 film look with warm natural skin tones, soft contrast, and fine grain.",
    "portra_800": "Apply a Kodak Portra 800 film look with warm colors, moderate grain, and natural contrast.",
    "tri_x_400": "Apply a Kodak Tri-X 400 black and white film look with strong contrast and classic grain.",
    "velvia_50": "Apply a Fujifilm Velvia 50 slide film look with vivid saturated colors, deep shadows, and fine grain.",
    "vision3_250d": "Apply a Kodak Vision3 250D cinematic daylight film look with soft highlight rolloff and fine grain.",
    "vision3_500t": "Apply a Kodak Vision3 500T cinematic tungsten film look with warm shadows and visible film grain.",
}


def parse_csv_floats(value: str) -> list[float]:
    try:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IP2P parameter grid search on one image.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--style", default="portra_400", choices=list(STYLE_PROMPTS) + ["all"])
    parser.add_argument("--model", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ip2p_grid")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--image-guidance", type=parse_csv_floats, default=parse_csv_floats("1.0,1.5,2.0"))
    parser.add_argument("--guidance", type=parse_csv_floats, default=parse_csv_floats("5.0,7.5,10.0"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> tuple[torch.device, torch.dtype]:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.float16
        if torch.backends.mps.is_available():
            return torch.device("mps"), torch.float16
        return torch.device("cpu"), torch.float32
    device = torch.device(device_arg)
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    return device, dtype


def resize_for_ip2p(image: Image.Image, max_side: int = 512) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    if max(width, height) <= max_side:
        return image
    scale = max_side / max(width, height)
    new_width = max(8, int(width * scale) // 8 * 8)
    new_height = max(8, int(height * scale) // 8 * 8)
    return image.resize((new_width, new_height), Image.LANCZOS)


def load_pipeline(model: str, device: torch.device, dtype: torch.dtype, local_files_only: bool):
    from diffusers import StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model,
        torch_dtype=dtype,
        safety_checker=None,
        local_files_only=local_files_only,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main() -> int:
    args = parse_args()
    image_path = args.image if args.image.is_absolute() else ROOT / args.image
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    styles = sorted(STYLE_PROMPTS) if args.style == "all" else [args.style]
    combinations = [
        (style, image_guidance, guidance)
        for style in styles
        for image_guidance in args.image_guidance
        for guidance in args.guidance
    ]
    print(f"Planned runs: {len(combinations)}")
    for style, image_guidance, guidance in combinations:
        print(f"{style}: image_guidance={image_guidance:g}, guidance={guidance:g}")
    if args.dry_run:
        return 0

    device, dtype = resolve_device(args.device)
    image = resize_for_ip2p(Image.open(image_path), args.max_side)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args.model, device, dtype, args.local_files_only)
    manifest_rows = []

    try:
        for style, image_guidance, guidance in combinations:
            generator = torch.Generator(device=device.type).manual_seed(args.seed)
            prompt = STYLE_PROMPTS[style]
            stem = (
                f"{image_path.stem}_{style}_ig{image_guidance:g}_g{guidance:g}"
                .replace(".", "p")
                .replace(" ", "_")
            )
            output_path = output_dir / f"{stem}.jpg"
            start = time.time()
            result = pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=args.steps,
                image_guidance_scale=image_guidance,
                guidance_scale=guidance,
                generator=generator,
            ).images[0]
            elapsed = time.time() - start
            result.save(output_path, "JPEG", quality=95)
            row = {
                "input": str(image_path),
                "style": style,
                "prompt": prompt,
                "image_guidance_scale": image_guidance,
                "guidance_scale": guidance,
                "steps": args.steps,
                "max_side": args.max_side,
                "seed": args.seed,
                "output": str(output_path),
                "seconds": round(elapsed, 3),
            }
            manifest_rows.append(row)
            print(f"Saved {output_path} ({elapsed:.1f}s)")
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    csv_path = output_dir / f"{image_path.stem}_grid_manifest.csv"
    json_path = output_dir / f"{image_path.stem}_grid_manifest.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    json_path.write_text(json.dumps(manifest_rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Manifest: {csv_path}")
    print(f"Manifest: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
