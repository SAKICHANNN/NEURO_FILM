#!/usr/bin/env python3
"""Build per-film Lab color statistics for deterministic color transfer."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-style film color statistics in Lab space.")
    parser.add_argument("--film-root", type=Path, default=ROOT / "data" / "film_domain")
    parser.add_argument("--output", type=Path, default=ROOT / "configs" / "film_color_stats.json")
    parser.add_argument("--max-images", type=int, default=120)
    parser.add_argument("--max-side", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_image(path: Path, max_side: int) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def robust_stats(values: np.ndarray) -> dict:
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": np.maximum(values.std(axis=0), 1e-3).tolist(),
        "p01": np.percentile(values, 1, axis=0).tolist(),
        "p50": np.percentile(values, 50, axis=0).tolist(),
        "p99": np.percentile(values, 99, axis=0).tolist(),
    }


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    film_root = args.film_root.resolve()
    output = args.output.resolve()
    result = {
        "schema_version": 1,
        "color_space": "CIELAB",
        "source": str(film_root),
        "max_images_per_style": args.max_images,
        "max_side": args.max_side,
        "styles": {},
    }

    for style_dir in sorted(path for path in film_root.iterdir() if path.is_dir()):
        images = sorted(path for path in style_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            continue
        rng.shuffle(images)
        images = images[: args.max_images]
        pixels: list[np.ndarray] = []
        failed = 0
        for path in images:
            try:
                lab = rgb2lab(load_image(path, args.max_side))
            except Exception:
                failed += 1
                continue
            pixels.append(lab.reshape(-1, 3))
        if not pixels:
            continue
        values = np.concatenate(pixels, axis=0)
        result["styles"][style_dir.name] = {
            "image_count": len(pixels),
            "failed_count": failed,
            **robust_stats(values),
        }
        print(f"{style_dir.name}: images={len(pixels)} failed={failed}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
