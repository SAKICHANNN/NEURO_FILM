#!/usr/bin/env python3
"""Build pseudo-target image sets for Neural Film LUT V2 experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
STYLE_NAMES = ["ektar_100", "portra_400", "portra_800", "velvia_50", "vision3_250d", "vision3_500t"]
TARGET_STRENGTHS = [
    ("s0p0_identity", 0.0, "identity", 0.0),
    ("s0p35_safe", 0.35, "safe_rich", 0.0),
    ("s0p70_soft_film", 0.70, "blend_safe_film", 0.65),
    ("s1p0_film", 1.0, "film_response", 1.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Neural Film LUT V2 pseudo-targets.")
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
    parser.add_argument("--styles", default=",".join(STYLE_NAMES))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "neural_film_lut_v2" / "targets_v1")
    parser.add_argument("--output-margin", type=int, default=4)
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
    if max_side > 0:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb(rgb: np.ndarray, path: Path, output_margin: int) -> None:
    if output_margin > 0:
        low = output_margin / 255.0
        high = 1.0 - low
        rgb = np.clip(rgb, low, high)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path, "PNG")


def main() -> int:
    args = parse_args()
    styles = [item.strip() for item in args.styles.split(",") if item.strip()]
    paths = source_paths(args.source_manifest, args.limit)
    output_dir = args.output_dir.resolve()
    rows = []
    for image_index, input_path in enumerate(paths, start=1):
        source = load_rgb(input_path, args.max_side)
        for style in styles:
            safe_path = args.teacher_root / style / "safe_rich" / f"{image_index:02d}_{style}_safe_rich.png"
            film_path = args.teacher_root / style / "after" / f"{image_index:02d}_{style}_film_response.png"
            if not safe_path.exists() or not film_path.exists():
                continue
            safe = load_rgb(safe_path, args.max_side)
            film = load_rgb(film_path, args.max_side)
            height = min(source.shape[0], safe.shape[0], film.shape[0])
            width = min(source.shape[1], safe.shape[1], film.shape[1])
            source_crop = source[:height, :width]
            safe_crop = safe[:height, :width]
            film_crop = film[:height, :width]
            for label, strength, target_kind, film_mix in TARGET_STRENGTHS:
                if target_kind == "identity":
                    target = source_crop
                elif target_kind == "safe_rich":
                    target = safe_crop
                elif target_kind == "film_response":
                    target = film_crop
                else:
                    target = safe_crop * (1.0 - film_mix) + film_crop * film_mix
                target_path = output_dir / style / label / f"{image_index:02d}_{style}_{label}.png"
                save_rgb(target, target_path, args.output_margin)
                rows.append(
                    {
                        "image_index": image_index,
                        "style": style,
                        "strength": strength,
                        "target_kind": target_kind,
                        "source": str(input_path),
                        "target": str(target_path),
                    }
                )
    manifest_path = output_dir / "targets_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "image_count": len(paths),
                "row_count": len(rows),
                "styles": styles,
                "target_strengths": [
                    {"label": label, "strength": strength, "target_kind": target_kind, "film_mix": film_mix}
                    for label, strength, target_kind, film_mix in TARGET_STRENGTHS
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
