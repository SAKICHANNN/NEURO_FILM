#!/usr/bin/env python3
"""Apply deterministic film-effect layers to an image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx import FilmLayer, composite_layers, layer_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Composite film-effect layers.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--profile-config", type=Path, default=ROOT / "configs" / "filmfx_profiles.yaml")
    parser.add_argument("--profile", default="clean")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-layers", action="store_true")
    parser.add_argument("--metrics", type=Path, default=None)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path, "PNG")


def build_layers(profile: dict, shape: tuple[int, int, int]) -> list[FilmLayer]:
    height, width, _ = shape
    layers = []
    for item in profile.get("layers", []):
        color = np.asarray(item.get("color", [0.0, 0.0, 0.0]), dtype=np.float32)
        rgb = np.broadcast_to(color.reshape(1, 1, 3), shape).copy()
        alpha = np.full((height, width, 1), float(item.get("alpha", 0.0)), dtype=np.float32)
        layers.append(FilmLayer(name=item["name"], mode=item["mode"], rgb=rgb, alpha=alpha))
    return layers


def main() -> int:
    args = parse_args()
    doc = OmegaConf.to_container(OmegaConf.load(args.profile_config), resolve=True)
    profile = doc.get("profiles", {}).get(args.profile)
    if profile is None:
        raise ValueError(f"Profile {args.profile!r} not found in {args.profile_config}")
    base = load_rgb(args.input)
    layers = build_layers(profile, base.shape)
    out = composite_layers(base, layers, output_margin=int(profile.get("output_margin", 0)))
    save_rgb(out, args.output)

    if args.write_layers:
        layer_dir = args.output.parent / f"{args.output.stem}_layers"
        for layer in layers:
            if layer.rgb is not None and layer.alpha is not None:
                save_rgb(layer.rgb * layer.alpha, layer_dir / f"{layer.name}.png")
    metrics = {"profile": args.profile, "layers": [layer_metrics(layer) for layer in layers]}
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
