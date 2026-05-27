#!/usr/bin/env python3
"""Smoke-test film-effect layer compositing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx import FilmLayer, composite_layers, layer_metrics  # noqa: E402


def main() -> int:
    x = np.linspace(0.05, 0.95, 96, dtype=np.float32)
    y = np.linspace(0.08, 0.92, 64, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    base = np.stack([xx, yy, 0.45 + xx * 0.2], axis=2)
    warm = np.zeros_like(base)
    warm[..., 0] = 1.0
    warm[..., 1] = 0.35
    alpha = np.clip((xx - 0.55) * 0.25, 0.0, 0.08)[..., None]
    residual = np.zeros_like(base)
    residual[..., 2] = -0.015
    layers = [
        FilmLayer(name="warm_screen", mode="screen", rgb=warm, alpha=alpha),
        FilmLayer(name="blue_residual", mode="residual", residual=residual),
    ]
    out = composite_layers(base, layers, output_margin=4)
    arr = np.rint(out * 255.0).astype(np.uint8)
    if arr.min() < 4 or arr.max() > 251:
        raise AssertionError(f"bounds failed: {arr.min()}..{arr.max()}")
    out_dir = ROOT / "outputs" / "filmfx" / "layer_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(base * 255.0).astype(np.uint8), mode="RGB").save(out_dir / "base.png", "PNG")
    Image.fromarray(arr, mode="RGB").save(out_dir / "composite.png", "PNG")
    print({"bounds": [int(arr.min()), int(arr.max())], "layers": [layer_metrics(layer) for layer in layers]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
