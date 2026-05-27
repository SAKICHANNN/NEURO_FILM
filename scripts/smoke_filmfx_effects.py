#!/usr/bin/env python3
"""Smoke-test deterministic film-effect layers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx import composite_layers, layer_metrics  # noqa: E402
from src.filmfx.effects import dust_scratch_layer, grain_residual_layer, halation_layer  # noqa: E402


def save(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path, "PNG")


def main() -> int:
    x = np.linspace(0.02, 0.98, 180, dtype=np.float32)
    y = np.linspace(0.04, 0.96, 120, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    base = np.stack([xx, yy, 0.18 + xx * 0.55], axis=2)
    base[25:48, 115:145] = 1.0
    layers = [
        grain_residual_layer(base, strength=0.012, seed=3),
        halation_layer(base, strength=0.14),
        dust_scratch_layer(base.shape, strength=0.18, seed=5),
    ]
    composite = composite_layers(base, layers, output_margin=4)
    arr = np.rint(composite * 255.0).astype(np.uint8)
    if arr.min() < 4 or arr.max() > 251:
        raise AssertionError(f"bounds failed: {arr.min()}..{arr.max()}")
    metrics = {"bounds": [int(arr.min()), int(arr.max())], "layers": [layer_metrics(layer) for layer in layers]}
    out_dir = ROOT / "outputs" / "filmfx" / "effects_smoke"
    save(base, out_dir / "base.png")
    save(composite, out_dir / "composite.png")
    for layer in layers:
        if layer.mode == "residual":
            view = np.clip(0.5 + layer.residual * 8.0, 0.0, 1.0)
        else:
            view = layer.rgb * layer.alpha
        save(view, out_dir / f"{layer.name}.png")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
