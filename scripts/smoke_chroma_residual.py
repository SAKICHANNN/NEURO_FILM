#!/usr/bin/env python3
"""Smoke-test Lab chroma residual rendering."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.chroma_residual import bounded_chroma_residual, constant_residual  # noqa: E402


def main() -> int:
    x = np.linspace(0.05, 0.95, 96, dtype=np.float32)
    y = np.linspace(0.10, 0.90, 64, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    rgb = np.stack([xx, yy, 0.55 + xx * 0.15], axis=2).astype(np.float32)
    before_lab = rgb2lab(rgb)
    delta = constant_residual(rgb.shape[:2], delta_a=7.0, delta_b=-5.0)
    out = bounded_chroma_residual(rgb, delta, max_delta=8.0, output_margin=4)
    after_lab = rgb2lab(out)
    l_delta = float(np.abs(after_lab[..., 0] - before_lab[..., 0]).mean())
    arr = np.rint(out * 255.0).astype(np.uint8)
    if arr.min() < 4 or arr.max() > 251:
        raise AssertionError(f"output bounds failed: {arr.min()}..{arr.max()}")
    if l_delta > 0.35:
        raise AssertionError(f"Lab L changed too much: {l_delta}")
    out_dir = ROOT / "outputs" / "chroma_residual" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), mode="RGB").save(out_dir / "before.png", "PNG")
    Image.fromarray(arr, mode="RGB").save(out_dir / "after.png", "PNG")
    print({"mean_abs_l_delta": l_delta, "bounds": [int(arr.min()), int(arr.max())]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
