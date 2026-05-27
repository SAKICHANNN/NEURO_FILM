#!/usr/bin/env python3
"""Smoke-test differentiable Neural LUT components."""

from __future__ import annotations

import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.color_lut import BasisLUT, TinyLUTEncoder, apply_lut, identity_lut  # noqa: E402


def main() -> int:
    torch.manual_seed(7)
    image = torch.rand(2, 3, 24, 32, requires_grad=True)
    lut = identity_lut(17, dtype=image.dtype)
    identity_out = apply_lut(image, lut)
    max_error = (identity_out - image).abs().max().item()
    if max_error > 1e-5:
        raise AssertionError(f"identity LUT max error too high: {max_error}")

    basis = BasisLUT(size=17, num_basis=3)
    encoder = TinyLUTEncoder(num_basis=3, style_count=8)
    style_index = torch.tensor([0, 5], dtype=torch.long)
    weights = encoder(image, style_index)
    batch_lut = basis(weights)
    out = apply_lut(image, batch_lut)
    loss = out.mean()
    loss.backward()

    if image.grad is None or not torch.isfinite(image.grad).all():
        raise AssertionError("image gradient missing or invalid")
    if basis.basis.grad is None or not torch.isfinite(basis.basis.grad).all():
        raise AssertionError("basis gradient missing or invalid")

    print(
        {
            "identity_max_error": max_error,
            "weights_shape": list(weights.shape),
            "batch_lut_shape": list(batch_lut.shape),
            "output_shape": list(out.shape),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
