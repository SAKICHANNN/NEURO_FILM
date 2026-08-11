from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.neutral_confident_ao6_direction import (
    load_contract,
    neutral_confident_direction_target,
)

ROOT = Path(__file__).resolve().parents[1]


def test_cb18_neutral_limit_and_chroma_magnitude() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb18_neutral_confident_ao6_direction_v1.json"
    )
    assert contract["experiment_id"] == "U5.R2CB18"
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    base = np.full((5, 7, 3), 0.4, dtype=np.float32)
    ao6 = np.empty_like(base)
    ao6[..., 0] = 0.8
    ao6[..., 1] = 0.2
    ao6[..., 2] = 0.5
    target = neutral_confident_direction_target(
        base,
        ao6,
        weights=weights,
        boundary_epsilon=1.0 / 65535.0,
    )
    assert np.array_equal(target, base)

    rng = np.random.default_rng(1818)
    base = rng.uniform(0.1, 0.9, size=(13, 17, 3)).astype(np.float32)
    ao6 = rng.uniform(0.1, 0.9, size=base.shape).astype(np.float32)
    target = neutral_confident_direction_target(
        base,
        ao6,
        weights=weights,
        boundary_epsilon=1.0 / 65535.0,
    )
    base64 = base.astype(np.float64)
    target64 = target.astype(np.float64)
    base_luma = np.sum(base64 * weights, axis=-1)
    target_luma = np.sum(target64 * weights, axis=-1)
    base_norm = np.linalg.norm(base64 - base_luma[..., None], axis=-1)
    target_norm = np.linalg.norm(target64 - target_luma[..., None], axis=-1)
    assert float(np.max(np.abs(target_luma - base_luma))) <= 1e-7
    assert float(np.max(np.abs(target_norm - base_norm))) <= 1e-7
