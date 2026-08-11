from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.safe_base_ao6_chroma_direction import ao6_direction_target, load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_cb17_direction_target_preserves_base_luma_and_chroma_norm() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb17_safe_base_ao6_chroma_direction_v1.json"
    )
    assert contract["experiment_id"] == "U5.R2CB17"
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    rng = np.random.default_rng(1717)
    base = rng.uniform(0.1, 0.9, size=(17, 19, 3)).astype(np.float32)
    ao6 = rng.uniform(0.1, 0.9, size=base.shape).astype(np.float32)
    target = ao6_direction_target(base, ao6, weights=weights)
    base64 = base.astype(np.float64)
    target64 = target.astype(np.float64)
    base_luma = np.sum(base64 * weights, axis=-1)
    target_luma = np.sum(target64 * weights, axis=-1)
    base_norm = np.linalg.norm(base64 - base_luma[..., None], axis=-1)
    target_norm = np.linalg.norm(target64 - target_luma[..., None], axis=-1)
    assert float(np.max(np.abs(target_luma - base_luma))) <= 1e-7
    assert float(np.max(np.abs(target_norm - base_norm))) <= 1e-7
