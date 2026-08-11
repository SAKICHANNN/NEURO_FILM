from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.spatially_coherent_ao6_direction import (
    KERNEL,
    load_contract,
    spatially_coherent_direction_target,
)

ROOT = Path(__file__).resolve().parents[1]


def test_cb19_kernel_and_target_invariants() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb19_spatially_coherent_ao6_direction_v1.json"
    )
    assert contract["experiment_id"] == "U5.R2CB19"
    assert np.array_equal(KERNEL, np.asarray([0.25, 0.5, 0.25]))
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    rng = np.random.default_rng(1919)
    base = rng.uniform(0.1, 0.9, size=(13, 17, 3)).astype(np.float32)
    ao6 = rng.uniform(0.1, 0.9, size=base.shape).astype(np.float32)
    target = spatially_coherent_direction_target(base, ao6, weights=weights)
    base64 = base.astype(np.float64)
    target64 = target.astype(np.float64)
    base_luma = np.sum(base64 * weights, axis=-1)
    target_luma = np.sum(target64 * weights, axis=-1)
    base_norm = np.linalg.norm(base64 - base_luma[..., None], axis=-1)
    target_norm = np.linalg.norm(target64 - target_luma[..., None], axis=-1)
    assert float(np.max(np.abs(target_luma - base_luma))) <= 1e-7
    assert float(np.max(np.abs(target_norm - base_norm))) <= 1e-7
