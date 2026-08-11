from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.global_chroma_procrustes import (
    _plane_basis,
    global_chroma_procrustes_target,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_cb20_recovers_one_global_chroma_rotation() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb20_global_chroma_procrustes_v1.json"
    )
    assert contract["experiment_id"] == "U5.R2CB20"
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    basis = _plane_basis(weights)
    rng = np.random.default_rng(2020)
    luma = rng.uniform(0.35, 0.65, size=(13, 17))
    xy = rng.uniform(-0.08, 0.08, size=(13, 17, 2))
    angle = np.deg2rad(23.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    base = np.asarray(luma[..., None] + xy @ basis.T, dtype=np.float32)
    ao6 = np.asarray(luma[..., None] + (xy @ rotation.T) @ basis.T, dtype=np.float32)
    target = global_chroma_procrustes_target(base, ao6, weights=weights)
    assert float(np.max(np.abs(target - ao6))) <= 2e-7
