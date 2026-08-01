from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.fivek_strict_interior_lut_casebank import (
    _apply_bank,
    fit_strict_interior_lut_bank,
)


ROOT = Path(__file__).resolve().parents[1]


def test_strict_interior_lut_bank_reuses_mature_fit_and_stays_bounded() -> None:
    mature = json.loads(
        (
            ROOT
            / "configs/u5_r2bj2_fivek_strict_interior_lut_basis_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    axis = np.linspace(0.02, 0.98, 16, dtype=np.float64)
    source = np.stack(np.meshgrid(axis, axis, [0.5], indexing="ij"), axis=-1)[
        :, :, 0, :
    ]
    target = np.clip(source * np.asarray([1.08, 0.94, 1.02]), 0.0, 1.0)
    bank = fit_strict_interior_lut_bank(
        [{"pair_id": "synthetic", "source": source, "target": target}],
        mature["operator"],
    )
    outputs = _apply_bank(source.reshape(-1, 3), bank)
    assert bank.shape == (1, 4, 4, 4, 3)
    assert outputs.shape == (1, source.size // 3, 3)
    assert np.all(np.isfinite(bank))
    assert np.min(outputs) >= 0.0
    assert np.max(outputs) <= 1.0
