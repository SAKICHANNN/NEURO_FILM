from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.fivek_projection_curve_casebank import (
    fit_projection_curve_bank,
    projection_curve_fit_eligible,
)


ROOT = Path(__file__).resolve().parents[1]


def test_projection_curve_bank_uses_the_mature_fit_contract() -> None:
    mature = json.loads(
        (ROOT / "configs/u5_r2bn1_fivek_projection_curve_basis_development_v1.json").read_text(
            encoding="utf-8"
        )
    )
    axis = np.linspace(0.05, 0.95, 128, dtype=np.float64)
    source = np.stack(
        np.meshgrid(axis, axis, [0.5], indexing="ij"), axis=-1
    )[:, :, 0, :]
    target = np.clip(source * np.asarray([1.02, 0.98, 1.0]), 0.0, 1.0)
    bank = fit_projection_curve_bank(
        [{"pair_id": "synthetic", "source": source, "target": target}],
        mature["operator"],
    )
    assert bank.shape == (1, 16, 9, 3)
    assert np.all(np.isfinite(bank))
    eligible, count = projection_curve_fit_eligible(source, mature["operator"])
    assert eligible is True
    assert count >= 16 * 9
