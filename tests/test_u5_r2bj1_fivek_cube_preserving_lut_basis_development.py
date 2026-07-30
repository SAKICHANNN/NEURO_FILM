from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.fivek_cube_preserving_lut_basis_development import (
    FiveKCubeLUTError,
    apply_cube_preserving_lut,
    fit_cube_preserving_lut,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2bj1_fivek_cube_preserving_lut_basis_development_v1.json"
)


def test_contract_is_parent_bound_and_fail_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert len(validate_contract(ROOT, config)["populations"]) == 3
    drifted = json.loads(CONFIG.read_text(encoding="utf-8"))
    drifted["operator"]["node_coefficient_maximum"] = 0.3
    with pytest.raises(FiveKCubeLUTError):
        validate_contract(ROOT, drifted)


def test_cube_proof_holds_for_extreme_nodes_and_inputs() -> None:
    rng = np.random.default_rng(20260731)
    source = rng.random((257, 263, 3), dtype=np.float64)
    source[0, :3] = np.eye(3)
    for coefficient in (-0.25, 0.25):
        lut = np.full((4, 4, 4, 3), coefficient)
        output = apply_cube_preserving_lut(source, lut)
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)
        assert np.array_equal(output[0, :3], source[0, :3])


def test_fit_improves_known_enveloped_transform() -> None:
    rng = np.random.default_rng(83)
    source = rng.random((96, 80, 3), dtype=np.float64)
    coefficient = np.asarray([0.12, -0.08, 0.05])
    target = source + 4.0 * source * (1.0 - source) * coefficient
    fitted = fit_cube_preserving_lut(
        source,
        target,
        grid_size=4,
        sample_stride=2,
        identity_shrinkage=0.01,
        smoothness=0.1,
        coefficient_minimum=-0.25,
        coefficient_maximum=0.25,
    )
    before = float(np.sqrt(np.mean((source - target) ** 2)))
    after = float(
        np.sqrt(
            np.mean(
                (apply_cube_preserving_lut(source, fitted) - target) ** 2
            )
        )
    )
    assert after < before * 0.05
