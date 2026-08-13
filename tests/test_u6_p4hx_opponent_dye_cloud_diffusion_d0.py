from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.opponent_dye_cloud_diffusion_d0 import (
    _diffuse_opponent_density,
    _validate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4hx_opponent_dye_cloud_diffusion_d0_v1.json"


def test_contract_is_frozen_and_parent_bound() -> None:
    contract = load_contract(CONFIG)
    _validate(contract)
    for binding in contract["parents"].values():
        assert (ROOT / binding["path"]).is_file()


def test_common_density_is_exactly_preserved() -> None:
    base = np.full((17, 19, 3), 0.5, dtype=np.float64)
    y, x = np.mgrid[:17, :19]
    delta = np.stack(
        (0.01 * np.sin(x), 0.01 * np.cos(y), 0.006 * np.sin(x + y)), axis=2
    )
    control = base * np.power(10.0, -delta)
    candidate, error = _diffuse_opponent_density(
        base, control, sigmas=[0.72, 1.08, 1.52], truncate=4.0
    )
    assert error <= 1e-12
    observed = -np.log10(candidate.astype(np.float64) / base)
    expected = np.mean(delta, axis=2)
    assert np.max(np.abs(np.mean(observed, axis=2) - expected)) < 1e-6


def test_mechanism_and_gate_drift_reject() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(contract)
    changed["mechanism"]["dye_diffusion_sigma_pixels_cmy"][0] = 0.73
    with pytest.raises(ValueError, match="mechanism drift"):
        _validate(changed)
    changed = copy.deepcopy(contract)
    changed["gates"][
        "maximum_candidate_to_control_high_frequency_chroma_p999_ratio"
    ] = 0.81
    with pytest.raises(ValueError, match="mechanism drift"):
        _validate(changed)
