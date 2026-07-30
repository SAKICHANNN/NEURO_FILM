from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.fivek_neutral_base_fixed_ao6_ablation import (
    FiveKFixedAO6AblationError,
    build_fixed_ao6_renderer,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs/u5_r2ay1_fivek_neutral_base_fixed_ao6_ablation_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_contract_binds_passing_neutral_base_and_exact_ao6() -> None:
    validated = validate_contract(ROOT, _config())
    assert validated["neutral_report"]["automatic_pass"] is True
    assert validated["ao6_config"]["base"]["candidate_id"] == (
        "density_then_anchor__density_s50"
    )


def test_renderer_is_finite_bounded_and_repeat_exact() -> None:
    config = _config()
    validated = validate_contract(ROOT, config)
    renderer = build_fixed_ao6_renderer(config, validated)
    source = np.linspace(0.0, 1.0, 9 * 11 * 3, dtype=np.float64).reshape(
        9, 11, 3
    )
    first, first_diagnostics = renderer(source)
    second, second_diagnostics = renderer(source)
    assert np.array_equal(first, second)
    assert first_diagnostics == second_diagnostics
    assert np.all(np.isfinite(first))
    assert np.min(first) >= 0.0
    assert np.max(first) <= 1.0


def test_contract_rejects_style_strength_change() -> None:
    config = _config()
    config["fixed_look"]["tone_strength"] = 0.16
    with pytest.raises(FiveKFixedAO6AblationError):
        validate_contract(ROOT, config)
