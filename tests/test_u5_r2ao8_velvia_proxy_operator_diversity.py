from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2ao8_velvia_proxy_operator_diversity import (
    CONFIG_SHA256,
    load_config,
)
from src.eval.velvia_proxy_operator_diversity import (
    VelviaOperatorDiversityError,
    evaluate_operator_diversity,
    residual_alignment,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/u5_r2ao8_velvia_proxy_operator_diversity_v1.json"
)


def _config() -> dict:
    return load_config(CONFIG_PATH, expected_sha256=CONFIG_SHA256)


def test_frozen_contract_loads_three_exact_operators() -> None:
    operators = validate_contract(ROOT, _config())
    assert set(operators) == {"chart", "palette", "combined"}


def test_contract_rejects_gate_drift() -> None:
    changed = copy.deepcopy(_config())
    changed["gates"]["minimum_nonbasic_chart_palette_delta_e76"] = 0.99
    with pytest.raises(VelviaOperatorDiversityError, match="contract"):
        validate_contract(ROOT, changed)


def test_config_hash_fails_closed() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(CONFIG_PATH, expected_sha256="0" * 64)


def test_strength_only_alignment_is_not_a_second_direction() -> None:
    residual = np.linspace(-0.2, 0.3, 300, dtype=np.float64).reshape(100, 3)
    result = residual_alignment(
        residual,
        0.5 * residual,
        minimum_scale=0.0,
        maximum_scale=2.0,
    )
    assert result["scale"] == pytest.approx(0.5, abs=1e-15)
    assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-15)
    assert result["target_residual_fraction"] <= 1e-15


def test_exact_evaluation_is_deterministic_and_controlled() -> None:
    first = evaluate_operator_diversity(ROOT, _config())
    second = evaluate_operator_diversity(ROOT, _config())
    assert json.dumps(first, sort_keys=True) == json.dumps(
        second, sort_keys=True
    )
    assert first["grid_rows"] == 4913
    assert all(
        row["passed"] for row in first["strength_only_negative_controls"]
    )
    assert set(first["checks"]) == {
        "ao4_cross_domain_evidence",
        "finite_and_in_cube",
        "strength_only_negative_controls",
        "median_perceptual_disagreement",
        "nonbasic_disagreement",
        "strength_aligned_direction_diversity",
        "cosine_direction_diversity",
    }
