from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2ao1_velvia_chart_operator_explainability import (
    CONFIG_SHA256,
    load_config,
)
from src.real_film.velvia_chart_explainability import (
    evaluate_chart_proxy,
    load_exact_pairs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ao1_velvia_chart_operator_explainability_v1.json"


def test_frozen_ao1_contract_is_strictly_development_only() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert config["split"]["row_count"] == 6
    assert config["split"]["columns_per_row"] == 4
    assert config["source"]["working_space"] == "linear_srgb_d65"
    assert not config["stock_response_claim_allowed"]
    assert not config["calibrated_reference_claim_allowed"]
    assert not config["production_integration_allowed"]


def test_exact_pair_loader_rejects_unpinned_json(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    path = tmp_path / "pairs.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON hash"):
        load_exact_pairs(path, config)


def test_grouped_evaluator_has_six_disjoint_folds() -> None:
    config = deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    config["models"]["restart_count"] = 1
    config["models"]["maximum_function_evaluations"] = 400
    values = np.linspace(0.02, 0.92, 72, dtype=np.float64).reshape(24, 3)
    target = np.clip(0.82 * values + 0.04, 0.0, 1.0)
    result = evaluate_chart_proxy(values, target, config)
    assert len(result["folds"]) == 6
    seen: list[int] = []
    for fold in result["folds"]:
        assert len(fold["development_indices"]) == 20
        assert len(fold["confirmation_indices"]) == 4
        assert not set(fold["development_indices"]) & set(
            fold["confirmation_indices"]
        )
        seen.extend(fold["confirmation_indices"])
    assert sorted(seen) == list(range(24))
    assert result["aggregate"]["identity"]["mean_confirmation_rgb_rmse"] > 0.0
