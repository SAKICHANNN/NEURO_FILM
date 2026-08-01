from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_measured_mtf_budget import (
    MeasuredMtfBudgetError,
    evaluate_budget,
    validate_contract,
)
from src.film_physics.measured_mtf_budget import MeasuredTotalFilmSpatialBudget

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p5m_measured_mtf_spatial_budget_v1.json"
LOD = ROOT / "outputs/u6_p5l_measured_mtf_lod_compiler/bundle_run1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_contract_rejects_policy_drift() -> None:
    config = _config()
    validate_contract(config)
    config["ownership_policy"]["scanner_mtf_is_separate_downstream_nuisance"] = False
    with pytest.raises(MeasuredMtfBudgetError):
        validate_contract(config)


def test_budget_maps_named_channels_to_rgb_and_rejects_double_count() -> None:
    payload = json.loads(LOD.read_text(encoding="utf-8"))
    budget = MeasuredTotalFilmSpatialBudget.from_lod_bundle(payload)
    impulse = np.zeros((31, 47, 3), dtype=np.float64)
    impulse[15, 23] = 1.0
    output = budget.apply(impulse)
    assert output.shape == impulse.shape
    assert not np.array_equal(output[..., 0], output[..., 2])
    budget.validate_active_stages(["scanner_mtf"])
    with pytest.raises(ValueError, match="forward_scatter"):
        budget.validate_active_stages(["forward_scatter", "scanner_mtf"])


def test_evaluation_is_exact_and_opens_only_residual_compilation(
    tmp_path: Path,
) -> None:
    first = evaluate_budget(
        root=ROOT, config=_config(), diagnostic_path=tmp_path / "first.png"
    )
    second = evaluate_budget(
        root=ROOT, config=_config(), diagnostic_path=tmp_path / "second.png"
    )
    assert first == second
    assert first["automatic_pass"] is True
    assert first["decision"] == "open_positive_residual_compiler"
    assert first["positive_residual_necessary_condition"] is True
    assert not first["residual_above_unity_channels"]
    assert first["residual_below_unity_channels"]
    assert all(first["row_partition_exact"].values())
    assert (tmp_path / "first.png").read_bytes() == (
        tmp_path / "second.png"
    ).read_bytes()


def test_parent_hash_drift_fails_closed(tmp_path: Path) -> None:
    config = _config()
    config["parents"]["p5l_bundle_sha256"] = "0" * 64
    with pytest.raises(MeasuredMtfBudgetError, match="parent hash mismatch"):
        evaluate_budget(root=ROOT, config=config, diagnostic_path=tmp_path / "x.png")
