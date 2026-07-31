from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.physical_covariance_moment_lod import CovarianceMomentLodError, evaluate_covariance_moment_lod, load_contract
from src.film_physics.derivative_conditioned_structure import gaussian_block_mean_variance_scale


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4ah_covariance_moment_lod_v1.json"


def test_gaussian_block_mean_variance_scale_boundaries() -> None:
    assert gaussian_block_mean_variance_scale(0.0, 2) == 0.25
    assert gaussian_block_mean_variance_scale(0.65, 1) == pytest.approx(1.0)
    factor2 = gaussian_block_mean_variance_scale(0.65, 2)
    factor4 = gaussian_block_mean_variance_scale(0.65, 4)
    assert 0.25 < factor2 < 1.0
    assert 0.0625 < factor4 < factor2
    with pytest.raises(ValueError):
        gaussian_block_mean_variance_scale(-1.0, 2)


def test_contract_mutation_fails_closed(tmp_path: Path) -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    value["gates"]["maximum_variance_ratio"] = 2.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(CovarianceMomentLodError, match="contract drift"):
        load_contract(path)


def test_frozen_evaluation_is_repeat_exact() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_covariance_moment_lod(contract, ROOT)
    second = evaluate_covariance_moment_lod(contract, ROOT)
    assert first == second
    assert first["passed"] is all(first["checks"].values())
